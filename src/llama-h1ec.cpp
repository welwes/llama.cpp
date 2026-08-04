#include "llama-h1ec.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>
#include <memory>

static size_t h1ec_align(size_t off) {
    return (off + 63) & ~(size_t) 63;
}

#ifdef _WIN32
#define h1_fseek64 _fseeki64
#else
#define h1_fseek64 fseeko
#endif

bool llama_h1ec::init(const llama_model & model, const std::vector<int32_t> & slots_per_layer) {
    if (slots_per_layer.size() != model.layers.size() || model.hparams.n_expert == 0) {
        return false;
    }
    int32_t max_slots = 0;
    for (int32_t s : slots_per_layer) {
        max_slots = std::max(max_slots, s);
    }
    if (max_slots <= 0) {
        return false;
    }

    ggml_backend_dev_t dev = nullptr;
    for (const auto & d : model.devices) {
        if (!d.is_meta && d.dev && ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = d.dev;
            break;
        }
    }
    if (dev == nullptr) {
        LLAMA_LOG_WARN("%s: no GPU device in model, h1ec disabled\n", __func__);
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    n_slots  = max_slots;
    n_expert = (int32_t) model.hparams.n_expert;
    if (const char * v = getenv("H1EC_CPU0"); v && atoi(v) != 0) {
        cpu_hit_id = 0; // диагностика: попадания -> эксперт 0 вместо скипа id<0
        LLAMA_LOG_WARN("%s: H1EC_CPU0=1 - hits redirect to expert 0 (no id<0 skip)\n", __func__);
    }
    if (const char * v = getenv("H1EC_NO_CPU_REMAP"); v && atoi(v) != 0) {
        no_cpu_remap = true; // диагностика: CPU-ветка без ремапа, попадания с весом 0
        LLAMA_LOG_WARN("%s: H1EC_NO_CPU_REMAP=1 - CPU branch computes hits too (weight 0)\n", __func__);
    }
    bool fat_pool = false;
    if (const char * v = getenv("H1EC_FAT"); v && atoi(v) != 0) {
        fat_pool = true; // диагностика: раскладка БЕЗ перекрытий (жирный пул — только на малом числе слоёв!)
        LLAMA_LOG_WARN("%s: H1EC_FAT=1 - non-overlapping pool layout\n", __func__);
    }

    const int n_layer = (int) model.layers.size();
    layers.resize(n_layer);

    struct ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) n_layer * 12 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(ip));

    // Раскладка: ПУЛЫ ПО ТИПУ КВАНТА. Хвост «мусорных» столбцов тензора
    // перекрывается со слотами следующих тензоров ТОГО ЖЕ типа: любые валидные
    // блоки этого типа декодируются в конечные числа. Кросс-типовой оверлап
    // ЗАПРЕЩЁН: байты чужого кванта, прочитанные как fp16-скейлы, могут дать
    // INF → по цепочке → INF·0 = NaN → NaN травит всю сеть (найдено 08-03).
    // Хвост каждой типовой зоны — n_expert столбцов, занулён buffer_clear'ом.
    struct pending_t {
        ggml_tensor * t;
        size_t        off;
    };
    std::vector<pending_t> pend;

    int n_enabled = 0;

    for (int il = 0; il < n_layer; il++) {
        const auto & src_l = model.layers[il];
        auto & l = layers[il];

        ggml_tensor * srcs[3] = { src_l.ffn_up_exps, src_l.ffn_gate_exps, src_l.ffn_down_exps };
        if (!srcs[0] || !srcs[1] || !srcs[2] || slots_per_layer[il] <= 0) {
            continue; // не-MoE слой или слой без бюджета
        }
        l.n_slots = slots_per_layer[il];
        if (srcs[0]->ne[2] != n_expert || srcs[1]->ne[2] != n_expert || srcs[2]->ne[2] != n_expert) {
            continue;
        }

        ggml_tensor ** dsts[3] = { &l.cache_up, &l.cache_gate, &l.cache_down };
        const char * names[3] = { "up", "gate", "down" };
        for (int k = 0; k < 3; k++) {
            ggml_tensor * src = srcs[k];
            ggml_tensor * t = ggml_new_tensor_3d(ctx.get(), src->type, src->ne[0], src->ne[1], l.n_slots + n_expert);
            ggml_format_name(t, "h1ec.%d.%s", il, names[k]);
            GGML_ASSERT(t->nb[2] == src->nb[2]); // одинаковый тип/размер среза эксперта
            *dsts[k] = t;
        }

        l.slot_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, n_expert);
        l.cpu_map  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, n_expert);
        l.mask     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        l.mask_ram = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        l.ram_map  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, n_expert);
        ggml_format_name(l.slot_map, "h1ec.%d.slot_map", il);
        ggml_format_name(l.cpu_map,  "h1ec.%d.cpu_map",  il);
        ggml_format_name(l.mask,     "h1ec.%d.mask",     il);
        ggml_format_name(l.mask_ram, "h1ec.%d.mask_ram", il);
        ggml_format_name(l.ram_map,  "h1ec.%d.ram_map",  il);

        l.enabled = true;
        n_enabled++;
    }

    // --- M1.2: RAM-ярус (pinned host) — бюджет из H1EC_RAM_MB ---
    size_t ram_budget = 0;
    if (const char * v = getenv("H1EC_RAM_MB"); v && atoi(v) > 0) {
        ram_budget = (size_t) atoi(v) * 1024ull * 1024ull;
    }
    if (ram_budget > 0 && n_enabled > 0) {
        for (int il = 0; il < n_layer; il++) {
            auto & l = layers[il];
            if (!l.enabled) {
                continue;
            }
            const auto & src_l = model.layers[il];
            const size_t slice = src_l.ffn_up_exps->nb[2] + src_l.ffn_gate_exps->nb[2] + src_l.ffn_down_exps->nb[2];
            l.ram_slots = (int32_t) std::min<size_t>(ram_budget / n_enabled / slice, (size_t) n_expert);
            if (l.ram_slots <= 0) {
                continue;
            }
            l.ram_up   = ggml_new_tensor_3d(ctx.get(), src_l.ffn_up_exps->type,   src_l.ffn_up_exps->ne[0],   src_l.ffn_up_exps->ne[1],   l.ram_slots);
            l.ram_gate = ggml_new_tensor_3d(ctx.get(), src_l.ffn_gate_exps->type, src_l.ffn_gate_exps->ne[0], src_l.ffn_gate_exps->ne[1], l.ram_slots);
            l.ram_down = ggml_new_tensor_3d(ctx.get(), src_l.ffn_down_exps->type, src_l.ffn_down_exps->ne[0], src_l.ffn_down_exps->ne[1], l.ram_slots);
            ggml_format_name(l.ram_up,   "h1ec.%d.ram_up",   il);
            ggml_format_name(l.ram_gate, "h1ec.%d.ram_gate", il);
            ggml_format_name(l.ram_down, "h1ec.%d.ram_down", il);
        }
    }

    if (n_enabled == 0) {
        LLAMA_LOG_WARN("%s: no MoE layers found, h1ec disabled\n", __func__);
        return false;
    }

    // размещение: группируем кэш-тензоры по типу, внутри группы — внахлёст
    size_t off = 0;
    std::vector<ggml_type> group_types;
    for (auto & l : layers) {
        if (!l.enabled) {
            continue;
        }
        for (ggml_tensor * t : { l.cache_up, l.cache_gate, l.cache_down }) {
            if (std::find(group_types.begin(), group_types.end(), t->type) == group_types.end()) {
                group_types.push_back(t->type);
            }
        }
    }
    for (ggml_type gt : group_types) {
        size_t group_max_nb2 = 0;
        for (auto & l : layers) {
            if (!l.enabled) {
                continue;
            }
            for (ggml_tensor * t : { l.cache_up, l.cache_gate, l.cache_down }) {
                if (t->type != gt) {
                    continue;
                }
                off = h1ec_align(off);
                pend.push_back({ t, off });
                off += (size_t) (fat_pool ? l.n_slots + n_expert : l.n_slots) * t->nb[2];
                group_max_nb2 = std::max(group_max_nb2, (size_t) t->nb[2]);
            }
        }
        // хвост зоны: мусорные столбцы последних тензоров группы читают отсюда (нули)
        off = h1ec_align(off) + (size_t) n_expert * group_max_nb2;
    }

    // slot_map/mask/mask_ram — в VRAM-буфер; cpu_map/ram_map — в CPU-память
    // (get_rows по CPU-картам = CPU-узел, режет GPU-сплит для перекрытия);
    // RAM-ярус — в pinned host (свопу недоступен, H2D-промоушены бесплатны)
    std::vector<pending_t> pend_cpu;
    std::vector<pending_t> pend_ram;
    size_t off_cpu = 0;
    size_t off_ram = 0;
    for (auto & l : layers) {
        if (!l.enabled) {
            continue;
        }
        for (ggml_tensor * t : { l.slot_map, l.mask, l.mask_ram }) {
            off = h1ec_align(off);
            pend.push_back({ t, off });
            off += ggml_nbytes(t);
        }
        for (ggml_tensor * t : { l.cpu_map, l.ram_map }) {
            off_cpu = h1ec_align(off_cpu);
            pend_cpu.push_back({ t, off_cpu });
            off_cpu += ggml_nbytes(t);
        }
        if (l.ram_up) {
            for (ggml_tensor * t : { l.ram_up, l.ram_gate, l.ram_down }) {
                off_ram = h1ec_align(off_ram);
                pend_ram.push_back({ t, off_ram });
                off_ram += ggml_nbytes(t);
            }
        }
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu_dev != nullptr);

    buf.reset(ggml_backend_buft_alloc_buffer(buft, off));
    buf_cpu.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(cpu_dev), off_cpu));
    if (!buf || !buf_cpu) {
        LLAMA_LOG_ERROR("%s: failed to allocate %.2f MiB on %s\n", __func__, off / 1024.0 / 1024.0, ggml_backend_dev_name(dev));
        layers.clear();
        return false;
    }
    if (off_ram > 0) {
        // page-locked память берём у GPU-девайса (CUDA_Host), но тензоры заводим в
        // CPU-обёртке над теми же страницами: тип буфера решает, какой бэкенд
        // считает op, а RAM-ветка обязана идти на CPU (id=-1 скип только там)
        ggml_backend_buffer_type_t pin_buft = ggml_backend_dev_host_buffer_type(dev);
        if (pin_buft != nullptr) {
            buf_ram_pin.reset(ggml_backend_buft_alloc_buffer(pin_buft, off_ram));
        }
        if (buf_ram_pin) {
            buf_ram.reset(ggml_backend_cpu_buffer_from_ptr(
                    ggml_backend_buffer_get_base(buf_ram_pin.get()), off_ram));
        } else {
            // фолбэк: обычная CPU-память (может уйти в своп под давлением RAM)
            buf_ram.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(cpu_dev), off_ram));
        }
        if (!buf_ram) {
            LLAMA_LOG_WARN("%s: RAM tier alloc failed (%.0f MiB) — tier disabled\n", __func__, off_ram / 1048576.0);
            for (auto & l : layers) {
                l.ram_slots = 0;
                l.ram_up = l.ram_gate = l.ram_down = nullptr;
            }
            pend_ram.clear();
        } else {
            LLAMA_LOG_INFO("%s: RAM tier %.0f MiB (%s)\n", __func__, off_ram / 1048576.0,
                    buf_ram_pin ? "page-locked" : "pageable");
        }
    }

    char * base     = (char *) ggml_backend_buffer_get_base(buf.get());
    char * base_cpu = (char *) ggml_backend_buffer_get_base(buf_cpu.get());
    char * base_ram = buf_ram ? (char *) ggml_backend_buffer_get_base(buf_ram.get()) : nullptr;
    for (auto * pv : { &pend, &pend_cpu, &pend_ram }) {
        char * b = pv == &pend ? base : pv == &pend_cpu ? base_cpu : base_ram;
        ggml_backend_buffer_t bb = pv == &pend ? buf.get() : pv == &pend_cpu ? buf_cpu.get() : buf_ram.get();
        for (const auto & p : *pv) {
            if (ggml_backend_tensor_alloc(bb, p.t, b + p.off) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: tensor alloc failed for %s\n", __func__, p.t->name);
                layers.clear();
                buf.reset();
                buf_cpu.reset();
                buf_ram.reset();
                return false;
            }
        }
    }

    // ПОМЕТКА WEIGHTS обязательна: только для weight-буферов планировщик прибивает
    // op к бэкенду буфера src0. Без неё листья висят с backend=NULL и sched по
    // эвристике утаскивает CPU-ветки на CUDA с копированием тензоров (NaN + тормоза;
    // найдено по GGML_SCHED_DEBUG-дампу 04.08)
    ggml_backend_buffer_set_usage(buf.get(),     GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(buf_cpu.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (buf_ram) {
        ggml_backend_buffer_set_usage(buf_ram.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // нули: любой квант из нулевых байтов декодируется в 0 (случайный мусор мог бы дать NaN)
    ggml_backend_buffer_clear(buf.get(), 0);

    // отдельный бэкенд (свой CUDA-стрим) для асинхронной заливки срезов
    backend_async.reset(ggml_backend_dev_init(dev, nullptr));

    for (auto & l : layers) {
        if (!l.enabled) {
            continue;
        }
        l.slot_eid.assign(l.n_slots, -1);
        l.h_slot.resize(n_expert);
        l.h_cpu.resize(n_expert);
        l.h_mask.assign(n_expert, 0.0f);
        l.ram_slot_eid.assign(std::max(l.ram_slots, 0), -1);
        l.h_ram.assign(n_expert, -1);
        l.h_mask_ram.assign(n_expert, 0.0f);
        for (int e = 0; e < n_expert; e++) {
            l.h_slot[e] = l.n_slots + e; // всё промах
            l.h_cpu [e] = e;
        }
        push_maps(l);
    }

    // --- M1.1: эксперт-блоб (H1EC_BLOB=path, файл от h1-blob-pack) ---
    if (const char * bp = getenv("H1EC_BLOB"); bp && *bp) {
        open_blob(model, bp);
    }

    LLAMA_LOG_INFO("%s: %d slots x %d layers, pool %.2f MiB on %s\n",
            __func__, n_slots, n_enabled, off / 1024.0 / 1024.0, ggml_backend_dev_name(dev));
    return true;
}

void llama_h1ec::push_maps(const llama_h1ec_layer & l) {
    ggml_backend_tensor_set(l.slot_map, l.h_slot.data(), 0, l.h_slot.size() * sizeof(int32_t));
    ggml_backend_tensor_set(l.cpu_map,  l.h_cpu.data(),  0, l.h_cpu.size()  * sizeof(int32_t));
    ggml_backend_tensor_set(l.mask,     l.h_mask.data(), 0, l.h_mask.size() * sizeof(float));
    // в граф уходит МАСКА-зависимая карта: слот только когда RAM-ярус активен
    // (эксперт в VRAM => -1, чтобы RAM-ветка даже не читала его строки)
    std::vector<int32_t> ram_ids(l.h_ram.size(), -1);
    for (size_t e = 0; e < l.h_ram.size(); e++) {
        if (l.h_mask_ram[e] > 0.0f) {
            ram_ids[e] = l.h_ram[e];
        }
    }
    ggml_backend_tensor_set(l.ram_map,  ram_ids.data(),  0, ram_ids.size()  * sizeof(int32_t));
    ggml_backend_tensor_set(l.mask_ram, l.h_mask_ram.data(), 0, l.h_mask_ram.size() * sizeof(float));
}

void llama_h1ec::open_blob(const llama_model & model, const char * path) {
    blob = fopen(path, "rb");
    if (!blob) {
        LLAMA_LOG_WARN("%s: cannot open blob '%s' — falling back to mmap reads\n", __func__, path);
        return;
    }
    struct {
        char magic[8]; uint32_t version, n_layers, n_expert_, reserved;
    } hdr;
    if (fread(&hdr, sizeof(hdr), 1, blob) != 1 || memcmp(hdr.magic, "H1BLOB1", 7) != 0 ||
        hdr.version != 1 || (int32_t) hdr.n_expert_ != n_expert) {
        LLAMA_LOG_WARN("%s: blob header mismatch — ignored\n", __func__);
        fclose(blob);
        blob = nullptr;
        return;
    }
    blob_index.assign(layers.size(), {});
    int loaded = 0;
    for (uint32_t i = 0; i < hdr.n_layers; i++) {
        // поля читаем по одному — в файле структура упакована (pack(1))
        int32_t  layer;
        uint64_t vals[4]; // base, up, gate, down
        if (fread(&layer, sizeof(layer), 1, blob) != 1 || fread(vals, sizeof(uint64_t), 4, blob) != 4) {
            LLAMA_LOG_WARN("%s: blob index truncated — ignored\n", __func__);
            fclose(blob);
            blob = nullptr;
            blob_index.clear();
            return;
        }
        if (layer < 0 || (size_t) layer >= layers.size()) {
            continue;
        }
        const auto & src_l = model.layers[layer];
        if (!layers[layer].enabled || !src_l.ffn_up_exps) {
            continue;
        }
        // валидация размеров срезов против реальных тензоров
        if (vals[1] != (uint64_t) src_l.ffn_up_exps->nb[2] ||
            vals[2] != (uint64_t) src_l.ffn_gate_exps->nb[2] ||
            vals[3] != (uint64_t) src_l.ffn_down_exps->nb[2]) {
            LLAMA_LOG_WARN("%s: blob slice sizes mismatch at layer %d — ignored\n", __func__, layer);
            fclose(blob);
            blob = nullptr;
            blob_index.clear();
            return;
        }
        blob_index[layer] = { vals[0], vals[1], vals[2], vals[3] };
        loaded++;
    }
    LLAMA_LOG_INFO("%s: expert blob '%s' loaded (%d layers)\n", __func__, path, loaded);
}

void llama_h1ec::set_bypass(bool on) {
    std::vector<int32_t> ident(n_expert);
    std::vector<int32_t> minus1(n_expert, -1);
    std::vector<float>   zeros(n_expert, 0.0f);
    for (int e = 0; e < n_expert; e++) {
        ident[e] = e;
    }
    for (auto & l : layers) {
        if (!l.enabled) {
            continue;
        }
        if (on) {
            ggml_backend_tensor_set(l.cpu_map,  ident.data(),  0, ident.size()  * sizeof(int32_t));
            ggml_backend_tensor_set(l.mask,     zeros.data(),  0, zeros.size()  * sizeof(float));
            ggml_backend_tensor_set(l.ram_map,  minus1.data(), 0, minus1.size() * sizeof(int32_t));
            ggml_backend_tensor_set(l.mask_ram, zeros.data(),  0, zeros.size()  * sizeof(float));
        } else {
            push_maps(l); // вернуть реальные карты
        }
    }
}

bool llama_h1ec::assign(const llama_model & model, int32_t il, int32_t slot, int32_t eid) {
    if (il < 0 || (size_t) il >= layers.size() || !layers[il].enabled) {
        return false;
    }
    auto & l = layers[il];
    if (slot < 0 || slot >= l.n_slots || eid >= n_expert) {
        return false;
    }

    if (l.slot_eid[slot] == eid) {
        return true;
    }

    // выселить прежнего жильца слота
    if (const int32_t old = l.slot_eid[slot]; old >= 0) {
        l.h_slot[old] = l.n_slots + old;
        l.h_mask[old] = 0.0f;
        if (l.h_ram[old] >= 0) {
            // эксперт остался в RAM-ярусе — возвращаем ему RAM-маску
            l.h_mask_ram[old] = 1.0f;
            l.h_cpu[old] = -1; // mmap-ветка его по-прежнему скипает
        } else {
            l.h_cpu[old] = old;
        }
        l.slot_eid[slot] = -1;
    }

    if (eid >= 0) {
        // эксперт уже сидит в другом слоте — освободить его там
        for (int s = 0; s < l.n_slots; s++) {
            if (l.slot_eid[s] == eid) {
                l.slot_eid[s] = -1;
            }
        }

        const auto & src_l = model.layers[il];
        ggml_tensor * srcs[3] = { src_l.ffn_up_exps, src_l.ffn_gate_exps, src_l.ffn_down_exps };
        ggml_tensor * dsts[3] = { l.cache_up, l.cache_gate, l.cache_down };
        std::vector<uint8_t> staging;
        for (int k = 0; k < 3; k++) {
            const size_t nb2 = srcs[k]->nb[2];
            // веса модели в host-памяти (pinned при -ngl) — async H2D прямо из них,
            // без стейджинга; sync один на пачку (flush перед следующим графом)
            if (backend_async && srcs[k]->buffer && ggml_backend_buffer_is_host(srcs[k]->buffer)) {
                ggml_backend_tensor_set_async(backend_async.get(), dsts[k],
                        (const char *) srcs[k]->data + (size_t) eid * nb2, (size_t) slot * nb2, nb2);
                dirty = true;
            } else {
                staging.resize(nb2);
                ggml_backend_tensor_get(srcs[k], staging.data(), (size_t) eid  * nb2, nb2);
                ggml_backend_tensor_set(dsts[k], staging.data(), (size_t) slot * nb2, nb2);
            }
        }

        l.h_slot[eid] = slot;
        if (!no_cpu_remap) {
            l.h_cpu[eid] = cpu_hit_id; // -1 = скип строки | 0 = режим H1EC_CPU0
        } // no_cpu_remap: h_cpu остаётся eid — CPU честно считает попадание с весом 0
        l.h_mask[eid] = 1.0f;
        l.h_mask_ram[eid] = 0.0f; // ярусы эксклюзивны: пока эксперт в VRAM, RAM молчит
        l.slot_eid[slot] = eid;
    }

    push_maps(l);
    return true;
}

bool llama_h1ec::assign_ram(const llama_model & model, int32_t il, int32_t slot, int32_t eid) {
    if (il < 0 || (size_t) il >= layers.size() || !layers[il].enabled) {
        return false;
    }
    auto & l = layers[il];
    if (l.ram_slots <= 0 || slot < 0 || slot >= l.ram_slots || eid >= n_expert) {
        return false;
    }
    if (l.ram_slot_eid[slot] == eid) {
        return true;
    }

    // выселить прежнего жильца ram-слота
    if (const int32_t old = l.ram_slot_eid[slot]; old >= 0) {
        l.h_ram[old] = -1;
        l.h_mask_ram[old] = 0.0f;
        if (l.h_mask[old] <= 0.0f) {
            l.h_cpu[old] = old; // нет ни в одном ярусе — снова считает mmap-ветка
        }
        l.ram_slot_eid[slot] = -1;
    }

    if (eid >= 0) {
        for (int s = 0; s < l.ram_slots; s++) {
            if (l.ram_slot_eid[s] == eid) {
                l.ram_slot_eid[s] = -1;
            }
        }

        const auto & src_l = model.layers[il];
        ggml_tensor * srcs[3] = { src_l.ffn_up_exps, src_l.ffn_gate_exps, src_l.ffn_down_exps };
        ggml_tensor * dsts[3] = { l.ram_up, l.ram_gate, l.ram_down };
        const blob_layer be = (blob && !blob_index.empty()) ? blob_index[il] : blob_layer{};
        if (blob && be.base > 0) {
            // блоб: ОДНО последовательное чтение всего эксперта прямо в host-тензоры
            const uint64_t bsz = be.up + be.gate + be.down;
            if (h1_fseek64(blob, (int64_t) (be.base + (uint64_t) eid * bsz), SEEK_SET) != 0) {
                return false;
            }
            const uint64_t sizes[3] = { be.up, be.gate, be.down };
            for (int k = 0; k < 3; k++) {
                char * dst = (char *) dsts[k]->data + (size_t) slot * dsts[k]->nb[2];
                if (fread(dst, 1, sizes[k], blob) != sizes[k]) {
                    LLAMA_LOG_WARN("%s: blob read failed (layer %d expert %d)\n", __func__, il, eid);
                    return false;
                }
            }
        } else {
            // фолбэк: 3 чтения из mmap модели (под давлением RAM — случайные и дорогие)
            for (int k = 0; k < 3; k++) {
                const size_t nb2 = srcs[k]->nb[2];
                memcpy((char *) dsts[k]->data + (size_t) slot * nb2,
                       (const char *) srcs[k]->data + (size_t) eid * nb2, nb2);
            }
        }

        l.h_ram[eid] = slot;
        l.ram_slot_eid[slot] = eid;
        if (l.h_mask[eid] <= 0.0f) {
            l.h_mask_ram[eid] = 1.0f; // ярус активен только если эксперт не в VRAM
            l.h_cpu[eid] = -1;        // mmap-ветка скипает
        }
    }

    push_maps(l);
    return true;
}

//
// автопилот: менеджер кэша в ядре (порт политики из h1-expert-cache)
//

llama_h1ec::~llama_h1ec() {
    flush();
    if (autopilot && stat_total > 0) {
        LLAMA_LOG_INFO("h1ec: hit rate %.1f%% (%lld/%lld), swaps %lld\n",
                100.0 * stat_hits / stat_total, stat_hits, stat_total, stat_swaps);
    }
}

void llama_h1ec::flush() {
    if (dirty && backend_async) {
        ggml_backend_synchronize(backend_async.get());
        dirty = false;
    }
}

// свопы слоя по счётчикам; гистерезис — кандидат заметно горячее жертвы, иначе пинг-понг
int llama_h1ec::update_layer(const llama_model & model, int32_t il, int32_t budget) {
    auto & l = layers[il];
    if (l.score.empty()) {
        return 0;
    }
    std::vector<int> order(l.score.size());
    for (size_t i = 0; i < order.size(); i++) {
        order[i] = (int) i;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return l.score[a] > l.score[b]; });

    int swaps = 0;
    for (int rank = 0; rank < (int) order.size() && rank < l.n_slots && swaps < budget; rank++) {
        const int eid = order[rank];
        if (l.score[eid] <= 0.0 || l.h_mask[eid] > 0.0f) {
            continue; // мёртвый или уже в кэше
        }
        int victim_slot = -1;
        double victim_score = 1e300;
        for (int s = 0; s < l.n_slots; s++) {
            const int ve = l.slot_eid[s];
            const double vs = ve < 0 ? -1.0 : l.score[ve];
            if (vs < victim_score) {
                victim_score = vs;
                victim_slot  = s;
            }
        }
        if (victim_slot < 0 || l.score[eid] < victim_score * 1.5 + 2.0) {
            break;
        }
        if (!assign(model, il, victim_slot, eid)) {
            break;
        }
        swaps++;
    }

    // RAM-ярус: следующие по рангу эксперты после VRAM-топа (второй эшелон)
    for (int rank = l.n_slots; l.ram_slots > 0 && rank < (int) order.size() &&
            rank < l.n_slots + l.ram_slots && swaps < budget; rank++) {
        const int eid = order[rank];
        if (l.score[eid] <= 0.0 || l.h_ram[eid] >= 0 || l.h_mask[eid] > 0.0f) {
            continue;
        }
        int victim_slot = -1;
        double victim_score = 1e300;
        for (int s = 0; s < l.ram_slots; s++) {
            const int ve = l.ram_slot_eid[s];
            const double vs = ve < 0 ? -1.0 : l.score[ve];
            if (vs < victim_score) {
                victim_score = vs;
                victim_slot  = s;
            }
        }
        if (victim_slot < 0 || l.score[eid] < victim_score * 1.5 + 2.0) {
            break;
        }
        if (!assign_ram(model, il, victim_slot, eid)) {
            break;
        }
        swaps++;
    }
    return swaps;
}

void llama_h1ec::post_decode(const llama_model & model, int32_t n_tokens) {
    if (!autopilot) {
        return;
    }
    std::vector<int32_t> ids;
    for (size_t il = 0; il < layers.size(); il++) {
        auto & l = layers[il];
        if (!l.enabled || l.sel_last == nullptr) {
            continue;
        }
        const int64_t n = ggml_nelements(l.sel_last);
        ids.resize(n);
        ggml_backend_tensor_get(l.sel_last, ids.data(), 0, n * sizeof(int32_t));
        if (l.score.empty()) {
            l.score.assign(n_expert, 0.0);
        }
        for (int64_t i = 0; i < n; i++) {
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert) {
                continue;
            }
            l.score[e] += 1.0;
            stat_total++;
            if (l.h_mask[e] > 0.0f) {
                stat_hits++;
            }
        }
    }

    tokens_since_update += n_tokens;
    if (tokens_since_update < update_every) {
        return;
    }
    tokens_since_update = 0;

    for (auto & l : layers) {
        for (auto & s : l.score) {
            s *= 0.9; // полураспад ~7 апдейтов
        }
    }
    int budget = swap_budget;
    for (size_t il = 0; il < layers.size() && budget > 0; il++) {
        if (!layers[il].enabled) {
            continue;
        }
        const int done = update_layer(model, (int32_t) il, budget);
        budget    -= done;
        stat_swaps += done;
    }
    flush(); // все async-заливки пачки должны сесть до следующего графа
}

//
// public C API
//

#include "llama.h"

bool llama_h1ec_init(struct llama_model * model, int32_t n_slots) {
    if (model == nullptr) {
        return false;
    }
    return llama_h1ec_init_layers(model, nullptr, n_slots);
}

bool llama_h1ec_init_layers(struct llama_model * model, const int32_t * slots_per_layer, int32_t n) {
    if (model == nullptr || model->h1ec) {
        return false;
    }
    std::vector<int32_t> slots;
    if (slots_per_layer == nullptr) {
        slots.assign(model->layers.size(), n); // n = единая ёмкость
    } else {
        if ((size_t) n != model->layers.size()) {
            return false;
        }
        slots.assign(slots_per_layer, slots_per_layer + n);
    }
    auto h1ec = std::make_unique<llama_h1ec>();
    if (!h1ec->init(*model, slots)) {
        return false;
    }
    model->h1ec = std::move(h1ec);
    return true;
}

int32_t llama_h1ec_n_slots(const struct llama_model * model) {
    return model && model->h1ec ? model->h1ec->n_slots : 0;
}

int32_t llama_h1ec_layer_slots(const struct llama_model * model, int32_t il) {
    if (model == nullptr || !model->h1ec || il < 0 || (size_t) il >= model->h1ec->layers.size()) {
        return 0;
    }
    return model->h1ec->layers[il].n_slots;
}

bool llama_h1ec_assign(struct llama_model * model, int32_t il, int32_t slot, int32_t expert_id) {
    if (model == nullptr || !model->h1ec) {
        return false;
    }
    return model->h1ec->assign(*model, il, slot, expert_id);
}

void llama_h1ec_bypass(struct llama_model * model, bool on) {
    if (model && model->h1ec) {
        model->h1ec->set_bypass(on);
    }
}

void llama_h1ec_flush(struct llama_model * model) {
    if (model && model->h1ec) {
        model->h1ec->flush();
    }
}

bool llama_h1ec_assign_ram(struct llama_model * model, int32_t il, int32_t slot, int32_t expert_id) {
    if (model == nullptr || !model->h1ec) {
        return false;
    }
    return model->h1ec->assign_ram(*model, il, slot, expert_id);
}

int32_t llama_h1ec_layer_ram_slots(const struct llama_model * model, int32_t il) {
    if (model == nullptr || !model->h1ec || il < 0 || (size_t) il >= model->h1ec->layers.size()) {
        return 0;
    }
    return model->h1ec->layers[il].ram_slots;
}
