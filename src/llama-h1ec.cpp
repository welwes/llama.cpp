#include "llama-h1ec.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>
#include <memory>

static size_t h1ec_align(size_t off) {
    return (off + 63) & ~(size_t) 63;
}

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
        /*.mem_size   =*/ (size_t) n_layer * 8 * ggml_tensor_overhead(),
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
        ggml_format_name(l.slot_map, "h1ec.%d.slot_map", il);
        ggml_format_name(l.cpu_map,  "h1ec.%d.cpu_map",  il);
        ggml_format_name(l.mask,     "h1ec.%d.mask",     il);

        l.enabled = true;
        n_enabled++;
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

    // slot_map/mask — в тот же VRAM-буфер; cpu_map — в CPU-память (это делает
    // get_rows по нему CPU-узлом и режет GPU-сплит для перекрытия, см. llama-graph.cpp)
    std::vector<pending_t> pend_cpu;
    size_t off_cpu = 0;
    for (auto & l : layers) {
        if (!l.enabled) {
            continue;
        }
        for (ggml_tensor * t : { l.slot_map, l.mask }) {
            off = h1ec_align(off);
            pend.push_back({ t, off });
            off += ggml_nbytes(t);
        }
        off_cpu = h1ec_align(off_cpu);
        pend_cpu.push_back({ l.cpu_map, off_cpu });
        off_cpu += ggml_nbytes(l.cpu_map);
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

    char * base     = (char *) ggml_backend_buffer_get_base(buf.get());
    char * base_cpu = (char *) ggml_backend_buffer_get_base(buf_cpu.get());
    for (auto * pv : { &pend, &pend_cpu }) {
        for (const auto & p : *pv) {
            char * b = pv == &pend ? base : base_cpu;
            ggml_backend_buffer_t bb = pv == &pend ? buf.get() : buf_cpu.get();
            if (ggml_backend_tensor_alloc(bb, p.t, b + p.off) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: tensor alloc failed for %s\n", __func__, p.t->name);
                layers.clear();
                buf.reset();
                buf_cpu.reset();
                return false;
            }
        }
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
        for (int e = 0; e < n_expert; e++) {
            l.h_slot[e] = l.n_slots + e; // всё промах
            l.h_cpu [e] = e;
        }
        push_maps(l);
    }

    LLAMA_LOG_INFO("%s: %d slots x %d layers, pool %.2f MiB on %s\n",
            __func__, n_slots, n_enabled, off / 1024.0 / 1024.0, ggml_backend_dev_name(dev));
    return true;
}

void llama_h1ec::push_maps(const llama_h1ec_layer & l) {
    ggml_backend_tensor_set(l.slot_map, l.h_slot.data(), 0, l.h_slot.size() * sizeof(int32_t));
    ggml_backend_tensor_set(l.cpu_map,  l.h_cpu.data(),  0, l.h_cpu.size()  * sizeof(int32_t));
    ggml_backend_tensor_set(l.mask,     l.h_mask.data(), 0, l.h_mask.size() * sizeof(float));
}

void llama_h1ec::set_bypass(bool on) {
    std::vector<int32_t> ident(n_expert);
    std::vector<float>   zeros(n_expert, 0.0f);
    for (int e = 0; e < n_expert; e++) {
        ident[e] = e;
    }
    for (auto & l : layers) {
        if (!l.enabled) {
            continue;
        }
        if (on) {
            ggml_backend_tensor_set(l.cpu_map, ident.data(), 0, ident.size() * sizeof(int32_t));
            ggml_backend_tensor_set(l.mask,    zeros.data(), 0, zeros.size() * sizeof(float));
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
        l.h_cpu [old] = old;
        l.h_mask[old] = 0.0f;
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
        l.slot_eid[slot] = eid;
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
