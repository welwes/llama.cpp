#include "llama-h1ec.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>
#include <memory>

static size_t h1ec_align(size_t off) {
    return (off + 63) & ~(size_t) 63;
}

bool llama_h1ec::init(const llama_model & model, int32_t n_slots_arg) {
    if (n_slots_arg <= 0 || model.hparams.n_expert == 0) {
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

    n_slots  = n_slots_arg;
    n_expert = (int32_t) model.hparams.n_expert;

    const int n_layer = (int) model.layers.size();
    layers.resize(n_layer);

    struct ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) n_layer * 8 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(ip));

    // раскладка пула: кэш-тензоры внахлёст (шаг = n_slots столбцов, вид на
    // n_slots+n_expert), затем общий хвост n_expert * max(размер эксперта)
    struct pending_t {
        ggml_tensor * t;
        size_t        off;
    };
    std::vector<pending_t> pend;

    size_t off = 0;
    size_t max_nb2 = 0;
    int n_enabled = 0;

    for (int il = 0; il < n_layer; il++) {
        const auto & src_l = model.layers[il];
        auto & l = layers[il];

        ggml_tensor * srcs[3] = { src_l.ffn_up_exps, src_l.ffn_gate_exps, src_l.ffn_down_exps };
        if (!srcs[0] || !srcs[1] || !srcs[2]) {
            continue; // не-MoE слой
        }
        if (srcs[0]->ne[2] != n_expert || srcs[1]->ne[2] != n_expert || srcs[2]->ne[2] != n_expert) {
            continue;
        }

        ggml_tensor ** dsts[3] = { &l.cache_up, &l.cache_gate, &l.cache_down };
        const char * names[3] = { "up", "gate", "down" };
        for (int k = 0; k < 3; k++) {
            ggml_tensor * src = srcs[k];
            ggml_tensor * t = ggml_new_tensor_3d(ctx.get(), src->type, src->ne[0], src->ne[1], n_slots + n_expert);
            ggml_format_name(t, "h1ec.%d.%s", il, names[k]);
            GGML_ASSERT(t->nb[2] == src->nb[2]); // одинаковый тип/размер среза эксперта
            off = h1ec_align(off);
            pend.push_back({ t, off });
            off += (size_t) n_slots * t->nb[2]; // шаг БЕЗ хвоста — хвост внахлёст
            max_nb2 = std::max(max_nb2, (size_t) t->nb[2]);
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

    off = h1ec_align(off) + (size_t) n_expert * max_nb2; // общий мусорный хвост

    // карты — в тот же буфер, после хвоста
    for (auto & l : layers) {
        if (!l.enabled) {
            continue;
        }
        for (ggml_tensor * t : { l.slot_map, l.cpu_map, l.mask }) {
            off = h1ec_align(off);
            pend.push_back({ t, off });
            off += ggml_nbytes(t);
        }
    }

    buf.reset(ggml_backend_buft_alloc_buffer(buft, off));
    if (!buf) {
        LLAMA_LOG_ERROR("%s: failed to allocate %.2f MiB on %s\n", __func__, off / 1024.0 / 1024.0, ggml_backend_dev_name(dev));
        layers.clear();
        return false;
    }

    char * base = (char *) ggml_backend_buffer_get_base(buf.get());
    for (const auto & p : pend) {
        if (ggml_backend_tensor_alloc(buf.get(), p.t, base + p.off) != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: tensor alloc failed for %s\n", __func__, p.t->name);
            layers.clear();
            buf.reset();
            return false;
        }
    }

    // нули: любой квант из нулевых байтов декодируется в 0 (случайный мусор мог бы дать NaN)
    ggml_backend_buffer_clear(buf.get(), 0);

    for (auto & l : layers) {
        if (!l.enabled) {
            continue;
        }
        l.slot_eid.assign(n_slots, -1);
        l.h_slot.resize(n_expert);
        l.h_cpu.resize(n_expert);
        l.h_mask.assign(n_expert, 0.0f);
        for (int e = 0; e < n_expert; e++) {
            l.h_slot[e] = n_slots + e; // всё промах
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

bool llama_h1ec::assign(const llama_model & model, int32_t il, int32_t slot, int32_t eid) {
    if (il < 0 || (size_t) il >= layers.size() || !layers[il].enabled) {
        return false;
    }
    if (slot < 0 || slot >= n_slots || eid >= n_expert) {
        return false;
    }
    auto & l = layers[il];

    if (l.slot_eid[slot] == eid) {
        return true;
    }

    // выселить прежнего жильца слота
    if (const int32_t old = l.slot_eid[slot]; old >= 0) {
        l.h_slot[old] = n_slots + old;
        l.h_cpu [old] = old;
        l.h_mask[old] = 0.0f;
        l.slot_eid[slot] = -1;
    }

    if (eid >= 0) {
        // эксперт уже сидит в другом слоте — освободить его там
        for (int s = 0; s < n_slots; s++) {
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
            staging.resize(nb2);
            ggml_backend_tensor_get(srcs[k], staging.data(), (size_t) eid  * nb2, nb2);
            ggml_backend_tensor_set(dsts[k], staging.data(), (size_t) slot * nb2, nb2);
        }

        l.h_slot[eid] = slot;
        l.h_cpu [eid] = 0;    // попадание: CPU-ветка считает эксперта 0 с весом 0
        l.h_mask[eid] = 1.0f;
        l.slot_eid[slot] = eid;
    }

    push_maps(l);
    return true;
}

//
// public C API
//

#include "llama.h"

bool llama_h1ec_init(struct llama_model * model, int32_t n_slots) {
    if (model == nullptr || model->h1ec) {
        return false;
    }
    auto h1ec = std::make_unique<llama_h1ec>();
    if (!h1ec->init(*model, n_slots)) {
        return false;
    }
    model->h1ec = std::move(h1ec);
    return true;
}

int32_t llama_h1ec_n_slots(const struct llama_model * model) {
    return model && model->h1ec ? model->h1ec->n_slots : 0;
}

bool llama_h1ec_assign(struct llama_model * model, int32_t il, int32_t slot, int32_t expert_id) {
    if (model == nullptr || !model->h1ec) {
        return false;
    }
    return model->h1ec->assign(*model, il, slot, expert_id);
}
