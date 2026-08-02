#pragma once

// H1 expert cache (h1ec) — VRAM-кэш экспертов MoE, схема v3 «сплит суммы с
// масками весов гейта» (см. fork/M2-DESIGN.md в репо adaptive-llm-runtime).
//
// Идея: out = Σ w_i·expert_i(x) разбивается на две ветки mul_mat_id:
// GPU-ветка считает по VRAM-кэшу (промахи уходят в уникальные «мусорные»
// столбцы n_slots+eid и гасятся весом 0), CPU-ветка — по оригинальным
// экспертам (попадания уходят в эксперта 0 и гасятся весом 0; дубли id
// CPU-реализация переваривает, CUDA — НЕТ, поэтому мусорные столбцы).
//
// Кэш-тензоры слоёв — виды внахлёст поверх одного VRAM-буфера: мусорный
// хвост слоя перекрывается со слотами следующих слоёв, доплата за хвост —
// n_expert столбцов на весь пул. Буфер зануляется: нулевые байты любого
// кванта декодируются в 0, случайные могли бы дать NaN (NaN·0=NaN).
//
// Жизненный цикл: llama_h1ec_init(model, n_slots) ПОСЛЕ загрузки модели и
// ДО создания контекста; заполнение llama_h1ec_assign() между llama_decode
// (обновляет только содержимое тензоров, топология графа не меняется).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <vector>

struct llama_model;

struct llama_h1ec_layer {
    // [n_embd, n_ff_exp, n_slots + n_expert] — хвост n_expert = мусор для промахов
    ggml_tensor * cache_up   = nullptr;
    ggml_tensor * cache_gate = nullptr;
    ggml_tensor * cache_down = nullptr;

    ggml_tensor * slot_map = nullptr; // I32 [1, n_expert]: eid -> слот | n_slots+eid (промах)
    ggml_tensor * cpu_map  = nullptr; // I32 [1, n_expert]: eid -> 0 (попадание) | eid
    ggml_tensor * mask     = nullptr; // F32 [1, n_expert]: 1.0 если eid в кэше

    // host-состояние
    std::vector<int32_t> slot_eid; // слот -> eid | -1
    std::vector<int32_t> h_slot;
    std::vector<int32_t> h_cpu;
    std::vector<float>   h_mask;

    bool enabled = false;
};

struct llama_h1ec {
    int32_t n_slots  = 0;
    int32_t n_expert = 0;

    std::vector<llama_h1ec_layer> layers;

    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf; // один VRAM-буфер: кэш-пул (внахлёст) + карты

    // выделяет пул на первом GPU-девайсе модели; false = кэш недоступен
    bool init(const llama_model & model, int32_t n_slots);

    // положить эксперта eid в слот slot слоя il (eid < 0 = освободить слот);
    // копирует срезы up/gate/down и обновляет карты на GPU
    bool assign(const llama_model & model, int32_t il, int32_t slot, int32_t eid);

    const llama_h1ec_layer * get_layer(int il) const {
        if (il < 0 || (size_t) il >= layers.size() || !layers[il].enabled) {
            return nullptr;
        }
        return &layers[il];
    }

private:
    void push_maps(const llama_h1ec_layer & l);
};
