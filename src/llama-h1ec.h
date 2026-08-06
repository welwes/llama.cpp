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

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct llama_model;

struct llama_h1ec_layer {
    int32_t n_slots = 0; // ёмкость слоя (layer-aware бюджет — слоям с плохим reuse больше)

    // [n_embd, n_ff_exp, n_slots + n_expert] — хвост n_expert = мусор для промахов
    ggml_tensor * cache_up   = nullptr;
    ggml_tensor * cache_gate = nullptr;
    ggml_tensor * cache_down = nullptr;

    // --- RAM-ярус (M1.2): pinned-host кэш экспертов, спасает от mmap/SSD ---
    // Хвостов/мусора НЕТ: промахи RAM-ветки идут в id=-1 (CPU-скип строк)
    int32_t ram_slots = 0;
    ggml_tensor * ram_up   = nullptr; // [n_embd, n_ff_exp, ram_slots], pinned host
    ggml_tensor * ram_gate = nullptr;
    ggml_tensor * ram_down = nullptr;
    ggml_tensor * ram_map  = nullptr; // I32 [1,n_expert] CPU: eid -> ram-слот | -1
                                      // (-1 и когда эксперт в VRAM — ярусы эксклюзивны по маскам)
    ggml_tensor * mask_ram = nullptr; // F32 [1,n_expert] VRAM: 1.0 если считает RAM-ярус

    ggml_tensor * slot_map = nullptr; // I32 [1, n_expert] VRAM: eid -> слот | n_slots+eid (промах)
    ggml_tensor * cpu_map  = nullptr; // I32 [1, n_expert] CPU:  eid -> -1 (попадание: строка
                                      // пропускается CPU mul_mat_id без чтения весов) | eid.
                                      // Лежит в CPU-памяти намеренно: get_rows по нему — CPU-узел,
                                      // он режет GPU-сплит и даёт перекрытие CPU/GPU (llama-graph.cpp)
    ggml_tensor * mask     = nullptr; // F32 [1, n_expert] VRAM: 1.0 если eid в кэше

    // host-состояние
    std::vector<int32_t> slot_eid; // слот -> eid | -1
    std::vector<int32_t> h_slot;
    std::vector<int32_t> h_cpu;
    std::vector<float>   h_mask;

    // RAM-ярус
    std::vector<int32_t> ram_slot_eid; // ram-слот -> eid | -1
    std::vector<int32_t> h_ram;        // eid -> ram-слот | -1 (эксклюзивно с VRAM)
    std::vector<float>   h_mask_ram;

    // автопилот: затухающие счётчики использования + стэш выбранных экспертов
    // последнего декода (записывается при построении графа, читается post_decode)
    std::vector<double>   score;
    mutable ggml_tensor * sel_last = nullptr;

    bool enabled = false;
};

struct llama_h1ec {
    int32_t n_slots  = 0; // максимум по слоям (для справки; ёмкость слоя — layers[il].n_slots)
    int32_t n_expert = 0;

    std::vector<llama_h1ec_layer> layers;

    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;     // VRAM-буфер: кэш-пул (внахлёст) + slot_map/mask
    ggml_backend_buffer_ptr buf_cpu; // CPU-буфер: cpu_map всех слоёв

    // собственный бэкенд для асинхронной заливки срезов (отдельный CUDA-стрим);
    // источники — pinned-память модели, копии летят без стейджинга, sync = flush()
    ggml_backend_ptr backend_async;
    bool             dirty = false; // есть незавершённые async-копии
    ggml_backend_buffer_type_t pin_buft = nullptr; // host-pinned buft GPU-девайса:
    // async-копии ТОЛЬКО из него (cudaMemcpyAsync из mmap-страниц Windows =
    // invalid argument; найдено на GLM 04.08 — 30B работал из-за --no-mmap/pinned)

    // --- M1: RAM-ярус + эксперт-блоб ---
    // Память держит buf_ram_pin (CUDA_Host = page-locked, свопу недоступна, H2D
    // быстрый), а ТЕНЗОРЫ живут в buf_ram — CPU-обёртке над теми же страницами:
    // иначе планировщик по типу буфера отдал бы RAM-ветку CUDA, где id=-1 не
    // поддержан (найдено 04.08: NaN с первого декода фазы A)
    ggml_backend_buffer_ptr buf_ram_pin;
    ggml_backend_buffer_ptr buf_ram;
    FILE *  blob = nullptr;          // H1EC_BLOB: эксперт-блоб (h1-blob-pack)
    struct blob_layer {
        uint64_t base = 0, up = 0, gate = 0, down = 0; // офсет и размеры срезов
    };
    std::vector<blob_layer> blob_index; // по слоям модели (base=0 => слоя в блобе нет)

    // дождаться всех async-заливок; ОБЯЗАН случиться до следующего графа
    // (llama_context::decode вызывает сам как страховку)
    void flush();

    // выделяет пул на первом GPU-девайсе модели; false = кэш недоступен.
    // slots_per_layer: ёмкость каждого слоя (0 = слой без кэша); размер = число слоёв модели
    bool init(const llama_model & model, const std::vector<int32_t> & slots_per_layer);

    // положить эксперта eid в слот slot слоя il (eid < 0 = освободить слот);
    // копирует срезы up/gate/down и обновляет карты на GPU.
    // src_data != nullptr — готовые байты up|gate|down (из префетч-стейджинга)
    bool assign(const llama_model & model, int32_t il, int32_t slot, int32_t eid,
                const uint8_t * src_data = nullptr);

    // RAM-ярус: положить эксперта в ram-слот (источник: блоб — 1 seq-read,
    // иначе mmap модели — 3 случайных чтения); маски эксклюзивны с VRAM
    bool assign_ram(const llama_model & model, int32_t il, int32_t slot, int32_t eid,
                    const uint8_t * src_data = nullptr);

    // --- M1.3: префетч — фоновый IO-поток + двухфазный коммит ---
    // request() ставит эксперта в очередь чтения (диск читается ПАРАЛЛЕЛЬНО
    // декоду), commit_ready() в безопасной точке (между декодами) переносит
    // готовые данные в тензоры/карты. Жертву слота выбирает коммит.
    void request(int32_t il, int32_t eid, int tier); // tier: 0 = VRAM, 1 = RAM
    int  commit_ready(const llama_model & model);    // вернёт число закоммиченных

    bool prefetch_on = false; // H1EC_PREFETCH (деф. 1 при наличии блоба)

private:
    struct pf_item {
        int32_t il = -1, eid = -1;
        int     tier = 0;
        std::vector<uint8_t> data; // up|gate|down подряд (байты срезов слоя)
        bool    ok = false;
    };
    std::thread              pf_thread;
    std::mutex               pf_mutex;   // очередь+готовые
    std::condition_variable  pf_cv;
    std::deque<pf_item>      pf_queue;
    std::deque<pf_item>      pf_done;
    bool                     pf_stop = false;
    std::mutex               io_mutex;   // сериализация чтений блоба (FILE* один)
    const llama_model *      pf_model = nullptr; // для mmap-фолбэка в воркере

    void pf_worker();
    bool read_expert(const llama_model & model, int32_t il, int32_t eid, std::vector<uint8_t> & out);

    // профиль горячести: счётчики экспертов переживают перезапуск — прогрев
    // следующего запуска берёт реально горячих, а не «первых N» (важно для
    // моделей > RAM, где онлайн-роллинг занимает сотни токенов)
    std::string profile_path; // H1EC_PROFILE | <блоб>.profile | пусто = выкл
    void load_profile();
    void save_profile();

public:
    // --- автопилот (менеджер в ядре) ---
    // Включается ТОЛЬКО при env-инициализации (H1EC_SLOTS у любого штатного
    // инструмента: llama-cli, llama-server...). Явный вызов llama_h1ec_init*
    // из своего кода оставляет autopilot=false — политикой рулит вызывающий.
    bool    autopilot    = false;
    int32_t update_every = 16; // H1EC_UPDATE_EVERY
    int32_t swap_budget  = 16; // H1EC_SWAPS
    // куда CPU-ветка редиректит попадания: -1 = скип строки (наш патч ggml-cpu),
    // 0 = эксперт 0 с весом 0 (диагностический рубильник H1EC_CPU0=1 — если с ним
    // мусор исчезает, значит в живом графе mmid идёт путём, где id<0 не поддержан)
    int32_t cpu_hit_id   = -1;
    // H1EC_NO_CPU_REMAP=1: CPU-ветка вообще без ремапа (попадания считаются на CPU
    // со своим id и весом 0 — экономии нет, но путь пуленепробиваемый). Если мусор
    // исчезает только с этим — виновата CPU-ветка; если остаётся — GPU-кэш-ветка.
    bool    no_cpu_remap = false;
    int32_t tokens_since_update = 0;
    long long stat_hits = 0, stat_hits_ram = 0, stat_total = 0, stat_swaps = 0;

    // вызывается из llama_context::decode после каждого h1-декода (n_tokens<=8):
    // копит счётчики из sel_last, раз в update_every токенов — затухание и свопы
    void post_decode(const llama_model & model, int32_t n_tokens);

    // обход кэша БЕЗ его потери: on=true пушит нулевые маски + identity cpu_map
    // (всё считается стоковым CPU-путём), on=false возвращает реальные карты.
    // Для численной верификации: логиты с кэшем обязаны совпадать с логитами без
    void set_bypass(bool on);

    ~llama_h1ec();

    const llama_h1ec_layer * get_layer(int il) const {
        if (il < 0 || (size_t) il >= layers.size() || !layers[il].enabled) {
            return nullptr;
        }
        return &layers[il];
    }

private:
    void push_maps(const llama_h1ec_layer & l);
    void open_blob(const llama_model & model, const char * path);
    int  update_layer(const llama_model & model, int32_t il, int32_t budget);
};
