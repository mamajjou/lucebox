// Opt-in Laguna mixed engine.
//
// Prefill and decode want opposite MoE placements on a 24 GiB card:
//   * prefill needs a complete expert stack so mul_mat_id can use the large
//     batched GPU path;
//   * decode benefits from the persistent hot-GPU/cold-CPU split.
//
// This backend keeps a single model and a single KV cache.  For each MoE layer
// during prefill it temporarily releases that layer's persistent hot buffer,
// stages the complete 256-expert layer from the existing GGUF mmap, evaluates
// the prompt in large batches, then restores the calibrated hot subset before
// moving on.  Decode is the existing proven Laguna hybrid path.
//
// Enable with DFLASH_LAGUNA_MIXED_PREFILL=1.  The ordinary Laguna backend is
// unchanged when the flag is absent.

#pragma once

#include "laguna_backend.h"
#include "common/step_graph.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace dflash::common {

class LagunaMixedBackend final : public LagunaBackend {
public:
    explicit LagunaMixedBackend(const LagunaBackendArgs & args)
        : LagunaBackend(args) {}

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override {
        if (!mixed_prefill_enabled() || !hybrid_mode_ || !moe_hybrid_) {
            return LagunaBackend::generate_impl(req, io);
        }

        mixed_restore_ok_ = true;
        GenerateResult result = generate_mixed(req, io);
        if (result.ok()) {
            if (!routing_stats_out_path_.empty() && routing_stats_) {
                std::string serr;
                if (!routing_stats_->save_csv(routing_stats_out_path_, &serr)) {
                    std::fprintf(stderr,
                        "[laguna-mixed] profile save failed: %s\n", serr.c_str());
                } else {
                    std::fprintf(stderr,
                        "[laguna-mixed] profile saved: %s\n",
                        routing_stats_out_path_.c_str());
                }
            }
            maybe_post_request_swap();
            return result;
        }

        // The base path resets KV state before starting, so an explicit
        // fallback is safe only when every staged layer was restored.
        if (mixed_restore_ok_ && mixed_fallback_enabled()) {
            std::fprintf(stderr,
                "[laguna-mixed] staged prefill failed; falling back to hybrid prefill\n");
            return LagunaBackend::generate_impl(req, io);
        }
        return result;
    }

private:
    struct PinnedPlacement {
        std::vector<int32_t> hot_ids;
        int cache_slots = 0;
    };

    bool mixed_restore_ok_ = true;

    static bool env_is_one(const char * name) {
        const char * e = std::getenv(name);
        return e && e[0] == '1' && e[1] == '\0';
    }

    static bool mixed_prefill_enabled() {
        return env_is_one("DFLASH_LAGUNA_MIXED_PREFILL");
    }

    static bool mixed_fallback_enabled() {
        return env_is_one("DFLASH_LAGUNA_MIXED_PREFILL_FALLBACK");
    }

    int mixed_prefill_chunk() const {
        int value = 256;
        if (const char * e = std::getenv("DFLASH_LAGUNA_MIXED_PREFILL_CHUNK")) {
            const int parsed = std::atoi(e);
            if (parsed > 0) value = parsed;
        }
        return std::max(1, std::min(value, std::max(1, args_.chunk)));
    }

    size_t mixed_reserve_bytes() const {
        int mb = 384;
        if (const char * e = std::getenv("DFLASH_LAGUNA_MIXED_RESERVE_MB")) {
            const int parsed = std::atoi(e);
            if (parsed >= 0) mb = parsed;
        }
        return (size_t)mb << 20;
    }

    static bool mixed_verbose() {
        return env_is_one("DFLASH_LAGUNA_MIXED_VERBOSE");
    }

    static void free_hot_graphs(MoeHybridLayerStorage & st) {
        st.hot_graph.free();
        st.hot_batched_graph.free();
        for (auto & graph : st.hot_batched_mixed) graph.free();
        st.shared_batched_graph.free();
    }

    static void release_hot_storage(MoeHybridLayerStorage & st) {
        free_hot_graphs(st);
        if (st.hot_buf) {
            ggml_backend_buffer_free(st.hot_buf);
            st.hot_buf = nullptr;
        }
        if (st.hot_ctx) {
            ggml_free(st.hot_ctx);
            st.hot_ctx = nullptr;
        }
        st.gate_hot = nullptr;
        st.up_hot = nullptr;
        st.down_hot = nullptr;
        st.gate_up_hot = nullptr;
    }

    static ggml_tensor * new_expert_tensor(ggml_context * ctx,
                                            const ggml_tensor * shape,
                                            int expert_count) {
        if (!ctx || !shape || expert_count <= 0) return nullptr;
        const int64_t ne[4] = {
            shape->ne[0], shape->ne[1], expert_count, 1
        };
        return ggml_new_tensor(ctx, shape->type, 4, ne);
    }

    bool upload_expert_tensor(ggml_tensor * dst,
                              const ExpertFileRegion & region,
                              size_t expert_bytes,
                              const std::vector<int32_t> & ids,
                              bool identity_full,
                              std::string & err) const {
        if (!dst || expert_bytes == 0 || ids.empty()) {
            err = "missing staged expert tensor metadata";
            return false;
        }
        if (!moe_hybrid_ || !moe_hybrid_->mmap_data) {
            err = "mixed prefill requires retained GGUF mmap";
            return false;
        }

        const size_t required = expert_bytes * ids.size();
        const size_t full_required = expert_bytes * (size_t)w_.n_expert;
        if (region.offset > moe_hybrid_->mmap_size ||
            full_required > region.size ||
            region.offset + full_required > moe_hybrid_->mmap_size) {
            err = "expert tensor region outside retained GGUF mmap";
            return false;
        }
        if (ggml_nbytes(dst) < required) {
            err = "staged expert tensor allocation is too small";
            return false;
        }

        const auto * src = static_cast<const uint8_t *>(moe_hybrid_->mmap_data)
                        + region.offset;
        if (identity_full && ids.size() == (size_t)w_.n_expert) {
            ggml_backend_tensor_set(dst, src, 0, full_required);
            return true;
        }

        std::vector<uint8_t> packed(required);
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t id = ids[i];
            if (id < 0 || id >= w_.n_expert) {
                err = "expert id outside model range while restoring placement";
                return false;
            }
            std::memcpy(packed.data() + i * expert_bytes,
                        src + (size_t)id * expert_bytes,
                        expert_bytes);
        }
        ggml_backend_tensor_set(dst, packed.data(), 0, packed.size());
        return true;
    }

    bool allocate_hot_storage(MoeHybridLayerStorage & st,
                              const MoeLayerDesc & desc,
                              const LayerExpertRegions & regions,
                              const std::vector<int32_t> & hot_ids,
                              int cache_slots,
                              bool identity_full,
                              std::string & err) {
        const int active = (int)hot_ids.size();
        const int allocated = active + std::max(0, cache_slots);
        if (active <= 0 || allocated <= 0) {
            err = "cannot allocate an empty hot expert stack";
            return false;
        }

        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        st.hot_ctx = ggml_init(ip);
        if (!st.hot_ctx) {
            err = "mixed hot_ctx allocation failed";
            return false;
        }

        if (st.fused_gate_up) {
            st.gate_up_hot = new_expert_tensor(st.hot_ctx, desc.ffn_gate_up_exps,
                                               allocated);
            st.down_hot = new_expert_tensor(st.hot_ctx, desc.ffn_down_exps,
                                            allocated);
        } else {
            st.gate_hot = new_expert_tensor(st.hot_ctx, desc.ffn_gate_exps,
                                            allocated);
            st.up_hot = new_expert_tensor(st.hot_ctx, desc.ffn_up_exps,
                                          allocated);
            st.down_hot = new_expert_tensor(st.hot_ctx, desc.ffn_down_exps,
                                            allocated);
        }

        st.hot_buf = ggml_backend_alloc_ctx_tensors(st.hot_ctx, backend_);
        if (!st.hot_buf) {
            err = "mixed full-layer GPU allocation failed";
            release_hot_storage(st);
            return false;
        }
        ggml_backend_buffer_set_usage(st.hot_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // Spare cache slots must contain valid zero bytes.  Uninitialised quant
        // scales can dequantise to NaN even when their combine weight is zero.
        ggml_backend_buffer_clear(st.hot_buf, 0);

        bool ok = true;
        if (st.fused_gate_up) {
            ok = upload_expert_tensor(st.gate_up_hot, regions.gate_up_exps,
                                      st.gate_up_expert_bytes, hot_ids,
                                      identity_full, err) &&
                 upload_expert_tensor(st.down_hot, regions.down_exps,
                                      st.down_expert_bytes, hot_ids,
                                      identity_full, err);
        } else {
            ok = upload_expert_tensor(st.gate_hot, regions.gate_exps,
                                      st.gate_expert_bytes, hot_ids,
                                      identity_full, err) &&
                 upload_expert_tensor(st.up_hot, regions.up_exps,
                                      st.up_expert_bytes, hot_ids,
                                      identity_full, err) &&
                 upload_expert_tensor(st.down_hot, regions.down_exps,
                                      st.down_expert_bytes, hot_ids,
                                      identity_full, err);
        }
        if (!ok) {
            release_hot_storage(st);
            return false;
        }

        st.hot_expert_ids = hot_ids;
        st.hot_active = active;
        st.cache_slots = std::max(0, cache_slots);
        st.spare_global.assign((size_t)st.cache_slots, -1);
        st.spare_lru.assign((size_t)st.cache_slots, 0);
        st.lru_clock = 0;

        st.hot_local_by_global.assign((size_t)w_.n_expert, -1);
        st.cold_local_by_global.assign((size_t)w_.n_expert, -1);
        std::memset(st.expert_vram_mask, 0, sizeof(st.expert_vram_mask));
        for (size_t i = 0; i < hot_ids.size(); ++i) {
            const int32_t id = hot_ids[i];
            st.hot_local_by_global[(size_t)id] = (int32_t)i;
            if (id >= 0 && id < 256) {
                st.expert_vram_mask[id >> 6] |= 1ULL << (id & 63);
            }
        }
        for (size_t i = 0; i < st.cold_expert_ids.size(); ++i) {
            const int32_t id = st.cold_expert_ids[i];
            if (id >= 0 && id < w_.n_expert &&
                st.hot_local_by_global[(size_t)id] < 0) {
                st.cold_local_by_global[(size_t)id] = (int32_t)i;
            }
        }
        return true;
    }

    size_t staged_layer_bytes(const MoeHybridLayerStorage & st) const {
        const size_t per_expert = st.fused_gate_up
            ? st.gate_up_expert_bytes + st.down_expert_bytes
            : st.gate_expert_bytes + st.up_expert_bytes + st.down_expert_bytes;
        return per_expert * (size_t)w_.n_expert;
    }

    bool restore_pinned_layer(int il,
                              const PinnedPlacement & saved,
                              std::string & err) {
        auto & st = moe_hybrid_->layers[(size_t)il];
        const MoeLayerDesc desc = make_moe_layer_desc(w_.layers[(size_t)il]);
        const auto & regions = moe_hybrid_->layer_regions[(size_t)il];
        release_hot_storage(st);
        if (!allocate_hot_storage(st, desc, regions, saved.hot_ids,
                                  saved.cache_slots, false, err)) {
            mixed_restore_ok_ = false;
            std::fprintf(stderr,
                "[laguna-mixed] FATAL: failed to restore layer %d placement: %s\n",
                il, err.c_str());
            return false;
        }
        return true;
    }

    bool stage_full_layer(int il,
                          PinnedPlacement & saved,
                          std::string & err) {
        auto & st = moe_hybrid_->layers[(size_t)il];
        saved.hot_ids = st.hot_expert_ids;
        saved.cache_slots = st.cache_slots;

        ggml_backend_synchronize(backend_);
        release_hot_storage(st);

        const size_t need = staged_layer_bytes(st);
        size_t free_bytes = 0, total_bytes = 0;
        if (ggml_backend_dev_t dev = ggml_backend_get_device(backend_)) {
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        }
        const size_t reserve = mixed_reserve_bytes();
        if (free_bytes > 0 && need + reserve > free_bytes) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                "layer %d needs %.1f MiB plus %.1f MiB reserve; only %.1f MiB free",
                il, need / 1048576.0, reserve / 1048576.0,
                free_bytes / 1048576.0);
            err = msg;
            std::string restore_err;
            restore_pinned_layer(il, saved, restore_err);
            return false;
        }

        std::vector<int32_t> all_ids((size_t)w_.n_expert);
        std::iota(all_ids.begin(), all_ids.end(), 0);
        const MoeLayerDesc desc = make_moe_layer_desc(w_.layers[(size_t)il]);
        const auto & regions = moe_hybrid_->layer_regions[(size_t)il];
        if (!allocate_hot_storage(st, desc, regions, all_ids, 0, true, err)) {
            std::string restore_err;
            restore_pinned_layer(il, saved, restore_err);
            return false;
        }
        if (mixed_verbose()) {
            std::fprintf(stderr,
                "[laguna-mixed] staged layer %d: %d experts, %.1f MiB\n",
                il, w_.n_expert, need / 1048576.0);
        }
        return true;
    }

    static bool build_prefn_step(StepGraph & sg,
                                 const LagunaTargetWeights & w,
                                 LagunaTargetCache & cache,
                                 ggml_backend_t backend,
                                 int il,
                                 int kv_start,
                                 int n_tokens) {
        step_graph_free(sg);

        const int n_embd = w.n_embd;
        const bool is_full = laguna_is_full_attn_layer(w, il);
        const bool is_dense = il < w.n_layer_dense_lead;
        const LagunaTargetLayer & L = w.layers[(size_t)il];
        const int kv_len = kv_start + n_tokens;
        const int n_head = w.n_head_arr[il];
        const int n_head_kv = w.n_head_kv;
        const int head_dim = w.head_dim;

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 4096
                    + ggml_graph_overhead() + 8 * 1024 * 1024;
        ip.no_alloc = true;
        sg.ctx = ggml_init(ip);
        if (!sg.ctx) return false;
        sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);

        sg.inp_embed = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32,
                                          n_embd, n_tokens);
        ggml_set_input(sg.inp_embed);
        ggml_set_name(sg.inp_embed, "inp_embed");

        sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(sg.positions);

        ggml_tensor * attn_mask = nullptr;
        if (kv_len > 0) {
            attn_mask = ggml_new_tensor_4d(sg.ctx, GGML_TYPE_F32,
                                           kv_len, n_tokens, 1, 1);
            ggml_set_input(attn_mask);
            sg.attn_mask = attn_mask;
        }

        ggml_tensor * inp = sg.inp_embed;
        ggml_tensor * cur = ggml_rms_norm(sg.ctx, inp, 1e-6f);
        cur = ggml_mul(sg.ctx, cur, L.attn_norm);

        const int q_dim = n_head * head_dim;
        ggml_tensor * Qcur = ggml_mul_mat(sg.ctx, L.wq, cur);
        ggml_tensor * Kcur = ggml_mul_mat(sg.ctx, L.wk, cur);
        ggml_tensor * Vcur = ggml_mul_mat(sg.ctx, L.wv, cur);
        ggml_tensor * gate = ggml_softplus(
            sg.ctx, ggml_mul_mat(sg.ctx, L.wqkv_gate, cur));

        Qcur = ggml_reshape_3d(sg.ctx, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(sg.ctx, Kcur, head_dim, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(sg.ctx, Vcur, head_dim, n_head_kv, n_tokens);

        Qcur = ggml_mul(sg.ctx,
            ggml_rms_norm(sg.ctx, Qcur, 1e-6f), L.q_norm);
        Kcur = ggml_mul(sg.ctx,
            ggml_rms_norm(sg.ctx, Kcur, 1e-6f), L.k_norm);

        const float rope_th = is_full
            ? w.rope_freq_base_full : w.rope_freq_base_swa;
        const int n_rot = is_full ? w.n_rot_full : w.n_rot_swa;
        const float ext_factor = is_full ? 1.0f : 0.0f;
        const float beta_fast = is_full ? w.yarn_beta_fast : 32.0f;
        const float beta_slow = is_full ? w.yarn_beta_slow : 1.0f;
        const int n_ctx_orig = is_full ? w.yarn_orig_ctx : 0;
        const float freq_scale = is_full ? 1.0f / w.yarn_factor : 1.0f;

        Qcur = ggml_rope_ext(sg.ctx, Qcur, sg.positions, nullptr,
                             n_rot, GGML_ROPE_TYPE_NEOX,
                             n_ctx_orig, rope_th, freq_scale,
                             ext_factor, 1.0f, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(sg.ctx, Kcur, sg.positions, nullptr,
                             n_rot, GGML_ROPE_TYPE_NEOX,
                             n_ctx_orig, rope_th, freq_scale,
                             ext_factor, 1.0f, beta_fast, beta_slow);

        ggml_tensor * cache_k = cache.attn_k[(size_t)il];
        ggml_tensor * cache_v = cache.attn_v[(size_t)il];
        ggml_tensor * Kcur_T = ggml_permute(sg.ctx, Kcur, 0, 2, 1, 3);
        ggml_tensor * Vcur_T = ggml_permute(sg.ctx, Vcur, 0, 2, 1, 3);

        ggml_tensor * k_view = ggml_view_3d(
            sg.ctx, cache_k, head_dim, n_tokens, n_head_kv,
            cache_k->nb[1], cache_k->nb[2],
            cache_k->nb[1] * (size_t)kv_start);
        ggml_build_forward_expand(sg.gf, ggml_cpy(sg.ctx, Kcur_T, k_view));

        ggml_tensor * v_view = ggml_view_3d(
            sg.ctx, cache_v, head_dim, n_tokens, n_head_kv,
            cache_v->nb[1], cache_v->nb[2],
            cache_v->nb[1] * (size_t)kv_start);
        ggml_build_forward_expand(sg.gf, ggml_cpy(sg.ctx, Vcur_T, v_view));

        ggml_tensor * Qfa = ggml_cont(
            sg.ctx, ggml_permute(sg.ctx, Qcur, 0, 2, 1, 3));
        ggml_tensor * Kfa = ggml_view_3d(
            sg.ctx, cache_k, head_dim, kv_len, n_head_kv,
            cache_k->nb[1], cache_k->nb[2], 0);
        ggml_tensor * Vfa = ggml_view_3d(
            sg.ctx, cache_v, head_dim, kv_len, n_head_kv,
            cache_v->nb[1], cache_v->nb[2], 0);
        ggml_tensor * mask_f16 = attn_mask
            ? ggml_cast(sg.ctx, attn_mask, GGML_TYPE_F16) : nullptr;
        ggml_tensor * attn = ggml_flash_attn_ext(
            sg.ctx, Qfa, Kfa, Vfa, mask_f16,
            1.0f / std::sqrt((float)head_dim), 0.0f, 0.0f);

        ggml_tensor * gate_b = ggml_reshape_3d(
            sg.ctx, gate, 1, n_head, n_tokens);
        attn = ggml_mul(sg.ctx, attn,
                        ggml_cast(sg.ctx, gate_b, attn->type));
        attn = ggml_reshape_2d(sg.ctx, attn, q_dim, n_tokens);

        ggml_tensor * ffn_inp = ggml_add(
            sg.ctx, ggml_mul_mat(sg.ctx, L.wo, attn), inp);
        if (is_dense) {
            ggml_tensor * normed = ggml_mul(
                sg.ctx, ggml_rms_norm(sg.ctx, ffn_inp, 1e-6f), L.ffn_norm);
            ggml_tensor * g = ggml_mul_mat(sg.ctx, L.w_gate, normed);
            ggml_tensor * u = ggml_mul_mat(sg.ctx, L.w_up, normed);
            ggml_tensor * d = ggml_mul_mat(
                sg.ctx, L.w_down, ggml_swiglu_split(sg.ctx, g, u));
            sg.hidden_input = ggml_add(sg.ctx, d, ffn_inp);
            ggml_set_output(sg.hidden_input);
            ggml_build_forward_expand(sg.gf, sg.hidden_input);
        } else {
            ggml_tensor * normed = ggml_mul(
                sg.ctx, ggml_rms_norm(sg.ctx, ffn_inp, 1e-6f), L.ffn_norm);
            sg.ffn_post = normed;
            sg.ffn_residual = ffn_inp;
            ggml_tensor * probs = ggml_sigmoid(
                sg.ctx, ggml_mul_mat(sg.ctx, L.ffn_gate_inp, normed));
            ggml_tensor * selected = ggml_top_k(
                sg.ctx, ggml_add(sg.ctx, probs, L.ffn_exp_probs_b),
                w.n_expert_used);
            ggml_tensor * probs_3d = ggml_reshape_3d(
                sg.ctx, probs, 1, w.n_expert, n_tokens);
            ggml_tensor * weights = ggml_get_rows(
                sg.ctx, probs_3d, selected);
            weights = ggml_reshape_2d(
                sg.ctx, weights, w.n_expert_used, n_tokens);
            weights = ggml_div(sg.ctx, weights, ggml_sum_rows(sg.ctx, weights));
            if (w.expert_weights_scale != 1.0f) {
                weights = ggml_scale(sg.ctx, weights, w.expert_weights_scale);
            }
            sg.moe_weights = weights;
            sg.moe_selected.resize(1);
            sg.moe_selected[0] = selected;
            ggml_set_output(normed);
            ggml_set_output(ffn_inp);
            ggml_set_output(selected);
            ggml_set_output(weights);
            ggml_build_forward_expand(sg.gf, normed);
            ggml_build_forward_expand(sg.gf, ffn_inp);
            ggml_build_forward_expand(sg.gf, selected);
            ggml_build_forward_expand(sg.gf, weights);
        }

        if (!sg.alloc) {
            sg.alloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
        }
        return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
    }

    bool eval_full_stack_batched(MoeHybridLayerStorage & st,
                                 const MoeLayerDesc & desc,
                                 const float * input,
                                 const int32_t * selected,
                                 const float * weights,
                                 int n_tokens,
                                 std::vector<float> & output,
                                 std::string & err) {
        if (st.hot_active != w_.n_expert ||
            (int)st.hot_expert_ids.size() != w_.n_expert) {
            err = "full-stack prefill invoked without all experts staged";
            return false;
        }

        CachedHotBatchedGraph & graph = st.hot_batched_graph;
        if (!graph.valid() || graph.n_tokens != n_tokens) {
            if (!build_cached_hot_batched_graph(
                    graph, backend_, st, desc,
                    make_moe_hybrid_config(w_), n_tokens)) {
                err = "failed to build full-stack batched MoE graph";
                return false;
            }
        }

        const size_t hidden_count = (size_t)w_.n_embd * (size_t)n_tokens;
        const size_t route_count = (size_t)w_.n_expert_used * (size_t)n_tokens;
        ggml_backend_tensor_set(graph.inp, input, 0,
                                sizeof(float) * hidden_count);
        // The staged stack is identity ordered: global expert id == local id.
        ggml_backend_tensor_set(graph.sel, selected, 0,
                                sizeof(int32_t) * route_count);
        ggml_backend_tensor_set(graph.wts, weights, 0,
                                sizeof(float) * route_count);
        if (ggml_backend_graph_compute(backend_, graph.gf)
                != GGML_STATUS_SUCCESS) {
            err = "full-stack batched MoE compute failed";
            return false;
        }
        output.resize(hidden_count);
        ggml_backend_tensor_get(graph.output, output.data(), 0,
                                sizeof(float) * hidden_count);
        return true;
    }

    bool project_logits(const float * hidden,
                        std::vector<float> & logits_out,
                        std::string & err) {
        ggml_init_params ip{};
        ip.mem_size = 64 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            err = "prefill logits context allocation failed";
            return false;
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
        ggml_tensor * h_in = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w_.n_embd, 1);
        ggml_set_input(h_in);
        ggml_tensor * normed = ggml_mul(
            ctx, ggml_rms_norm(ctx, h_in, 1e-6f), w_.out_norm);
        ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, normed);
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            err = "prefill logits graph allocation failed";
            return false;
        }
        ggml_backend_tensor_set(h_in, hidden, 0,
                                sizeof(float) * (size_t)w_.n_embd);
        if (ggml_backend_graph_compute(backend_, gf)
                != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            err = "prefill logits compute failed";
            return false;
        }
        logits_out.resize((size_t)w_.embedder.n_vocab);
        ggml_backend_tensor_get(logits, logits_out.data(), 0,
                                sizeof(float) * logits_out.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return true;
    }

    GenerateResult generate_mixed(const GenerateRequest & req,
                                  const DaemonIO & io) {
        GenerateResult result;
        DaemonIO out_io = io.with_token_callback(req.on_token);
        const bool should_emit = req.stream || (bool)out_io.on_token;
        const int N = (int)req.prompt.size();

        sampler_ = req.sampler;
        if (req.do_sample && sampler_.seed != 0) {
            sampler_rng_.seed(sampler_.seed);
        }
        if (N <= 0) {
            result.fail(GenerateErrorCode::BackendSpecific, "empty_prompt");
            return result;
        }
        if (N + req.n_gen > args_.max_ctx) {
            result.fail(GenerateErrorCode::ContextOverflow);
            return result;
        }
        if (!moe_hybrid_->has_mmap() ||
            moe_hybrid_->layer_regions.size() < (size_t)w_.n_layer) {
            result.fail(GenerateErrorCode::BackendSpecific,
                        "mixed_prefill_missing_mmap");
            return result;
        }
        if (kvflash_active() &&
            N > kvflash_tokens_ - kvflash_pager_.chunk_tokens()) {
            std::fprintf(stderr,
                "[laguna-mixed] prompt %d exceeds identity prefill pool %d\n",
                N, kvflash_tokens_);
            result.fail(GenerateErrorCode::ContextOverflow);
            return result;
        }

        reset_laguna_target_cache(cache_);
        if (kvflash_active()) {
            kvflash_pager_.reset();
            if (!kvflash_alloc_span(0, N)) {
                result.fail(GenerateErrorCode::BackendSpecific, "kvflash_slot");
                return result;
            }
        }
        if (!ensure_moe_expert_compute()) {
            result.fail(GenerateErrorCode::BackendSpecific,
                        "moe_expert_compute");
            return result;
        }

        const int hidden = w_.n_embd;
        const int n_used = w_.n_expert_used;
        const int chunk = std::min(mixed_prefill_chunk(), N);
        std::fprintf(stderr,
            "[laguna-mixed] full-stack staged prefill: prompt=%d chunk=%d "
            "then hybrid decode\n", N, chunk);

        std::vector<float> embed_all((size_t)N * (size_t)hidden);
        if (!w_.embedder.embed(req.prompt.data(), N, embed_all.data())) {
            result.fail(GenerateErrorCode::BackendSpecific, "embed_prefill");
            return result;
        }

        auto t_pf0 = std::chrono::steady_clock::now();
        StepGraph prefn;
        std::string error;
        for (int il = 0; il < w_.n_layer; ++il) {
            const bool dense = il < w_.n_layer_dense_lead;
            const bool full_attn = laguna_is_full_attn_layer(w_, il);
            PinnedPlacement saved;
            bool staged = false;
            bool layer_ok = true;

            if (!dense) {
                if (!stage_full_layer(il, saved, error)) {
                    layer_ok = false;
                } else {
                    staged = true;
                }
            }

            for (int chunk_start = 0;
                 layer_ok && chunk_start < N;
                 chunk_start += chunk) {
                const int chunk_len = std::min(chunk, N - chunk_start);
                if (!build_prefn_step(prefn, w_, cache_, backend_, il,
                                      chunk_start, chunk_len)) {
                    error = "prefill graph build failed";
                    layer_ok = false;
                    break;
                }

                ggml_backend_tensor_set(
                    prefn.inp_embed,
                    embed_all.data() + (size_t)chunk_start * (size_t)hidden,
                    0, sizeof(float) * (size_t)chunk_len * (size_t)hidden);

                std::vector<int32_t> positions((size_t)chunk_len);
                for (int i = 0; i < chunk_len; ++i) {
                    positions[(size_t)i] = chunk_start + i;
                }
                ggml_backend_tensor_set(prefn.positions, positions.data(), 0,
                    sizeof(int32_t) * positions.size());

                if (prefn.attn_mask) {
                    const int kv_len = chunk_start + chunk_len;
                    std::vector<float> mask(
                        (size_t)kv_len * (size_t)chunk_len, -INFINITY);
                    for (int q = 0; q < chunk_len; ++q) {
                        const int abs_q = chunk_start + q;
                        const int win_lo = full_attn
                            ? 0
                            : std::max(0, abs_q - w_.sliding_window + 1);
                        for (int k = win_lo; k <= abs_q && k < kv_len; ++k) {
                            mask[(size_t)q * (size_t)kv_len + (size_t)k] = 0.0f;
                        }
                    }
                    ggml_backend_tensor_set(prefn.attn_mask, mask.data(), 0,
                                            sizeof(float) * mask.size());
                }

                if (ggml_backend_graph_compute(backend_, prefn.gf)
                        != GGML_STATUS_SUCCESS) {
                    error = "prefill attention/router compute failed";
                    layer_ok = false;
                    break;
                }

                if (dense) {
                    std::vector<float> layer_out(
                        (size_t)chunk_len * (size_t)hidden);
                    ggml_backend_tensor_get(prefn.hidden_input,
                        layer_out.data(), 0,
                        sizeof(float) * layer_out.size());
                    std::memcpy(
                        embed_all.data()
                            + (size_t)chunk_start * (size_t)hidden,
                        layer_out.data(), sizeof(float) * layer_out.size());
                    continue;
                }

                std::vector<float> residuals(
                    (size_t)chunk_len * (size_t)hidden);
                std::vector<float> ffn_input(
                    (size_t)chunk_len * (size_t)hidden);
                std::vector<int32_t> selected(
                    (size_t)chunk_len * (size_t)n_used);
                std::vector<float> weights(
                    (size_t)chunk_len * (size_t)n_used);
                ggml_backend_tensor_get(prefn.ffn_residual, residuals.data(), 0,
                                        sizeof(float) * residuals.size());
                ggml_backend_tensor_get(prefn.ffn_post, ffn_input.data(), 0,
                                        sizeof(float) * ffn_input.size());
                ggml_backend_tensor_get(prefn.moe_selected[0], selected.data(), 0,
                                        sizeof(int32_t) * selected.size());
                ggml_backend_tensor_get(prefn.moe_weights, weights.data(), 0,
                                        sizeof(float) * weights.size());

                if (routing_stats_) {
                    for (int i = 0; i < chunk_len; ++i) {
                        routing_stats_->observe(
                            il, selected.data() + (size_t)i * (size_t)n_used,
                            n_used);
                    }
                }
                if (routing_collector_) {
                    for (int i = 0; i < chunk_len; ++i) {
                        routing_collector_->record(
                            il,
                            ffn_input.data() + (size_t)i * (size_t)hidden,
                            hidden,
                            selected.data() + (size_t)i * (size_t)n_used,
                            n_used);
                    }
                }

                std::vector<float> ffn_out;
                MoeHybridLayerStorage & st =
                    moe_hybrid_->layers[(size_t)il];
                const MoeLayerDesc desc =
                    make_moe_layer_desc(w_.layers[(size_t)il]);
                if (!eval_full_stack_batched(
                        st, desc, ffn_input.data(), selected.data(),
                        weights.data(), chunk_len, ffn_out, error)) {
                    layer_ok = false;
                    break;
                }
                for (int i = 0; i < chunk_len; ++i) {
                    const float * ffn = ffn_out.data()
                        + (size_t)i * (size_t)hidden;
                    const float * res = residuals.data()
                        + (size_t)i * (size_t)hidden;
                    float * dst = embed_all.data()
                        + (size_t)(chunk_start + i) * (size_t)hidden;
                    for (int j = 0; j < hidden; ++j) {
                        dst[j] = ffn[j] + res[j];
                    }
                }
            }

            if (staged) {
                std::string restore_error;
                if (!restore_pinned_layer(il, saved, restore_error)) {
                    error = restore_error;
                    layer_ok = false;
                }
            }
            if (!layer_ok) {
                step_graph_destroy(prefn);
                result.fail(GenerateErrorCode::PrefillFailed,
                            error.empty() ? "mixed_prefill_failed" : error);
                return result;
            }
        }
        step_graph_destroy(prefn);

        cache_.cur_pos = N;
        cache_.last_tok = req.prompt.back();
        std::vector<float> last_logits;
        if (!project_logits(
                embed_all.data() + (size_t)(N - 1) * (size_t)hidden,
                last_logits, error)) {
            result.fail(GenerateErrorCode::BackendSpecific, error);
            return result;
        }
        auto t_pf1 = std::chrono::steady_clock::now();
        result.prefill_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();
        std::fprintf(stderr,
            "[laguna-mixed] prefill complete: %.3f s (%.1f tok/s); "
            "persistent hybrid placement restored\n",
            result.prefill_s,
            result.prefill_s > 0.0 ? N / result.prefill_s : 0.0);

        auto argmax = [](const std::vector<float> & logits) {
            int best = 0;
            float value = logits[0];
            for (size_t i = 1; i < logits.size(); ++i) {
                if (logits[i] > value) {
                    value = logits[i];
                    best = (int)i;
                }
            }
            return best;
        };

        std::vector<int32_t> history = req.prompt;
        auto pick = [&](const std::vector<float> & logits) {
            return req.do_sample
                ? sample_logits(logits.data(), (int)logits.size(),
                                req.sampler, history, sampler_rng_)
                : argmax(logits);
        };

        int next_tok = pick(last_logits);
        result.tokens.reserve((size_t)req.n_gen);
        const BudgetHook & budget_hook = req.budget_hook;
        bool close_started = false;
        int close_pos = 0;
        auto maybe_force_close = [&](int32_t & tok, int committed) {
            if (budget_hook.close_token_ids.empty()) return;
            if (close_started &&
                close_pos < (int)budget_hook.close_token_ids.size()) {
                tok = budget_hook.close_token_ids[(size_t)close_pos++];
                return;
            }
            if (close_started) return;
            if (req.n_gen - committed <= budget_hook.hard_limit_remaining) {
                const int32_t first = budget_hook.close_token_ids.front();
                if (tok == first) {
                    close_started = true;
                    close_pos = 1;
                    return;
                }
                tok = first;
                close_started = true;
                close_pos = 1;
                result.budget_forced_close = true;
            }
        };

        std::vector<float> act_cur((size_t)hidden);
        auto t_dec0 = std::chrono::steady_clock::now();
        for (int s = 0; s < req.n_gen; ++s) {
            maybe_force_close(next_tok, s);
            if (!std::getenv("DFLASH_IGNORE_EOS") &&
                (next_tok == w_.eos_id || next_tok == w_.eos_chat_id)) {
                break;
            }
            result.tokens.push_back(next_tok);
            history.push_back(next_tok);
            if (should_emit) {
                out_io.emit(next_tok);
                if (out_io.cancelled) break;
            }

            int32_t argmax_tok = 0;
            if (!hybrid_forward_one_token(
                    next_tok, cache_.cur_pos, act_cur, argmax_tok)) {
                result.fail(GenerateErrorCode::DecodeFailed);
                break;
            }
            cache_.cur_pos++;
            cache_.last_tok = next_tok;
            kvflash_maybe_reselect(history, s + 1);
            // Match the existing hybrid path: full-logit sampling is not yet
            // returned from hybrid_forward_one_token, so decode uses argmax.
            next_tok = argmax_tok;
        }
        auto t_dec1 = std::chrono::steady_clock::now();
        result.decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();

        if (should_emit) out_io.emit(-1);
        if (result.error &&
            result.error->code == GenerateErrorCode::Incomplete) {
            result.succeed();
        }
        return result;
    }
};

}  // namespace dflash::common
