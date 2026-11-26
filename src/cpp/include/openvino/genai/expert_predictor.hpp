// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "openvino/runtime/properties.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

class ExpertPredictorImpl;

/**
 * @brief Expert predictor for lookahead expert scheduling (LES) of Mixture-of-Experts models.
 *
 * The predictor holds one small head per MoE layer. The head of MoE layer L predicts, one layer ahead, which routed
 * experts the router of layer L selects for the current token:
 *
 *     cur  = router input of layer L-1 for the current token   (post_attention_layernorm output)
 *     prev = router input of layer L for the previous token of the same sequence
 *     experts(L) = top_k(sigmoid(W_L * cur + W'_L * prev) + b_L)
 *
 * It is read from a safetensors file with the tensors `expert_predictors.<i>.{weight, previous_weight,
 * selection_bias, input_mode_code}`, where `<i>` is the 1-based number of the decoder layer, so MoE layer L
 * (0-based) uses `expert_predictors.<L+1>`.
 *
 * GenAI builds every head as a separate small ov::Model and passes them to the device plugin that supports it
 * (GPU property MOE_LES_PREDICTORS); the language model itself is not changed. The plugin runs the heads only if
 * MoE experts are offloaded (OFFLOAD_RATIO > 0): the predictions tell it which offloaded experts it can start
 * loading early, the model still computes with its own router.
 */
class OPENVINO_GENAI_EXPORTS ExpertPredictor {
public:
    ExpertPredictor() = default;

    /**
     * @param path Path to the predictor file in safetensors format.
     * @param top_k Number of experts predicted per MoE layer and token. 0 means the number of experts the router
     *     selects. A larger value covers more of the router choices at the cost of loading more experts.
     * @param use_previous_token Use the previous token input of the heads (`previous_weight`). Without it, the heads
     *     use the current token only: the prediction is a bit less accurate, but no per-sequence state is kept.
     */
    explicit ExpertPredictor(const std::filesystem::path& path, size_t top_k = 0, bool use_previous_token = true);

    explicit operator bool() const {
        return static_cast<bool>(m_pimpl);
    }

    size_t get_top_k() const {
        return m_top_k;
    }

    bool get_use_previous_token() const {
        return m_use_previous_token;
    }

    const std::shared_ptr<ExpertPredictorImpl>& get_impl() const {
        return m_pimpl;
    }

private:
    std::shared_ptr<ExpertPredictorImpl> m_pimpl;
    size_t m_top_k = 0;
    bool m_use_previous_token = true;
};

/**
 * @brief Pipeline property that attaches an expert predictor to the language model, e.g.
 *     ov::genai::LLMPipeline pipe(models_path, "GPU", ov::genai::expert_predictor(ov::genai::ExpertPredictor(path)));
 * The property also accepts the path to the predictor file as a string. Supported by LLMPipeline with the stateful
 * (SDPA) and the continuous batching (PA) backends.
 */
static constexpr ov::Property<ExpertPredictor> expert_predictor{"expert_predictor"};

}  // namespace genai
}  // namespace ov
