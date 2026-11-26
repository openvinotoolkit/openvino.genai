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

class LESPredictorImpl;

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
 * where top_k is the number of experts the router of the model selects per token.
 *
 * It is read from a safetensors file with the tensors `expert_predictors.<i>.{weight, previous_weight,
 * selection_bias, input_mode_code}`, where `<i>` is the 1-based number of the decoder layer, so MoE layer L
 * (0-based) uses `expert_predictors.<L+1>`.
 *
 * GenAI builds the heads of all MoE layers in one separate ov::Model, as independent subgraphs, and passes it in
 * memory to the device plugin that supports it (GPU properties GPU_LES_PREDICTOR and GPU_MOE_LOOKAHEAD_PREFETCH);
 * the language model itself is not changed. With MoE offloading (OFFLOAD_RATIO > 0), the plugin runs the head of the
 * next MoE layer in decode steps and keeps the previous token input of every head: the predictions tell it which
 * offloaded experts it can start loading early, the model still computes with its own router.
 */
class OPENVINO_GENAI_EXPORTS LESPredictor {
public:
    LESPredictor() = default;

    /**
     * @param path Path to the predictor file in safetensors format.
     */
    explicit LESPredictor(const std::filesystem::path& path);

    explicit operator bool() const {
        return static_cast<bool>(m_pimpl);
    }

    const std::shared_ptr<LESPredictorImpl>& get_impl() const {
        return m_pimpl;
    }

private:
    std::shared_ptr<LESPredictorImpl> m_pimpl;
};

/**
 * @brief Pipeline property that attaches a LES predictor to the language model, given as a LESPredictor or as the
 * path to the predictor file:
 *     ov::genai::LLMPipeline pipe(models_path, "GPU", ov::genai::les_predictor(ov::genai::LESPredictor(path)));
 *     ov::genai::LLMPipeline pipe(models_path, "GPU", ov::genai::les_predictor(path));
 * In an ov::AnyMap (e.g. from Python or a JSON config), the value can also be a std::string or std::filesystem::path.
 * Supported by LLMPipeline with the stateful (SDPA) and the continuous batching (PA) backends.
 */
class LESPredictorProperty : public ov::Property<LESPredictor> {
public:
    inline constexpr static const char* name() {
        return "les_predictor";
    }

    constexpr LESPredictorProperty() : ov::Property<LESPredictor>(name()) {}

    inline std::pair<std::string, ov::Any> operator()(const LESPredictor& predictor) const {
        return ov::Property<LESPredictor>::operator()(predictor);
    }

    // Reads the predictor file.
    inline std::pair<std::string, ov::Any> operator()(const std::filesystem::path& path) const {
        return operator()(LESPredictor(path));
    }
};

static constexpr LESPredictorProperty les_predictor;

}  // namespace genai
}  // namespace ov
