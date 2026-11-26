// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/genai/expert_predictor.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

class ExpertPredictorImpl {
public:
    struct Head {
        std::shared_ptr<ov::op::v0::Constant> weight;           // [num_experts, hidden_size], current token
        std::shared_ptr<ov::op::v0::Constant> previous_weight;  // [num_experts, hidden_size], previous token
        std::shared_ptr<ov::op::v0::Constant> selection_bias;   // [num_experts]
        int64_t input_mode = 0;
    };

    explicit ExpertPredictorImpl(const std::filesystem::path& path);

    std::map<size_t, Head> heads;  // key: 0-based index of the MoE decoder layer the head predicts
    size_t hidden_size = 0;
    size_t num_experts = 0;
};

namespace les {

// GPU plugin properties (the same in the openvino branches les_support and PR 40237): the predictor model (value:
// std::shared_ptr<ov::Model>, see build_predictor_model) that predicts the experts of the next MoE layer, and the
// lookahead prefetch of MoE offloading (OFFLOAD_RATIO) that uses the predictions. PR 40237 also needs the previous
// token input of the heads and top_k equal to the router top-k; les_support accepts any top_k.
inline constexpr const char* PLUGIN_PREDICTOR_PROPERTY = "GPU_LES_PREDICTOR";
inline constexpr const char* PLUGIN_LOOKAHEAD_PROPERTY = "GPU_MOE_LOOKAHEAD_PREFETCH";

// Names in the predictor model of the head of MoE layer L, + the 0-based MoE layer index L. The plugin tells the
// previous token input by "prev" in the input name.
inline constexpr const char* MODEL_HIDDEN_PREFIX = "les_hidden.";            // [?, hidden]: router input of layer L-1
inline constexpr const char* MODEL_PREV_HIDDEN_PREFIX = "les_prev_hidden.";  // [?, hidden]: router input of layer L, previous token
inline constexpr const char* MODEL_OUTPUT_PREFIX = "les_experts.";           // [?, top_k] i32: predicted experts

struct PredictorInfo {
    std::vector<size_t> layers;  // MoE layers with a head, ascending
    size_t hidden_size = 0;
    size_t top_k = 0;         // predicted experts per token
    size_t router_top_k = 0;  // experts the router selects per token
};

// Removes the expert_predictor property from `properties`; returns true and fills `predictor` if it was set.
bool extract_from_properties(utils::SharedOptional<const ov::AnyMap>& properties,
                             std::optional<ExpertPredictor>& predictor);

// Checks the predictor against the MoE layers of `model`, which it does not change.
PredictorInfo analyze(const std::shared_ptr<ov::Model>& model, const ExpertPredictor& predictor);

// The heads of the layers of `info` in one model, as independent subgraphs with their own weights: for layer L,
// inputs MODEL_HIDDEN_PREFIX<L> and MODEL_PREV_HIDDEN_PREFIX<L>, output MODEL_OUTPUT_PREFIX<L> =
// top_k(sigmoid(W cur + W' prev) + b).
std::shared_ptr<ov::Model> build_predictor_model(const ExpertPredictor& predictor, const PredictorInfo& info);

// Passes the predictor model of `model` to `device` (PLUGIN_PREDICTOR_PROPERTY, and PLUGIN_LOOKAHEAD_PROPERTY unless
// set by the user) if the device supports it; returns whether it did. The plugin keeps the previous token input of
// every head itself. The predictor model is not part of a compiled model blob, so the model cache (CACHE_DIR) is
// disabled then.
bool add_plugin_properties(utils::SharedOptional<const ov::AnyMap>& properties,
                           const std::string& device,
                           const std::shared_ptr<ov::Model>& model,
                           const ExpertPredictor& predictor);

}  // namespace les
}  // namespace genai
}  // namespace ov
