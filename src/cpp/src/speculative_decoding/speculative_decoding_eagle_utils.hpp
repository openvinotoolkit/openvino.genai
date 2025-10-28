// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "openvino/genai/generation_config.hpp"

#include "openvino/runtime/core.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace genai {
namespace speculative_decoding {

constexpr std::size_t DEFAULT_NUM_ASSISTANT_TOKENS = 4;

// Set num_assistant_tokens to default if not specified and validate config
void ensure_num_assistant_tokens_is_set(ov::genai::GenerationConfig& config);

// Eagle3 runtime configuration
struct Eagle3RTInfo {
    bool eagle3_mode = false;
    std::vector<int> hidden_layers_list;
    std::filesystem::path dt_mapping_table;
};

Eagle3RTInfo extract_eagle_mode_from_config(ov::AnyMap& config, const std::filesystem::path& models_path = {});

// Share embedding weights from main model to draft model
void share_embedding_weights(std::shared_ptr<ov::Model>& main_model, std::shared_ptr<ov::Model>& draft_model);

// Move FC layer from draft model to main model
void shift_fc_from_draft_to_main(std::shared_ptr<ov::Model>& main_model, std::shared_ptr<ov::Model>& draft_model);

// Extract d2t mapping table constant from model
std::shared_ptr<ov::op::v0::Constant> extract_d2t_mapping_table(std::shared_ptr<ov::Model>& model);

// Remove d2t result node from model
void remove_d2t_result_node(std::shared_ptr<ov::Model>& model);

// Add hidden state output from specified layers
void hidden_state_transform(std::shared_ptr<ov::Model>& model, const std::vector<int>& hidden_layers_to_abstract);

}  // namespace speculative_decoding
}  // namespace genai
}  // namespace ov
