// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "openvino/genai/generation_config.hpp"

#include "openvino/runtime/core.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace genai {
namespace utils {

/**
 * @brief EAGLE3 speculative decoding model transformations and configuration utilities.
 *
 * This namespace provides functions for configuring and transforming models to support EAGLE3
 * speculative decoding, including extracting hidden states, sharing embeddings between draft
 * and main models, and managing model topology modifications.
 */
namespace eagle3 {

/**
 * @brief Runtime configuration for EAGLE3 speculative decoding.
 */
struct Eagle3RTInfo {
    bool eagle3_mode = false;                 ///< Enable EAGLE3 mode
    std::vector<int32_t> hidden_layers_list;  ///< Indices of layers to extract hidden states from
    std::filesystem::path dt_mapping_table;   ///< Path to draft-to-target mapping table
};

/**
 * @brief Extracts EAGLE3 configuration from model config.
 * @param config Model configuration map.
 * @param models_path Path to model directory for reading config.json if needed.
 * @return Eagle3RTInfo structure with extracted configuration.
 * @note If hidden_layers_list is not provided, defaults to [2, num_layers/2, num_layers-3].
 */
Eagle3RTInfo extract_eagle3_info_from_config(ov::AnyMap& config, const std::filesystem::path& models_path = {});

/**
 * @brief Applies EAGLE3 runtime info from model to properties map.
 * @param model Model containing rt_info with eagle3_mode and possibly hidden_layers_list.
 * @param properties Properties map to update with eagle3 configuration.
 * @note If model has eagle3_mode=true in rt_info, sets properties["eagle3_mode"]=true.
 *       If model has hidden_layers_list in rt_info, copies it to properties.
 */
void apply_eagle3_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties);

/**
 * @brief Shares embedding weights between main and draft models.
 * @param main_model Main (target) model.
 * @param draft_model Draft model for speculative decoding.
 * @note For current supported models (e.g., LLaMA3 and Qwen2), EAGLE3 models have no embedding
 *       weights in the torch weights and reuse the target model's embeddings. Future models with
 *       their own embedding layer (e.g., GPT-OSS) will need to use their own embedding weights.
 */
void share_vocabulary(const std::shared_ptr<ov::Model>& main_model, const std::shared_ptr<ov::Model>& draft_model);

/**
 * @brief Moves FC layer from draft model to main model.
 * @param draft_model Draft model (modified to remove FC layer).
 * @param main_model Main model (modified to include FC layer).
 * @throws Exception if FC layer is not found in draft model.
 */
void move_fc_from_draft_to_main(std::shared_ptr<ov::Model>& draft_model, std::shared_ptr<ov::Model>& main_model);

/**
 * @brief Extracts draft-to-target mapping table constant from model.
 * @param model Model to extract mapping table from.
 * @return Constant node containing the mapping table, or nullptr if not found.
 */
std::shared_ptr<ov::op::v0::Constant> extract_d2t_mapping_table(const std::shared_ptr<ov::Model>& model);

/**
 * @brief Extracts hidden states from specified decoder layers.
 *
 * This function modifies the provided model by identifying and extracting the outputs of residual
 * Add nodes corresponding to the specified hidden layers. The extracted hidden states are then
 * added as new result nodes to the model, either concatenated (if multiple layers are specified)
 * or as a single output.
 *
 * @param model Model to transform.
 * @param hidden_layers_to_abstract Layer indices to extract (1 for draft, 3 for main model).
 * @throws Exception if the number of layers is not 1 or 3, or if extraction fails.
 */
void transform_hidden_state(std::shared_ptr<ov::Model>& model, const std::vector<int32_t>& hidden_layers_to_abstract);

/**
 * @brief Slices hidden state tensor to extract only the last token's features.
 *
 * This function creates a view of the hidden state tensor containing only the features
 * corresponding to the last token position. Used in draft model forward pass to store
 * compact hidden state for next iteration.
 *
 * @param hidden_features Hidden state tensor with shape [1, seq_len, hidden_size].
 * @return Tensor view containing only the last token: [1, 1, hidden_size].
 * @throws Exception if tensor is empty or shape is invalid.
 */
ov::Tensor slice_hidden_state_for_last_token(const ov::Tensor& hidden_features);

}  // namespace eagle3
}  // namespace utils
}  // namespace genai
}  // namespace ov
