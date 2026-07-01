// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

/**
 * @brief Runtime configuration for DFlash speculative decoding.
 */
struct DFlashRTInfo {
    bool dflash_mode = false;
    int64_t mask_token_id = -1;
    std::vector<int32_t> target_layer_ids;
};

/**
 * @brief Applies DFlash runtime info from model RT info to a properties map.
 */
void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties);

/**
 * @brief Extracts and removes DFlash configuration from a properties map.
 */
DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config);

/**
 * @brief Exposes annotated target hidden states as a concatenated model output.
 *
 * DFlash target models exported by Optimum carry semantic hidden-state tensor names
 * in `hidden_states_decoder_layers` RT info. This function looks up the requested
 * decoder layers by annotation and exposes them as `last_hidden_state`.
 */
void expose_target_hidden_states(std::shared_ptr<ov::Model>& model, const std::vector<int32_t>& target_layer_ids);

/**
 * @brief Makes DFlash draft accept CB-native hidden states externally.
 *
 * DFlash draft graphs are exported with hidden_states shaped [1, seq_len, hidden],
 * while continuous batching produces hidden states as [seq_len, 1, hidden].
 * This transform changes the public input shape and inserts a reshape back to
 * the exported graph layout for existing consumers.
 */
void reshape_draft_hidden_states_input_for_cb(std::shared_ptr<ov::Model>& model);

/**
 * @brief Grafts the target lm_head onto the draft model.
 *
 * Clones only the weight side (input(1)) of the target's final lm_head MatMul - including any INT4
 * decompression subgraph - and builds a fresh MatMul(draft last_hidden_state, cloned_weight) using
 * the target's transpose flags. The draft `last_hidden_state` Result is replaced by a `logits`
 * Result. Run before `expose_target_hidden_states` adds a second target output.
 */
void attach_target_lm_head_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                    const std::shared_ptr<ov::Model>& draft_model);

/**
 * @brief Builds a minimal `input_ids -> Gather -> inputs_embeds` model from the target embedding.
 *
 * Clones the target token-embedding weight subgraph into a standalone model used to feed the DFlash
 * draft its token embeddings (consumed via EmbeddingsModel).
 */
std::shared_ptr<ov::Model> build_draft_embedder_model(const std::shared_ptr<ov::Model>& main_model);

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
