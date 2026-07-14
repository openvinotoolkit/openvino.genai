// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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
 * @brief A hidden-state locator resolved against a live target graph.
 */
struct DFlashHiddenStateLocator {
    std::string producer;
    ov::Output<ov::Node> output;
    size_t hidden_size;
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
 * @brief Parses target hidden-state RT info and resolves the requested locators.
 *
 * Returns std::nullopt when the metadata is absent.
 *
 * @throws ov::Exception for malformed metadata, duplicate locators, or a missing/ambiguous producer.
 */
std::optional<std::vector<DFlashHiddenStateLocator>> resolve_target_hidden_state_locators(
    const std::shared_ptr<ov::Model>& model,
    const std::vector<int32_t>& target_layer_ids);

/**
 * @brief Exposes annotated target hidden states, or uses the Eagle3 fallback when metadata is absent.
 */
void expose_target_hidden_states(std::shared_ptr<ov::Model>& model,
                                 const std::optional<std::vector<DFlashHiddenStateLocator>>& retained_locators,
                                 const std::vector<int32_t>& target_layer_ids);

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
 * Result. Run while the target is pristine, before PA and gather transformations.
 */
void attach_target_lm_head_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                    const std::shared_ptr<ov::Model>& draft_model);

/**
 * @brief Attaches the target token embedding into the draft graph.
 *
 * Clones the target token-embedding weight subgraph (including any INT4 decompression chain) and
 * splices `input_ids -> Gather(cloned_weight)` into the draft in place of its `inputs_embeds`
 * Parameter. The draft becomes self-contained (`input_ids`-native), mirroring how the target
 * lm_head is grafted. Run while the target is still pristine (before `SDPAToPagedAttention`).
 */
void attach_target_embedding_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                      const std::shared_ptr<ov::Model>& draft_model);

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
