// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
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

// The markov_head_top_k value that selects the dense Markov step instead of the Top-K candidate scorer.
inline constexpr int64_t TOP_K_FOR_MARKOV_STEP = 0;

/**
 * @brief Runtime configuration for DFlash speculative decoding.
 */
struct DFlashRTInfo {
    bool dflash_mode = false;
    int64_t mask_token_id = -1;
    std::vector<int32_t> target_layer_ids;
    size_t candidate_position_offset = 1;
    // A backbone with a Markov head, such as DSpark. The head is exported as a separate model, and the draft runs the
    // computation of it selected by markov_head_top_k: the dense step or the Top-K candidate scorer.
    size_t markov_rank = 0;
    std::string markov_head_type;
    std::shared_ptr<ov::Model> markov_head_computation;
    // TOP_K_FOR_MARKOV_STEP for the dense step, otherwise the K of the candidate scorer. Without markov_head_top_k,
    // draft_model() selects the candidate scorer with the exported K (64 in optimum-intel exports); this initial
    // value applies only to drafts without a Markov head.
    int64_t markov_head_top_k = TOP_K_FOR_MARKOV_STEP;
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
 * @brief Reads the Markov head model (openvino_markov_head_model.xml) exported next to a draft backbone with a
 * Markov head, such as DSpark, and keeps the computation selected by the markov_head_top_k draft property; see
 * select_markov_head_computation(). Consumes markov_head_top_k.
 *
 * Reads nothing unless apply_dflash_rt_info() found a Markov head in the backbone metadata.
 */
void read_markov_head(const std::filesystem::path& models_path, ov::AnyMap& properties);

/**
 * @brief Keeps one computation of the Markov head model and stores it in properties, together with the
 * selected markov_head_top_k, for extract_dflash_info_from_config().
 *
 * The Markov head model holds two computations over the same weights:
 * - the dense step, prev_token_ids [N] -> markov_bias [N, vocab], walked once per drafted position;
 * - the Top-K candidate scorer, draft_logits [T, vocab], anchor_token_ids [N], subsequence_begins [N + 1] ->
 *   edge_scores [T, K, K], candidate_ids [T, K], which scores whole blocks in one inference. The T rows hold N
 *   blocks back to back; block n spans rows subsequence_begins[n] to subsequence_begins[n + 1], and its anchor
 *   token precedes its first row. The graph does not check subsequence_begins; callers validate it with
 *   dflash_cb::validate_subsequence_begins().
 *
 * top_k TOP_K_FOR_MARKOV_STEP (0) keeps the dense step. Any other top_k keeps the candidate scorer with that K,
 * replacing the exported K when it differs. Without top_k, the candidate scorer keeps the exported K.
 */
void select_markov_head_computation(const std::shared_ptr<ov::Model>& markov_head_model,
                                    std::optional<int64_t> top_k,
                                    ov::AnyMap& properties);

/**
 * @brief Validates the dense step computation of the Markov head against the draft backbone:
 * prev_token_ids [N] -> markov_bias [N, vocab].
 */
void validate_markov_step(const std::shared_ptr<ov::Model>& markov_step, const DFlashRTInfo& info);

/**
 * @brief Validates the Top-K candidate scorer computation of the Markov head against the draft backbone:
 * draft_logits [T, vocab], anchor_token_ids [N], subsequence_begins [N + 1] -> edge_scores [T, K, K],
 * candidate_ids [T, K].
 *
 * edge_scores already include each candidate's draft logit; the greedy candidate walk uses them as is.
 */
void validate_candidate_scorer(const std::shared_ptr<ov::Model>& candidate_scorer, const DFlashRTInfo& info);

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
