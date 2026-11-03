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
    int64_t dflash_version = 1;
    int64_t mask_token_id = -1;
    std::vector<int32_t> target_layer_ids;
    size_t candidate_position_offset = 1;
    float gpu_min_activations_scale_factor = 0.0f;
    float input_embedding_scale = 1.0f;
    float output_multiplier = 1.0f;
    float final_logit_softcapping = 0.0f;
    // A backbone with a Markov head: DSpark (vanilla head) or DFlash-2 (gated head). The head is exported as a separate
    // model, and the draft runs the computation of it selected by markov_head_top_k: the dense step or the Top-K
    // candidate scorer; see is_markov_step().
    size_t markov_rank = 0;
    std::string markov_head_type;
    std::shared_ptr<ov::Model> markov_head_computation;
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
 * @brief Promotes DFlash-specific GPU requirements to explicit OpenVINO compile properties.
 *
 * The GPU plugin intentionally ignores generic activation-scale RT info for LLM graphs. DFlash v2
 * therefore stores a GPU minimum in its own metadata, which GenAI applies explicitly when compiling
 * the draft. A requested value above the minimum is preserved. A lower value set for the draft is
 * rejected, but one from the target's properties, which a draft without properties of its own
 * compiles with (inherits_target_properties), is raised to the minimum: it may be meant for the target.
 */
void apply_dflash_gpu_compile_properties(const DFlashRTInfo& info,
                                         const std::string& device,
                                         bool inherits_target_properties,
                                         ov::AnyMap& compile_properties);

/**
 * @brief Reads the Markov head model (openvino_markov_head_model.xml) exported next to a draft backbone with a
 * Markov head, such as DSpark or DFlash-2, and keeps the computation selected by the markov_head_top_k draft property;
 * see select_markov_head_computation(). Consumes markov_head_top_k.
 *
 * Reads nothing unless apply_dflash_rt_info() found a Markov head in the backbone metadata.
 */
void read_markov_head(const std::filesystem::path& models_path, ov::AnyMap& properties);

/**
 * @brief Keeps one computation of the Markov head model and stores it in properties for
 * extract_dflash_info_from_config().
 *
 * The Markov head model holds up to two computations over the same weights:
 * - the dense step, prev_token_ids [N] -> markov_bias [N, vocab], walked once per drafted position. Only vanilla
 *   heads have it; a gated head's transitions depend on the draft hidden states, so it has no dense step;
 * - the Top-K candidate scorer, draft_logits [T, vocab], anchor_token_ids [N], subsequence_begins [N + 1] ->
 *   edge_scores [T, K, K], candidate_ids [T, K], which scores whole blocks in one inference. A gated head also takes
 *   draft_hidden_states [T, hidden], the backbone's final hidden state of every row. The T rows hold N blocks back
 *   to back; block n spans rows subsequence_begins[n] to subsequence_begins[n + 1], and its anchor token precedes
 *   its first row. The graph does not check subsequence_begins; callers validate it with
 *   dflash_cb::validate_subsequence_begins().
 * These port names are the contract with the exporter, so a port whose meaning changes needs a new name.
 *
 * top_k TOP_K_FOR_MARKOV_STEP (0) keeps the dense step. Any other top_k keeps the candidate scorer with that K,
 * replacing the exported K when it differs. Without top_k, the candidate scorer keeps the exported K.
 *
 * @throws ov::Exception when top_k selects the dense step of a head without one.
 */
void select_markov_head_computation(const std::shared_ptr<ov::Model>& markov_head_model,
                                    std::optional<int64_t> top_k,
                                    ov::AnyMap& properties);

/**
 * @brief True when a computation kept by select_markov_head_computation() is the dense step rather than the
 * candidate scorer. The scorer's K is its dflash/top_k RT info.
 */
bool is_markov_step(const std::shared_ptr<ov::Model>& markov_head_computation);

/**
 * @brief Validates the dense step computation of the Markov head against the draft backbone:
 * prev_token_ids [N] -> markov_bias [N, vocab].
 */
void validate_markov_step(const std::shared_ptr<ov::Model>& markov_step, const DFlashRTInfo& info);

/**
 * @brief Validates the Top-K candidate scorer computation of the Markov head against the draft backbone:
 * draft_logits [T, vocab], anchor_token_ids [N], subsequence_begins [N + 1], and draft_hidden_states [T, hidden] for
 * a gated head -> edge_scores [T, K, K], candidate_ids [T, K].
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
 * @brief Moves the draft's hidden-state FC into the target.
 *
 * The draft projects the concatenated target hidden states with `MatMul(hidden_states, fc_weight)`.
 * Running it on the target's `last_hidden_state` makes the hidden states passed to the draft
 * `in_width / out_width` times smaller. The FC weight subgraph moves to the target with its
 * decompression chain, the draft FC consumers read `hidden_states` directly, and `hidden_states` is
 * resized to the projected width. The target FC carries a hash of the FC weights in its rt_info, which keys the
 * target's model cache on the draft. Run after `expose_target_hidden_states`.
 *
 * @throws ov::Exception when `hidden_states` does not feed only the data input of an FC MatMul, or when
 * the target hidden-state width does not match the draft `hidden_states` width.
 */
void move_fc_from_draft_to_main(const std::shared_ptr<ov::Model>& draft_model,
                                const std::shared_ptr<ov::Model>& main_model);

/**
 * @brief Grafts the target lm_head onto the draft model.
 *
 * Clones only the weight side (input(1)) of the target's final lm_head MatMul - including any INT4
 * decompression subgraph - and builds a fresh MatMul(draft last_hidden_state, cloned_weight) using
 * the target's transpose flags. The draft `last_hidden_state` Result is replaced by a `logits`
 * Result. Run while the target is pristine, before PA and gather transformations.
 */
void attach_target_lm_head_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                    const std::shared_ptr<ov::Model>& draft_model,
                                    bool keep_hidden_state = false);

/**
 * @brief Makes the draft's last_hidden_state output read the rows that the lm_head reads after
 * apply_gather_before_matmul_transformation(), so it holds the hidden state of every logits row.
 *
 * Run after the gather transformation, on a draft whose lm_head was grafted with keep_hidden_state.
 */
void gather_last_hidden_state_with_logits(const std::shared_ptr<ov::Model>& draft_model);

/**
 * @brief Attaches the target token embedding into the draft graph.
 *
 * Clones the target token-embedding weight subgraph (including any INT4 decompression chain) and
 * splices `input_ids -> Gather(cloned_weight)` into the draft in place of its `inputs_embeds`
 * Parameter. The draft becomes self-contained (`input_ids`-native), mirroring how the target
 * lm_head is grafted. Run while the target is still pristine (before `SDPAToPagedAttention`).
 */
void attach_target_embedding_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                      const std::shared_ptr<ov::Model>& draft_model,
                                      float input_embedding_scale = 1.0f);

/**
 * @brief Applies the target embedding scale to an inputs_embeds-native draft.
 *
 * Used by VLM drafts, whose target embedding pipeline remains external to the
 * draft graph.
 */
void scale_draft_inputs_embeds(const std::shared_ptr<ov::Model>& draft_model,
                               float input_embedding_scale = 1.0f);

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
