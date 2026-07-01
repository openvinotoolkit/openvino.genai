// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace genai {
namespace utils {

/**
 * @brief Multi-Token Prediction (MTP) draft-model transformations and configuration utilities.
 *
 * Qwen3.5 ships a single-layer MTP module (`openvino_mtp_model.xml`) alongside its decomposed VLM.
 * The MTP module drafts speculative tokens from the main model's `last_hidden_state`. Unlike EAGLE3,
 * the MTP module has no lm_head: its only Result is `last_hidden_state`, and logits are obtained by
 * applying the main model's tied embedding weight. These helpers graft that lm_head onto the MTP
 * graph so the standard continuous-batching model runner and sampler work unchanged.
 */
namespace mtp {

/**
 * @brief Runtime configuration for MTP speculative decoding.
 */
struct MtpRTInfo {
    bool mtp_mode = false;  ///< Enable MTP draft-model mode.
};

/**
 * @brief Extracts MTP configuration from a properties map, erasing consumed keys.
 * @param config Properties map that may contain the "mtp_mode" flag.
 * @return MtpRTInfo with the extracted configuration.
 */
MtpRTInfo extract_mtp_info_from_config(ov::AnyMap& config);

/**
 * @brief Copies MTP runtime info from the model's rt_info into a properties map.
 * @param model Model whose rt_info may contain "mtp_mode".
 * @param properties Properties map to update.
 * @note If the model has mtp_mode=true in rt_info, sets properties["mtp_mode"]=true.
 */
void apply_mtp_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties);

/**
 * @brief Extracts the tied lm_head weight subgraph root from the main language model.
 *
 * Locates the MatMul producing the `logits` Result and returns the node feeding its second input
 * (the tied embedding weight, possibly through a dequantization subgraph). The returned node is not
 * detached from the main model.
 *
 * @param main_model Main language model containing the lm_head.
 * @param transpose_weight [out] Set to the lm_head MatMul's transpose_b attribute so the graft can
 *        reproduce the same multiplication.
 * @return Output of the weight subgraph root, or an empty Output if the lm_head cannot be located.
 */
ov::Output<ov::Node> extract_tied_lm_head_weight(const std::shared_ptr<ov::Model>& main_model,
                                                 bool& transpose_weight);

/**
 * @brief Removes redundant round-trip Convert pairs (`Convert<T>(Convert<U>(x))` where `x` is already
 * of type `T`) from the model.
 *
 * The exported MTP module simulates a low-precision KV cache with a `Convert(f32->bf16)` immediately
 * followed by `Convert(bf16->f32)` on the KV-cache read path. This no-op round-trip sits between the
 * KV `Gather` and `Concat`, which prevents `SDPAToPagedAttention` from matching the KV pattern and
 * converting the attention. Eliminating the pairs restores a clean paged-attention conversion.
 *
 * @param model Model to transform.
 */
void remove_roundtrip_converts(const std::shared_ptr<ov::Model>& model);

/**
 * @brief Grafts an lm_head onto the MTP draft model so it outputs `logits` in addition to
 * `last_hidden_state`.
 *
 * Clones the tied embedding weight subgraph from the main model (to avoid cross-model references)
 * and adds `logits = MatMul(last_hidden_state, weight)` plus a `logits` Result to the MTP model.
 * The existing `last_hidden_state` Result is preserved, as it is reused for hidden-state export
 * during multi-token drafting.
 *
 * @param mtp_model MTP draft model to modify.
 * @param main_model Main language model providing the tied weight.
 * @throws Exception if the MTP `last_hidden_state` output or the main lm_head weight is not found.
 */
void graft_lm_head_on_mtp(std::shared_ptr<ov::Model>& mtp_model, const std::shared_ptr<ov::Model>& main_model);

}  // namespace mtp
}  // namespace utils
}  // namespace genai
}  // namespace ov
