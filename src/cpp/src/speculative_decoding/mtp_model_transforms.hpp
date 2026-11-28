// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace genai {
namespace utils {

namespace mtp {

// Draft-model property: token ids the MTP draft may propose (std::vector<int64_t>, or std::vector<int32_t>).
inline constexpr const char* MTP_DRAFT_VOCAB_PROPERTY = "mtp_draft_vocab";

struct MtpRTInfo {
    bool mtp_mode = false;
    // Sorted, unique token ids the draft lm_head is restricted to; empty means the full vocabulary.
    std::vector<int64_t> draft_vocab;
};

MtpRTInfo extract_mtp_info_from_config(ov::AnyMap& config);

void apply_mtp_rt_info(const std::shared_ptr<ov::Model>& model, ov::AnyMap& properties);

ov::Output<ov::Node> extract_tied_lm_head_weight(const std::shared_ptr<ov::Model>& main_model,
                                                 bool& transpose_weight);

// Expose `last_hidden_state` on the main model by adding a Result on the lm_head MatMul input.
// Newer optimum-intel exports no longer emit this output, so MTP must graft it back.
void expose_last_hidden_state(const std::shared_ptr<ov::Model>& model);

// Add logits = MatMul(last_hidden_state, cloned main lm_head weight).
void graft_lm_head_on_mtp(std::shared_ptr<ov::Model>& mtp_model, const std::shared_ptr<ov::Model>& main_model);

// Restrict the draft lm_head grafted by graft_lm_head_on_mtp() to the `draft_vocab` rows: every vocabulary-major
// constant under the lm_head weight (weight, scale, zero point) is sliced to those rows, and the draft logits are
// scattered back to the full vocabulary with -inf for the other token ids. The draft lm_head reads
// draft_vocab.size() / vocab_size of the weights, while the sampler, logit processors and the main model see
// full-vocabulary logits as before. The main model still verifies every candidate, so the output is unchanged;
// only the acceptance rate can drop when the target picks a token outside the subset.
// Call it after apply_gather_before_matmul_transformation(), which looks for the lm_head MatMul next to the logits.
void restrict_mtp_draft_vocab(const std::shared_ptr<ov::Model>& mtp_model, const std::vector<int64_t>& draft_vocab);

}  // namespace mtp
}  // namespace utils
}  // namespace genai
}  // namespace ov
