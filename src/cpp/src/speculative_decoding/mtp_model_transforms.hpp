// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace genai {
namespace utils {

namespace mtp {

struct MtpRTInfo {
    bool mtp_mode = false;
};

MtpRTInfo extract_mtp_info_from_config(ov::AnyMap& config);

void apply_mtp_rt_info(const std::shared_ptr<ov::Model>& model, ov::AnyMap& properties);

ov::Output<ov::Node> extract_tied_lm_head_weight(const std::shared_ptr<ov::Model>& main_model,
                                                 bool& transpose_weight);

// Expose `last_hidden_state` on the main model by adding a Result on the lm_head MatMul input.
// Newer optimum-intel exports no longer emit this output, so MTP must graft it back.
void expose_last_hidden_state(const std::shared_ptr<ov::Model>& model);

// Remove no-op Convert<T>(Convert<U>(x)) pairs that block PA conversion in current MTP exports.
void remove_roundtrip_converts(const std::shared_ptr<ov::Model>& model);

// Add logits = MatMul(last_hidden_state, cloned main lm_head weight).
void graft_lm_head_on_mtp(std::shared_ptr<ov::Model>& mtp_model, const std::shared_ptr<ov::Model>& main_model);

}  // namespace mtp
}  // namespace utils
}  // namespace genai
}  // namespace ov
