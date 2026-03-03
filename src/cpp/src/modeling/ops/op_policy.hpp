// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ov {
namespace genai {
namespace modeling {

struct OpPolicy {
    bool use_internal_fc = true;
    bool use_internal_rms = false;
    bool use_internal_rope = true;
    bool use_internal_sdpa = false;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
