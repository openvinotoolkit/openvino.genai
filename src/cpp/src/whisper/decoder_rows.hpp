// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace ov::genai {

/**
 * Maps surviving decoder rows to their previous physical positions.
 * Rows within the new batch width stay in place; surviving tail rows
 * fill the holes. This may reorder requests but allows safe in-place
 * KV-cache compaction without overwriting needed source rows.
 */
std::vector<int32_t> plan_surviving_decoder_rows(const std::vector<bool>& survivors);

}  // namespace ov::genai
