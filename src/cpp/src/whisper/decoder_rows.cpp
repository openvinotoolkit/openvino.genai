// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder_rows.hpp"

#include <algorithm>
#include <numeric>

#include "openvino/core/except.hpp"

namespace ov::genai {

std::vector<int32_t> plan_surviving_decoder_rows(const std::vector<bool>& survivors) {
    const size_t width = survivors.size();
    const size_t new_width = static_cast<size_t>(std::count(survivors.begin(), survivors.end(), true));

    std::vector<int32_t> source_rows(new_width);
    std::iota(source_rows.begin(), source_rows.end(), 0);

    size_t tail = new_width;
    for (size_t row = 0; row < new_width; ++row) {
        if (survivors[row]) {
            continue;
        }
        while (tail < width && !survivors[tail]) {
            ++tail;
        }
        OPENVINO_ASSERT(tail < width, "Internal error: no surviving decoder row left to fill row ", row, ".");
        source_rows[row] = static_cast<int32_t>(tail++);
    }
    return source_rows;
}

}  // namespace ov::genai
