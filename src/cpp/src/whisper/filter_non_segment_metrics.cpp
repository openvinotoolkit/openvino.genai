// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper/filter_non_segment_metrics.hpp"

namespace {

template <typename T>
void filter_by_ranges(std::vector<T>& value, size_t offset, std::vector<std::pair<size_t, size_t>>& ranges) {
    OPENVINO_ASSERT(ranges.empty() || value.size() >= (offset + ranges.back().second));
    std::vector<T> result{value.begin(), value.begin() + offset};
    for (auto [start, end] : ranges) {
        result.insert(result.end(), value.begin() + offset + start, value.begin() + offset + end);
    }

    value = result;
}

}  // namespace

namespace ov {
namespace genai {
namespace utils {

void filter_non_segment_metrics(ov::genai::RawPerfMetrics& raw_metrics,
                                size_t offset,
                                std::vector<std::pair<size_t, size_t>>& ranges) {
    filter_by_ranges(raw_metrics.m_token_infer_durations, offset, ranges);
    filter_by_ranges(raw_metrics.m_new_token_times, offset, ranges);
    filter_by_ranges(raw_metrics.m_batch_sizes, offset, ranges);
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
