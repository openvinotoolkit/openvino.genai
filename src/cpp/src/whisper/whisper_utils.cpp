// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper/whisper_utils.hpp"

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

void infer_with_perf_metrics(ov::InferRequest& request, ov::genai::RawPerfMetrics& raw_metrics) {
    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_end = std::chrono::steady_clock::now();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    raw_metrics.m_batch_sizes.emplace_back(1);
}

void filter_non_segment_metrics(ov::genai::RawPerfMetrics& raw_metrics,
                                size_t offset,
                                std::vector<std::pair<size_t, size_t>>& ranges) {
    filter_by_ranges(raw_metrics.m_token_infer_durations, offset, ranges);
    filter_by_ranges(raw_metrics.m_new_token_times, offset, ranges);
    filter_by_ranges(raw_metrics.m_batch_sizes, offset, ranges);
}

int64_t argmax(const ov::Tensor& logits, const size_t batch_idx) {
    if (logits.get_shape()[0] <= batch_idx) {
        OPENVINO_THROW("logits batch size doesn't match the number of beams");
    }

    size_t vocab_size = logits.get_shape().back();
    size_t batch_offset = batch_idx * logits.get_shape()[1] * vocab_size;
    size_t sequence_offset = (logits.get_shape()[1] - 1) * vocab_size;
    const float* logits_data = logits.data<const float>() + batch_offset + sequence_offset;

    int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
    float max_logit = logits_data[out_token];

    return out_token;
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
