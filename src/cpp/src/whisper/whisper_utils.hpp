// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "openvino/genai/perf_metrics.hpp"

namespace ov {
namespace genai {
namespace utils {

void infer_with_perf_metrics(ov::InferRequest& request, ov::genai::RawPerfMetrics& raw_metrics);

void filter_non_segment_metrics(ov::genai::RawPerfMetrics& raw_metrics,
                                size_t offset,
                                std::vector<std::pair<size_t, size_t>>& ranges);

int64_t argmax(const ov::Tensor& logits, const size_t batch_idx);

}  // namespace utils
}  // namespace genai
}  // namespace ov
