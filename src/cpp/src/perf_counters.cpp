// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "perf_counters.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/openvino.hpp"
#include <tuple>
#include <numeric>
#include <cmath>

namespace ov {
namespace genai {

void PerfCounters::add_timestamp(size_t batch_size) {
    m_new_token_times.emplace_back(std::chrono::steady_clock::now());
    m_batch_sizes.emplace_back(batch_size);
}


} // namespace genai
} // namespace ov
