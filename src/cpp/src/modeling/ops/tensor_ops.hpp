// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {
namespace tensor {

Tensor repeat(const Tensor& x, const std::vector<int64_t>& repeats);
Tensor repeat(const Tensor& x, std::initializer_list<int64_t> repeats);
Tensor tile(const Tensor& x, const std::vector<int64_t>& repeats);
Tensor tile(const Tensor& x, std::initializer_list<int64_t> repeats);
Tensor stack(const std::vector<Tensor>& xs, int64_t axis);
Tensor masked_scatter(const Tensor& input, const Tensor& mask, const Tensor& updates);
Tensor masked_add(const Tensor& input, const Tensor& mask, const Tensor& updates);
Tensor masked_fill(const Tensor& input, const Tensor& mask, float value);
Tensor pad(const Tensor& input,
           const std::vector<int64_t>& pads_begin,
           const std::vector<int64_t>& pads_end,
           float value = 0.0f);

std::vector<Tensor> split(const Tensor& input, int64_t num_splits, int64_t axis);
std::vector<Tensor> split(const Tensor& input, const std::vector<int64_t>& split_sizes, int64_t axis);

Tensor flip(const Tensor& input, int64_t axis);

}  // namespace tensor
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
