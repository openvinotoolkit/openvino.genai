// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "openvino/core/shape.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {
namespace numpy_utils {

// https://gist.github.com/lorenzoriano/5414671
template <typename T, typename U>
std::vector<T> linspace(U start, U end, size_t num, bool endpoint = false) {
    std::vector<T> indices;
    if (num != 0) {
        if (num == 1)
            indices.push_back(static_cast<T>(start));
        else {
            if (endpoint)
                --num;

            U delta = (end - start) / static_cast<U>(num);
            for (size_t i = 0; i < num; i++)
                indices.push_back(static_cast<T>(start + delta * i));

            if (endpoint)
                indices.push_back(static_cast<T>(end));
        }
    }
    return indices;
}

// Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
void rescale_zero_terminal_snr(std::vector<float>& betas);

// np.interp(...) implementation
std::vector<float> interp(const std::vector<std::int64_t>& x, const std::vector<size_t>& xp, const std::vector<float>& fp);

// concats two tensors by a given dimension
ov::Tensor concat(ov::Tensor tensor_1, ov::Tensor tensor_2, int axis);

void batch_copy(ov::Tensor src, ov::Tensor dst, size_t src_batch, size_t dst_batch, size_t batch_size = 1);
ov::Tensor repeat(const ov::Tensor input, const size_t num_images_per_prompt);

} // namespace ov
} // namespace genai
} // namespace numpy_utils
