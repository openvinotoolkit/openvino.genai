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
namespace nn {

Tensor conv3d(const Tensor& input,
              const Tensor& weight,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations = {1, 1, 1});

Tensor conv3d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations = {1, 1, 1});

Tensor causal_conv3d(const Tensor& input,
                     const Tensor& weight,
                     const std::vector<int64_t>& strides,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilations = {1, 1, 1});

Tensor causal_conv3d(const Tensor& input,
                     const Tensor& weight,
                     const Tensor& bias,
                     const std::vector<int64_t>& strides,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilations = {1, 1, 1});

Tensor conv2d(const Tensor& input,
              const Tensor& weight,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations = {1, 1});

Tensor conv2d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations = {1, 1});

Tensor layer_norm(const Tensor& input,
                  const Tensor& weight,
                  const Tensor* bias,
                  float eps,
                  int64_t axis = -1);

Tensor layer_norm(const Tensor& input,
                  const Tensor& weight,
                  float eps,
                  int64_t axis = -1);

Tensor group_norm(const Tensor& input,
                  const Tensor& weight,
                  const Tensor* bias,
                  int64_t num_groups,
                  float eps);

Tensor group_norm(const Tensor& input,
                  const Tensor& weight,
                  int64_t num_groups,
                  float eps);

Tensor gelu(const Tensor& input, bool approximate = true);

Tensor relu(const Tensor& input);

Tensor sigmoid(const Tensor& input);

Tensor tanh_activation(const Tensor& input);

// ================= 1D Convolution Operations =================

Tensor conv1d(const Tensor& input,
              const Tensor& weight,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations = {1},
              int64_t groups = 1);

Tensor conv1d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations = {1},
              int64_t groups = 1);

Tensor conv_transpose1d(const Tensor& input,
                        const Tensor& weight,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& pads_begin,
                        const std::vector<int64_t>& pads_end,
                        const std::vector<int64_t>& output_padding = {0},
                        const std::vector<int64_t>& dilations = {1},
                        int64_t groups = 1);

Tensor conv_transpose1d(const Tensor& input,
                        const Tensor& weight,
                        const Tensor& bias,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& pads_begin,
                        const std::vector<int64_t>& pads_end,
                        const std::vector<int64_t>& output_padding = {0},
                        const std::vector<int64_t>& dilations = {1},
                        int64_t groups = 1);

// ================= Batch Normalization =================

Tensor batch_norm(const Tensor& input,
                  const Tensor& gamma,
                  const Tensor& beta,
                  const Tensor& running_mean,
                  const Tensor& running_var,
                  float eps = 1e-5f);

// ================= Pooling Operations =================

Tensor adaptive_avg_pool1d(const Tensor& input, int64_t output_size);

Tensor avg_pool1d(const Tensor& input,
                  int64_t kernel_size,
                  int64_t stride,
                  int64_t padding = 0);

Tensor max_pool1d(const Tensor& input,
                  int64_t kernel_size,
                  int64_t stride,
                  int64_t padding = 0);

Tensor upsample_nearest(const Tensor& input, int64_t scale_h, int64_t scale_w);

Tensor upsample_nearest_3d(const Tensor& input, int64_t scale_t, int64_t scale_h, int64_t scale_w);

}  // namespace nn
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
