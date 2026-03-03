// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

ov::Output<ov::Node> const_scalar(OpContext* ctx, float value);
ov::Output<ov::Node> const_scalar(OpContext* ctx, int64_t value);
ov::Output<ov::Node> const_scalar(OpContext* ctx, int32_t value);
ov::Output<ov::Node> const_scalar(OpContext* ctx, bool value);
ov::Output<ov::Node> const_vec(OpContext* ctx, const std::vector<float>& values);
ov::Output<ov::Node> const_vec(OpContext* ctx, const std::vector<int64_t>& values);
ov::Output<ov::Node> const_vec(OpContext* ctx, const std::vector<int32_t>& values);
Tensor constant(const ov::Tensor& tensor, OpContext* ctx = nullptr);

Tensor matmul(const Tensor& a, const Tensor& b, bool ta = false, bool tb = false);
Tensor linear(const Tensor& x, const Tensor& weight);
std::pair<Tensor, Tensor> linear_attention(const Tensor& q,
                                           const Tensor& k,
                                           const Tensor& v,
                                           const Tensor& beta,
                                           const Tensor& g,
                                           const Tensor& initial_state);
Tensor moe3gemm_fused_compressed(const Tensor& input,
                                 const Tensor& gate_inp_weight,
                                 const Tensor& gate_exps_weight,
                                 const Tensor& gate_exps_scales,
                                 const Tensor& gate_exps_zps,
                                 const Tensor& up_exps_weight,
                                 const Tensor& up_exps_scales,
                                 const Tensor& up_exps_zps,
                                 const Tensor& down_exps_weight,
                                 const Tensor& down_exps_scales,
                                 const Tensor& down_exps_zps,
                                 int32_t hidden_size,
                                 int32_t inter_size,
                                 int32_t num_experts,
                                 int32_t top_k,
                                 size_t group_size,
                                 const ov::element::Type& out_type);
Tensor silu(const Tensor& x);
Tensor reduce_mean(const Tensor& x, int64_t axis, bool keepdim = true);
Tensor gather(const Tensor& data, const Tensor& indices, int64_t axis);
Tensor slice(const Tensor& data, int64_t start, int64_t stop, int64_t step, int64_t axis);
Tensor range(const Tensor& stop, int64_t start, int64_t step, const ov::element::Type& type);
Tensor range(const Tensor& start, const Tensor& stop, int64_t step, const ov::element::Type& type);
Tensor greater_equal(const Tensor& a, const Tensor& b);
Tensor less_equal(const Tensor& a, const Tensor& b);
Tensor where(const Tensor& cond, const Tensor& then_value, const Tensor& else_value);
Tensor concat(const std::vector<Tensor>& xs, int64_t axis);
Tensor rms(const Tensor& x, const Tensor& weight, float eps);
std::pair<Tensor, Tensor> split(const Tensor& data, int64_t num_splits, int32_t axis);
Tensor convert(const Tensor& x, const ov::element::Type& dst_type);

// Trigonometric operations (for SnakeBeta activation)
Tensor sin(const Tensor& x);
Tensor cos(const Tensor& x);

// Reduction operations
Tensor reduce_sum(const Tensor& x, int64_t axis, bool keepdim = true);
Tensor reduce_sum(const Tensor& x, const std::vector<int64_t>& axes, bool keepdim = true);

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
