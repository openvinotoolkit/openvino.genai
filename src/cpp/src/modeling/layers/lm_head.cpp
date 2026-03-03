// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/layers/lm_head.hpp"

#include <limits>

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"

namespace {

ov::genai::modeling::OpContext* resolve_context(const ov::genai::modeling::Tensor& a,
                                                const ov::genai::modeling::Tensor& b) {
    auto* a_ctx = a.context();
    auto* b_ctx = b.context();
    if (a_ctx && b_ctx && a_ctx != b_ctx) {
        OPENVINO_THROW("Tensor contexts do not match");
    }
    return a_ctx ? a_ctx : b_ctx;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {

LMHead::LMHead(const Tensor& weight) : Module(), weight_(weight) {}

LMHead::LMHead(BuilderContext& ctx, const std::string& name, Module* parent) : Module(name, ctx, parent) {
    weight_param_ = &register_parameter("weight");
}

void LMHead::tie_to(WeightParameter& other) {
    if (!weight_param_) {
        OPENVINO_THROW("LMHead has no registered parameter to tie");
    }
    weight_param_->tie_to(other);
}

WeightParameter& LMHead::weight_param() {
    if (!weight_param_) {
        OPENVINO_THROW("LMHead has no registered parameter");
    }
    return *weight_param_;
}

const WeightParameter& LMHead::weight_param() const {
    if (!weight_param_) {
        OPENVINO_THROW("LMHead has no registered parameter");
    }
    return *weight_param_;
}

const Tensor& LMHead::weight() const {
    if (weight_param_) {
        return weight_param_->value();
    }
    return weight_;
}

Tensor LMHead::forward(const Tensor& x) const {
    return ops::linear(x, weight());
}

Tensor LMHead::forward(const Tensor& x, const Tensor& cu_seqlens_q) const {
    auto* ctx = resolve_context(x, cu_seqlens_q);
    auto cu_i64 = cu_seqlens_q.to(ov::element::i64);

    // last_indices = cu_seqlens_q[1:] - 1
    auto cu_tail = ops::slice(cu_i64, 1, std::numeric_limits<int64_t>::max(), 1, 0);
    auto one = Tensor(ops::const_scalar(ctx, static_cast<int64_t>(1)), ctx);
    auto last_indices = cu_tail - one;

    // x[last_indices]
    auto x_last = ops::gather(x, last_indices, 0);
    return ops::linear(x_last, weight());
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov

