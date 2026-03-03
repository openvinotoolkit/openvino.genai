// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/ops.hpp"

#include <openvino/core/except.hpp>
#include <openvino/op/linear_attn.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/op/placeholder_extension.hpp>
#include <openvino/op/moe_3gemm_fused_compressed.hpp>
#include <ov_ops/fully_connected.hpp>

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
namespace ops {

ov::Output<ov::Node> const_scalar(OpContext* ctx, float value) {
    if (ctx) {
        return ctx->scalar_f32(value);
    }
    return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {value});
}

ov::Output<ov::Node> const_scalar(OpContext* ctx, int64_t value) {
    if (ctx) {
        return ctx->scalar_i64(value);
    }
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

ov::Output<ov::Node> const_scalar(OpContext* ctx, int32_t value) {
    if (ctx) {
        return ctx->scalar_i32(value);
    }
    return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {value});
}

ov::Output<ov::Node> const_scalar(OpContext* ctx, bool value) {
    if (ctx) {
        return ctx->scalar_bool(value);
    }
    return ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {value});
}

ov::Output<ov::Node> const_vec(OpContext* ctx, const std::vector<float>& values) {
    (void)ctx;
    return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{values.size()}, values);
}

ov::Output<ov::Node> const_vec(OpContext* ctx, const std::vector<int64_t>& values) {
    if (ctx) {
        return ctx->const_i64_vec(values);
    }
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

ov::Output<ov::Node> const_vec(OpContext* ctx, const std::vector<int32_t>& values) {
    if (ctx) {
        return ctx->const_i32_vec(values);
    }
    return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{values.size()}, values);
}

Tensor constant(const ov::Tensor& tensor, OpContext* ctx) {
    auto node = std::make_shared<ov::op::v0::Constant>(tensor);
    return Tensor(node, ctx);
}

Tensor matmul(const Tensor& a, const Tensor& b, bool ta, bool tb) {
    auto* ctx = resolve_context(a, b);
    auto node = std::make_shared<ov::op::v0::MatMul>(a.output(), b.output(), ta, tb);
    return Tensor(node, ctx);
}

Tensor linear(const Tensor& x, const Tensor& weight) {
    auto* ctx = resolve_context(x, weight);
    // Use standard MatMul with transpose_b=true instead of internal FullyConnected
    // MatMul is serializable to IR, GPU will convert it back to FullyConnected at compile time
    auto node = std::make_shared<ov::op::v0::MatMul>(x.output(), weight.output(), false, true);
    return Tensor(node, ctx);
}

std::pair<Tensor, Tensor> linear_attention(const Tensor& q,
                                           const Tensor& k,
                                           const Tensor& v,
                                           const Tensor& beta,
                                           const Tensor& g,
                                           const Tensor& initial_state) {
    auto* ctx = q.context();
    const Tensor* inputs[] = {&k, &v, &beta, &g, &initial_state};
    for (const auto* t : inputs) {
        auto* t_ctx = t->context();
        if (ctx && t_ctx && ctx != t_ctx) {
            OPENVINO_THROW("Tensor contexts do not match");
        }
        if (!ctx) {
            ctx = t_ctx;
        }
    }

    // Note: the OCL kernel expects input[3]=g, input[4]=beta (swapped relative to
    // the C++ API parameter order), so we pass g before beta here.
    ov::OutputVector args = {q.output(), k.output(), v.output(), g.output(), beta.output(), initial_state.output()};
    auto node = std::make_shared<ov::op::LinearAttention>(args);
    return {Tensor(node->output(0), ctx), Tensor(node->output(1), ctx)};
}

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
                                 const ov::element::Type& out_type) {
    auto* ctx = input.context();
    auto router = matmul(input, gate_inp_weight, false, true);
    auto hidden_f16 = input.to(ov::element::f16);

    ov::op::internal::MOE3GemmFusedCompressed::Config config;
    config.hidden_size = hidden_size;
    config.inter_size = inter_size;
    config.num_expert = num_experts;
    config.top_k = top_k;
    config.group_size = group_size;
    config.out_type = out_type;

    ov::OutputVector args = {
        hidden_f16.output(),
        router.output(),
        gate_exps_weight.output(),
        gate_exps_scales.output(),
        gate_exps_zps.output(),
        up_exps_weight.output(),
        up_exps_scales.output(),
        up_exps_zps.output(),
        down_exps_weight.output(),
        down_exps_scales.output(),
        down_exps_zps.output()
    };
    auto moe = std::make_shared<ov::op::internal::MOE3GemmFusedCompressed>(args, config);
    auto moe_f32 = std::make_shared<ov::op::v0::Convert>(moe, ov::element::f32);
    return Tensor(moe_f32, ctx);
}

Tensor silu(const Tensor& x) {
    auto node = std::make_shared<ov::op::v4::Swish>(x.output());
    return Tensor(node, x.context());
}

Tensor reduce_mean(const Tensor& x, int64_t axis, bool keepdim) {
    return x.mean(axis, keepdim);
}

Tensor gather(const Tensor& data, const Tensor& indices, int64_t axis) {
    auto* ctx = resolve_context(data, indices);
    auto axis_node = const_scalar(ctx, axis);
    auto node = std::make_shared<ov::op::v8::Gather>(data.output(), indices.output(), axis_node, 0);
    return Tensor(node, ctx);
}

Tensor slice(const Tensor& data, int64_t start, int64_t stop, int64_t step, int64_t axis) {
    auto* ctx = data.context();
    auto start_node = const_vec(ctx, std::vector<int64_t>{start});
    auto stop_node = const_vec(ctx, std::vector<int64_t>{stop});
    auto step_node = const_vec(ctx, std::vector<int64_t>{step});
    auto axes_node = const_vec(ctx, std::vector<int64_t>{axis});
    auto node = std::make_shared<ov::opset13::Slice>(data.output(), start_node, stop_node, step_node, axes_node);
    return Tensor(node, ctx);
}

Tensor range(const Tensor& stop, int64_t start, int64_t step, const ov::element::Type& type) {
    auto* ctx = stop.context();
    auto start_node = const_scalar(ctx, start);
    auto step_node = const_scalar(ctx, step);
    auto node = std::make_shared<ov::op::v4::Range>(start_node, stop.output(), step_node, type);
    return Tensor(node, ctx);
}

Tensor range(const Tensor& start, const Tensor& stop, int64_t step, const ov::element::Type& type) {
    auto* ctx = resolve_context(start, stop);
    auto step_node = const_scalar(ctx, step);
    auto node = std::make_shared<ov::op::v4::Range>(start.output(), stop.output(), step_node, type);
    return Tensor(node, ctx);
}

Tensor greater_equal(const Tensor& a, const Tensor& b) {
    auto* ctx = resolve_context(a, b);
    auto node =
        std::make_shared<ov::op::v1::GreaterEqual>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor less_equal(const Tensor& a, const Tensor& b) {
    auto* ctx = resolve_context(a, b);
    auto node =
        std::make_shared<ov::op::v1::LessEqual>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor where(const Tensor& cond, const Tensor& then_value, const Tensor& else_value) {
    auto* ctx = resolve_context(then_value, else_value);
    auto* cond_ctx = cond.context();
    if (ctx && cond_ctx && ctx != cond_ctx) {
        OPENVINO_THROW("Tensor contexts do not match");
    }
    if (!ctx) {
        ctx = cond_ctx;
    }
    auto node = std::make_shared<ov::op::v1::Select>(cond.output(),
                                                     then_value.output(),
                                                     else_value.output(),
                                                     ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor concat(const std::vector<Tensor>& xs, int64_t axis) {
    if (xs.empty()) {
        OPENVINO_THROW("concat requires at least one tensor");
    }
    OpContext* ctx = xs.front().context();
    for (const auto& x : xs) {
        auto* x_ctx = x.context();
        if (ctx && x_ctx && ctx != x_ctx) {
            OPENVINO_THROW("Tensor contexts do not match");
        }
        if (!ctx) {
            ctx = x_ctx;
        }
    }
    ov::OutputVector outputs;
    outputs.reserve(xs.size());
    for (const auto& x : xs) {
        outputs.push_back(x.output());
    }
    auto node = std::make_shared<ov::op::v0::Concat>(outputs, axis);
    return Tensor(node, ctx);
}

Tensor rms(const Tensor& x, const Tensor& weight, float eps) {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto var = xf.pow(2.0f).mean(-1, true);
    auto norm = xf * (var + eps).rsqrt();
    return norm.to(orig_dtype) * weight;
}

Tensor sin(const Tensor& x) {
    auto* ctx = x.context();
    auto node = std::make_shared<ov::op::v0::Sin>(x.output());
    return Tensor(node, ctx);
}

Tensor cos(const Tensor& x) {
    auto* ctx = x.context();
    auto node = std::make_shared<ov::op::v0::Cos>(x.output());
    return Tensor(node, ctx);
}

Tensor reduce_sum(const Tensor& x, int64_t axis, bool keepdim) {
    return reduce_sum(x, std::vector<int64_t>{axis}, keepdim);
}

Tensor reduce_sum(const Tensor& x, const std::vector<int64_t>& axes, bool keepdim) {
    auto* ctx = x.context();
    auto axes_const = const_vec(ctx, axes);
    auto node = std::make_shared<ov::op::v1::ReduceSum>(x.output(), axes_const, keepdim);
    return Tensor(node, ctx);
}

std::pair<Tensor, Tensor> split(const Tensor& data, int64_t num_splits, int32_t axis) {
    if (num_splits <= 0) {
        OPENVINO_THROW("ops::split: num_splits must be > 0");
    }
    auto* ctx = data.context();
    auto axis_node = const_scalar(ctx, static_cast<int32_t>(axis));
    auto node = std::make_shared<ov::op::v1::Split>(data.output(), axis_node, static_cast<size_t>(num_splits));

    return {Tensor(node->output(0), ctx), Tensor(node->output(1), ctx)};
}

Tensor convert(const Tensor& x, const ov::element::Type& dst_type) {
    auto* ctx = x.context();
    auto node = std::make_shared<ov::op::v0::Convert>(x.output(), dst_type);
    return Tensor(node, ctx);
}

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
