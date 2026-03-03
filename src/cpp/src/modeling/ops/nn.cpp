// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/nn.hpp"

#include <openvino/op/interpolate.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"

namespace {

std::vector<size_t> to_size_t(const std::vector<int64_t>& values) {
    std::vector<size_t> out;
    out.reserve(values.size());
    for (auto v : values) {
        out.push_back(static_cast<size_t>(v));
    }
    return out;
}

ov::genai::modeling::OpContext* resolve_context(const ov::genai::modeling::Tensor& a,
                                                const ov::genai::modeling::Tensor& b) {
    auto* a_ctx = a.context();
    auto* b_ctx = b.context();
    if (a_ctx && b_ctx && a_ctx != b_ctx) {
        OPENVINO_THROW("Tensor contexts do not match");
    }
    return a_ctx ? a_ctx : b_ctx;
}

ov::genai::modeling::Tensor reshape_conv_bias(const ov::genai::modeling::Tensor& bias, size_t target_rank) {
    auto rank = bias.output().get_partial_shape().rank();
    if (!rank.is_static()) {
        return bias;
    }
    auto rank_len = static_cast<size_t>(rank.get_length());
    if (rank_len == target_rank) {
        return bias;
    }
    if (rank_len == 1 && target_rank == 4) {
        return bias.unsqueeze({0, 2, 3});
    }
    if (rank_len == 1 && target_rank == 5) {
        return bias.unsqueeze({0, 2, 3, 4});
    }
    return bias;
}

}  // namespace

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
              const std::vector<int64_t>& dilations) {
    auto* ctx = resolve_context(input, weight);
    auto node = std::make_shared<ov::op::v1::Convolution>(input.output(),
                                                          weight.output(),
                                                          to_size_t(strides),
                                                          ov::CoordinateDiff(pads_begin.begin(), pads_begin.end()),
                                                          ov::CoordinateDiff(pads_end.begin(), pads_end.end()),
                                                          to_size_t(dilations));
    return Tensor(node, ctx);
}

Tensor conv3d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations) {
    auto conv = conv3d(input, weight, strides, pads_begin, pads_end, dilations);
    auto bias_reshaped = reshape_conv_bias(bias, 5);
    auto node = std::make_shared<ov::op::v1::Add>(conv.output(),
                                                  bias_reshaped.output(),
                                                  ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, conv.context());
}

Tensor causal_conv3d(const Tensor& input,
                     const Tensor& weight,
                     const std::vector<int64_t>& strides,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilations) {
    if (padding.size() != 3) {
        OPENVINO_THROW("causal_conv3d requires padding with 3 values");
    }
    std::vector<int64_t> pad_begin = {0, 0, padding[0] * 2, padding[1], padding[2]};
    std::vector<int64_t> pad_end = {0, 0, 0, padding[1], padding[2]};
    auto padded = ops::tensor::pad(input, pad_begin, pad_end, 0.0f);
    return conv3d(padded, weight, strides, {0, 0, 0}, {0, 0, 0}, dilations);
}

Tensor causal_conv3d(const Tensor& input,
                     const Tensor& weight,
                     const Tensor& bias,
                     const std::vector<int64_t>& strides,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilations) {
    if (padding.size() != 3) {
        OPENVINO_THROW("causal_conv3d requires padding with 3 values");
    }
    std::vector<int64_t> pad_begin = {0, 0, padding[0] * 2, padding[1], padding[2]};
    std::vector<int64_t> pad_end = {0, 0, 0, padding[1], padding[2]};
    auto padded = ops::tensor::pad(input, pad_begin, pad_end, 0.0f);
    return conv3d(padded, weight, bias, strides, {0, 0, 0}, {0, 0, 0}, dilations);
}

Tensor conv2d(const Tensor& input,
              const Tensor& weight,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations) {
    auto* ctx = resolve_context(input, weight);
    auto node = std::make_shared<ov::op::v1::Convolution>(input.output(),
                                                          weight.output(),
                                                          to_size_t(strides),
                                                          ov::CoordinateDiff(pads_begin.begin(), pads_begin.end()),
                                                          ov::CoordinateDiff(pads_end.begin(), pads_end.end()),
                                                          to_size_t(dilations));
    return Tensor(node, ctx);
}

Tensor conv2d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations) {
    auto conv = conv2d(input, weight, strides, pads_begin, pads_end, dilations);
    auto bias_reshaped = reshape_conv_bias(bias, 4);
    auto node = std::make_shared<ov::op::v1::Add>(conv.output(),
                                                  bias_reshaped.output(),
                                                  ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, conv.context());
}

Tensor layer_norm(const Tensor& input,
                  const Tensor& weight,
                  const Tensor* bias,
                  float eps,
                  int64_t axis) {
    auto orig_dtype = input.dtype();
    auto x = input.to(ov::element::f32);
    auto mean = x.mean(axis, true);
    auto diff = x - mean;
    auto var = diff.pow(2.0f).mean(axis, true);
    auto norm = diff * (var + eps).rsqrt();
    auto out = norm.to(orig_dtype) * weight;
    if (bias) {
        out = out + *bias;
    }
    return out;
}

Tensor layer_norm(const Tensor& input,
                  const Tensor& weight,
                  float eps,
                  int64_t axis) {
    return layer_norm(input, weight, nullptr, eps, axis);
}

Tensor group_norm(const Tensor& input,
                  const Tensor& weight,
                  const Tensor* bias,
                  int64_t num_groups,
                  float eps) {
    if (num_groups <= 0) {
        OPENVINO_THROW("group_norm requires num_groups > 0");
    }
    auto* ctx = resolve_context(input, weight);
    auto orig_dtype = input.dtype();

    auto x = input.to(ov::element::f32);
    auto grouped = x.reshape({0, num_groups, -1});
    auto mean = grouped.mean(2, true);
    auto diff = grouped - mean;
    auto var = diff.pow(2.0f).mean(2, true);
    auto norm = diff * (var + eps).rsqrt();
    auto restored = norm.reshape(shape::of(input));
    auto out = restored.to(orig_dtype);

    auto scale = weight.to(orig_dtype).reshape({1, -1, 1, 1});
    out = out * scale;
    if (bias) {
        auto shift = bias->to(orig_dtype).reshape({1, -1, 1, 1});
        out = out + shift;
    }
    return out;
}

Tensor group_norm(const Tensor& input,
                  const Tensor& weight,
                  int64_t num_groups,
                  float eps) {
    return group_norm(input, weight, nullptr, num_groups, eps);
}

Tensor gelu(const Tensor& input, bool approximate) {
    auto* ctx = input.context();
    auto mode = approximate ? ov::op::GeluApproximationMode::TANH
                            : ov::op::GeluApproximationMode::ERF;
    auto node = std::make_shared<ov::op::v7::Gelu>(input.output(), mode);
    return Tensor(node, ctx);
}

Tensor upsample_nearest(const Tensor& input, int64_t scale_h, int64_t scale_w) {
    if (scale_h <= 0 || scale_w <= 0) {
        OPENVINO_THROW("upsample_nearest requires positive scales");
    }
    auto* ctx = input.context();
    auto h = shape::dim(input, 2);
    auto w = shape::dim(input, 3);
    auto h_scale = ops::const_scalar(ctx, static_cast<int64_t>(scale_h));
    auto w_scale = ops::const_scalar(ctx, static_cast<int64_t>(scale_w));
    auto out_h = std::make_shared<ov::op::v1::Multiply>(h, h_scale);
    auto out_w = std::make_shared<ov::op::v1::Multiply>(w, w_scale);
    auto sizes = shape::make({out_h, out_w});
    auto axes = ops::const_vec(ctx, std::vector<int64_t>{2, 3});

    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::FLOOR;

    auto interp = std::make_shared<ov::op::v11::Interpolate>(input.output(), sizes, axes, attrs);
    return Tensor(interp, ctx);
}

Tensor upsample_nearest_3d(const Tensor& input, int64_t scale_t, int64_t scale_h, int64_t scale_w) {
    if (scale_t <= 0 || scale_h <= 0 || scale_w <= 0) {
        OPENVINO_THROW("upsample_nearest_3d requires positive scales");
    }
    auto* ctx = input.context();
    auto t = shape::dim(input, 2);
    auto h = shape::dim(input, 3);
    auto w = shape::dim(input, 4);
    auto t_scale = ops::const_scalar(ctx, static_cast<int64_t>(scale_t));
    auto h_scale = ops::const_scalar(ctx, static_cast<int64_t>(scale_h));
    auto w_scale = ops::const_scalar(ctx, static_cast<int64_t>(scale_w));
    auto out_t = std::make_shared<ov::op::v1::Multiply>(t, t_scale);
    auto out_h = std::make_shared<ov::op::v1::Multiply>(h, h_scale);
    auto out_w = std::make_shared<ov::op::v1::Multiply>(w, w_scale);
    auto sizes = shape::make({out_t, out_h, out_w});
    auto axes = ops::const_vec(ctx, std::vector<int64_t>{2, 3, 4});

    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::FLOOR;

    auto interp = std::make_shared<ov::op::v11::Interpolate>(input.output(), sizes, axes, attrs);
    return Tensor(interp, ctx);
}

// ================= Activation Functions =================

Tensor relu(const Tensor& input) {
    auto* ctx = input.context();
    auto node = std::make_shared<ov::op::v0::Relu>(input.output());
    return Tensor(node, ctx);
}

Tensor sigmoid(const Tensor& input) {
    auto* ctx = input.context();
    auto node = std::make_shared<ov::op::v0::Sigmoid>(input.output());
    return Tensor(node, ctx);
}

Tensor tanh_activation(const Tensor& input) {
    auto* ctx = input.context();
    auto node = std::make_shared<ov::op::v0::Tanh>(input.output());
    return Tensor(node, ctx);
}

// ================= 1D Convolution Operations =================

namespace {
Tensor reshape_conv1d_bias(const Tensor& bias, const Tensor& conv_output) {
    auto bias_rank = bias.output().get_partial_shape().rank();
    if (!bias_rank.is_static()) {
        return bias;
    }
    auto bias_rank_len = static_cast<size_t>(bias_rank.get_length());
    if (bias_rank_len != 1) {
        return bias;
    }

    auto out_pshape = conv_output.output().get_partial_shape();
    auto out_rank = out_pshape.rank();
    if (out_rank.is_static() && out_rank.get_length() == 3) {
        const auto& bias_dim = bias.output().get_partial_shape()[0];
        const auto& out_c = out_pshape[1];
        const auto& out_l = out_pshape[2];

        if (bias_dim.is_static() && out_c.is_static() && bias_dim.get_length() == out_c.get_length()) {
            // NCL: [N, C, L]
            return bias.unsqueeze({0, 2});
        }
        if (bias_dim.is_static() && out_l.is_static() && bias_dim.get_length() == out_l.get_length()) {
            // NLC: [N, L, C]
            return bias.unsqueeze({0, 1});
        }
    }

    // Default to NCL broadcast: [1, C, 1]
    return bias.unsqueeze({0, 2});
}
}  // namespace

Tensor conv1d(const Tensor& input,
              const Tensor& weight,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations,
              int64_t groups) {
    auto* ctx = resolve_context(input, weight);
    if (groups == 1) {
        auto node = std::make_shared<ov::op::v1::Convolution>(
            input.output(),
            weight.output(),
            to_size_t(strides),
            ov::CoordinateDiff(pads_begin.begin(), pads_begin.end()),
            ov::CoordinateDiff(pads_end.begin(), pads_end.end()),
            to_size_t(dilations));
        return Tensor(node, ctx);
    } else {
        auto node = std::make_shared<ov::op::v1::GroupConvolution>(
            input.output(),
            weight.output(),
            to_size_t(strides),
            ov::CoordinateDiff(pads_begin.begin(), pads_begin.end()),
            ov::CoordinateDiff(pads_end.begin(), pads_end.end()),
            to_size_t(dilations));
        return Tensor(node, ctx);
    }
}

Tensor conv1d(const Tensor& input,
              const Tensor& weight,
              const Tensor& bias,
              const std::vector<int64_t>& strides,
              const std::vector<int64_t>& pads_begin,
              const std::vector<int64_t>& pads_end,
              const std::vector<int64_t>& dilations,
              int64_t groups) {
    auto conv = conv1d(input, weight, strides, pads_begin, pads_end, dilations, groups);
    auto bias_reshaped = reshape_conv1d_bias(bias, conv);
    auto node = std::make_shared<ov::op::v1::Add>(conv.output(),
                                                  bias_reshaped.output(),
                                                  ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, conv.context());
}

Tensor conv_transpose1d(const Tensor& input,
                        const Tensor& weight,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& pads_begin,
                        const std::vector<int64_t>& pads_end,
                        const std::vector<int64_t>& output_padding,
                        const std::vector<int64_t>& dilations,
                        int64_t groups) {
    auto* ctx = resolve_context(input, weight);
    if (groups == 1) {
        auto node = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
            input.output(),
            weight.output(),
            to_size_t(strides),
            ov::CoordinateDiff(pads_begin.begin(), pads_begin.end()),
            ov::CoordinateDiff(pads_end.begin(), pads_end.end()),
            to_size_t(dilations),
            ov::op::PadType::EXPLICIT,
            ov::CoordinateDiff(output_padding.begin(), output_padding.end()));
        return Tensor(node, ctx);
    } else {
        auto node = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(
            input.output(),
            weight.output(),
            to_size_t(strides),
            ov::CoordinateDiff(pads_begin.begin(), pads_begin.end()),
            ov::CoordinateDiff(pads_end.begin(), pads_end.end()),
            to_size_t(dilations),
            ov::op::PadType::EXPLICIT,
            ov::CoordinateDiff(output_padding.begin(), output_padding.end()));
        return Tensor(node, ctx);
    }
}

Tensor conv_transpose1d(const Tensor& input,
                        const Tensor& weight,
                        const Tensor& bias,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& pads_begin,
                        const std::vector<int64_t>& pads_end,
                        const std::vector<int64_t>& output_padding,
                        const std::vector<int64_t>& dilations,
                        int64_t groups) {
    auto conv = conv_transpose1d(input, weight, strides, pads_begin, pads_end, output_padding, dilations, groups);
    auto bias_reshaped = reshape_conv1d_bias(bias, conv);
    auto node = std::make_shared<ov::op::v1::Add>(conv.output(),
                                                  bias_reshaped.output(),
                                                  ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, conv.context());
}

// ================= Batch Normalization =================

Tensor batch_norm(const Tensor& input,
                  const Tensor& gamma,
                  const Tensor& beta,
                  const Tensor& running_mean,
                  const Tensor& running_var,
                  float eps) {
    auto* ctx = input.context();
    if (!ctx) ctx = gamma.context();
    if (!ctx) ctx = beta.context();
    if (!ctx) ctx = running_mean.context();
    if (!ctx) ctx = running_var.context();
    
    auto node = std::make_shared<ov::op::v5::BatchNormInference>(
        input.output(),
        gamma.output(),
        beta.output(),
        running_mean.output(),
        running_var.output(),
        eps);
    return Tensor(node, ctx);
}

// ================= Pooling Operations =================

Tensor adaptive_avg_pool1d(const Tensor& input, int64_t output_size) {
    auto* ctx = input.context();
    auto output_shape = ops::const_vec(ctx, std::vector<int64_t>{output_size});
    auto node = std::make_shared<ov::op::v8::AdaptiveAvgPool>(input.output(), output_shape);
    return Tensor(node, ctx);
}

Tensor avg_pool1d(const Tensor& input,
                  int64_t kernel_size,
                  int64_t stride,
                  int64_t padding) {
    auto* ctx = input.context();
    auto node = std::make_shared<ov::op::v1::AvgPool>(
        input.output(),
        ov::Strides{static_cast<size_t>(stride)},
        ov::Shape{static_cast<size_t>(padding)},
        ov::Shape{static_cast<size_t>(padding)},
        ov::Shape{static_cast<size_t>(kernel_size)},
        true,  // exclude_pad
        ov::op::RoundingType::FLOOR,
        ov::op::PadType::EXPLICIT);
    return Tensor(node, ctx);
}

Tensor max_pool1d(const Tensor& input,
                  int64_t kernel_size,
                  int64_t stride,
                  int64_t padding) {
    auto* ctx = input.context();
    auto node = std::make_shared<ov::op::v8::MaxPool>(
        input.output(),
        ov::Strides{static_cast<size_t>(stride)},
        ov::Strides{1},  // dilations
        ov::Shape{static_cast<size_t>(padding)},
        ov::Shape{static_cast<size_t>(padding)},
        ov::Shape{static_cast<size_t>(kernel_size)},
        ov::op::RoundingType::FLOOR,
        ov::op::PadType::EXPLICIT,
        ov::element::i64,
        0);
    return Tensor(node->output(0), ctx);
}

}  // namespace nn
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
