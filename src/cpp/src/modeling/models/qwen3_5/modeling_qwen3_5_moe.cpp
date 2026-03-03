// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/modeling_qwen3_5_moe.hpp"

#include <cstring>
#include <unordered_map>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

namespace {

/// Lightweight in-memory WeightSource wrapping a single pre-split tensor.
/// Used to call finalizer.finalize() for gate/up halves independently.
class InMemoryWeightSource : public weights::WeightSource {
public:
    void add(const std::string& name, ov::Tensor tensor) {
        tensors_[name] = std::move(tensor);
    }

    std::vector<std::string> keys() const override {
        std::vector<std::string> k;
        k.reserve(tensors_.size());
        for (const auto& p : tensors_)
            k.push_back(p.first);
        return k;
    }

    bool has(const std::string& name) const override {
        return tensors_.count(name) > 0;
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        auto it = tensors_.find(name);
        OPENVINO_ASSERT(it != tensors_.end(), "InMemoryWeightSource: tensor not found: ", name);
        return it->second;
    }

private:
    std::unordered_map<std::string, ov::Tensor> tensors_;
};

/// Split an ov::Tensor in half along the given dimension (CPU memcpy).
/// Requires element type with bitwidth >= 8.
std::pair<ov::Tensor, ov::Tensor> split_tensor_on_dim(const ov::Tensor& tensor, size_t dim) {
    auto shape = tensor.get_shape();
    auto elem_type = tensor.get_element_type();

    OPENVINO_ASSERT(elem_type.bitwidth() >= 8, "split_tensor_on_dim requires element bitwidth >= 8");
    OPENVINO_ASSERT(dim < shape.size(), "split_tensor_on_dim: dim out of range");
    OPENVINO_ASSERT(shape[dim] % 2 == 0, "split_tensor_on_dim: dim size must be even, got ", shape[dim]);

    size_t outer = 1;
    for (size_t i = 0; i < dim; ++i)
        outer *= shape[i];
    size_t mid = shape[dim] / 2;
    size_t inner = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i)
        inner *= shape[i];

    ov::Shape half_shape = shape;
    half_shape[dim] = mid;

    ov::Tensor first(elem_type, half_shape);
    ov::Tensor second(elem_type, half_shape);

    size_t elem_size = elem_type.size();
    size_t half_bytes = mid * inner * elem_size;
    size_t full_bytes = 2 * half_bytes;

    const uint8_t* src = static_cast<const uint8_t*>(tensor.data());
    uint8_t* dst1 = static_cast<uint8_t*>(first.data());
    uint8_t* dst2 = static_cast<uint8_t*>(second.data());

    for (size_t o = 0; o < outer; ++o) {
        std::memcpy(dst1 + o * half_bytes, src + o * full_bytes, half_bytes);
        std::memcpy(dst2 + o * half_bytes, src + o * full_bytes + half_bytes, half_bytes);
    }

    return {first, second};
}

Tensor dequantize_packed_moe_weight(const Tensor& packed_weight,
                                    const Tensor& scales_e_g_o,
                                    const Tensor& zps_e_g_o,
                                    int32_t num_experts,
                                    int32_t out_features,
                                    int32_t in_features) {
    auto* op_ctx = packed_weight.context();

    // Convert auxiliary tensors from [E, G, O] to [E, O, G] to align with packed weight layout [E, O, G, GS].
    auto perm = ops::const_vec(op_ctx, std::vector<int64_t>{0, 2, 1});
    auto scales_e_o_g = Tensor(std::make_shared<ov::op::v1::Transpose>(scales_e_g_o.output(), perm), op_ctx);
    auto zps_e_o_g = Tensor(std::make_shared<ov::op::v1::Transpose>(zps_e_g_o.output(), perm), op_ctx);

    auto packed_f32 = packed_weight.to(ov::element::f32);
    auto zps_f32 = zps_e_o_g.unsqueeze(-1).to(ov::element::f32);
    auto scales_f32 = scales_e_o_g.unsqueeze(-1).to(ov::element::f32);

    // Dequantize from packed groups: [E, O, G, GS] -> [E, O, I].
    auto dequant_grouped = (packed_f32 - zps_f32) * scales_f32;
    return dequant_grouped.reshape({num_experts, out_features, in_features}, false);
}

}  // namespace

Qwen3_5SparseMoeBlock::Qwen3_5SparseMoeBlock(BuilderContext& ctx,
                                             const std::string& name,
                                             const Qwen3_5TextModelConfig& cfg,
                                             Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.hidden_size),
      expert_intermediate_size_(cfg.moe_intermediate_size),
      shared_intermediate_size_(cfg.shared_expert_intermediate_size),
      num_experts_(cfg.num_experts),
      top_k_(cfg.num_experts_per_tok > 0 ? cfg.num_experts_per_tok : 1),
      norm_topk_prob_(cfg.norm_topk_prob) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3_5 MoE activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || expert_intermediate_size_ <= 0 || shared_intermediate_size_ <= 0 || num_experts_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3_5 MoE configuration");
    }
    if (top_k_ <= 0 || top_k_ > num_experts_) {
        OPENVINO_THROW("Invalid Qwen3_5 MoE top-k configuration");
    }

    gate_param_ = &register_parameter("gate.weight");
    experts_gate_up_param_ = &register_parameter("experts.gate_up_proj");
    experts_down_param_ = &register_parameter("experts.down_proj");
    shared_expert_gate_param_ = &register_parameter("shared_expert_gate.weight");
    shared_gate_proj_param_ = &register_parameter("shared_expert.gate_proj.weight");
    shared_up_proj_param_ = &register_parameter("shared_expert.up_proj.weight");
    shared_down_proj_param_ = &register_parameter("shared_expert.down_proj.weight");

    experts_gate_up_param_->set_weight_loader([this](WeightParameter& param,
                                                     weights::WeightSource& source,
                                                     weights::WeightFinalizer& finalizer,
                                                     const std::string& weight_name,
                                                     const std::optional<int>& shard_id) {
        (void)shard_id;
        if (!param.context()) {
            OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
        }

        // Split fused gate_up_proj [E, 2*I, H] before quantization so gate/up
        // get independent scales and zero points.
        const ov::Tensor& fused_tensor = source.get_tensor(weight_name);
        auto [gate_tensor, up_tensor] = split_tensor_on_dim(fused_tensor, 1);

        std::string gate_name = weight_name;
        {
            auto pos = gate_name.find("gate_up_proj");
            OPENVINO_ASSERT(pos != std::string::npos, "Expected 'gate_up_proj' in weight name: ", weight_name);
            gate_name.replace(pos, std::string("gate_up_proj").length(), "gate_proj.weight");
        }
        std::string up_name = weight_name;
        {
            auto pos = up_name.find("gate_up_proj");
            OPENVINO_ASSERT(pos != std::string::npos, "Expected 'gate_up_proj' in weight name: ", weight_name);
            up_name.replace(pos, std::string("gate_up_proj").length(), "up_proj.weight");
        }

        InMemoryWeightSource gate_source;
        gate_source.add(gate_name, gate_tensor);
        auto gate_result = finalizer.finalize(gate_name, gate_source, *param.context());

        InMemoryWeightSource up_source;
        up_source.add(up_name, up_tensor);
        auto up_result = finalizer.finalize(up_name, up_source, *param.context());

        gate_weight_ = gate_result.primary;
        gate_scales_ = gate_result.get_auxiliary("scales");
        gate_zps_ = gate_result.get_auxiliary("zps");

        up_weight_ = up_result.primary;
        up_scales_ = up_result.get_auxiliary("scales");
        up_zps_ = up_result.get_auxiliary("zps");

        // Keep original parameter marked as loaded.
        param.bind(gate_result.primary);
    });

    experts_down_param_->set_weight_loader([this](WeightParameter& param,
                                                  weights::WeightSource& source,
                                                  weights::WeightFinalizer& finalizer,
                                                  const std::string& weight_name,
                                                  const std::optional<int>& shard_id) {
        (void)shard_id;
        if (!param.context()) {
            OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
        }
        auto weight = finalizer.finalize(weight_name, source, *param.context());
        param.bind(weight);

        down_scales_.reset();
        down_zps_.reset();
        if (weight.get_auxiliary("scales") != std::nullopt && weight.get_auxiliary("zps") != std::nullopt) {
            down_scales_ = weight.auxiliary.at("scales");
            down_zps_ = weight.auxiliary.at("zps");
        }
    });
}

const Tensor& Qwen3_5SparseMoeBlock::gate_weight() const {
    if (!gate_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock gate parameter is not registered");
    }
    return gate_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::down_expert_weights() const {
    if (!experts_down_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock experts.down_proj parameter is not registered");
    }
    return experts_down_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::shared_expert_gate_weight() const {
    if (!shared_expert_gate_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock shared_expert_gate parameter is not registered");
    }
    return shared_expert_gate_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::shared_gate_proj_weight() const {
    if (!shared_gate_proj_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock shared_expert.gate_proj parameter is not registered");
    }
    return shared_gate_proj_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::shared_up_proj_weight() const {
    if (!shared_up_proj_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock shared_expert.up_proj parameter is not registered");
    }
    return shared_up_proj_param_->value();
}

const Tensor& Qwen3_5SparseMoeBlock::shared_down_proj_weight() const {
    if (!shared_down_proj_param_) {
        OPENVINO_THROW("Qwen3_5SparseMoeBlock shared_expert.down_proj parameter is not registered");
    }
    return shared_down_proj_param_->value();
}

bool Qwen3_5SparseMoeBlock::can_use_fused_path() const {
    const bool has_gate_quant = gate_weight_.has_value() && gate_scales_.has_value() && gate_zps_.has_value();
    const bool has_up_quant = up_weight_.has_value() && up_scales_.has_value() && up_zps_.has_value();
    const bool has_down_quant = down_scales_.has_value() && down_zps_.has_value();
    if (!(has_gate_quant && has_up_quant && has_down_quant)) {
        return false;
    }

    // Fused compressed MoE kernel expects packed expert weights.
    const auto gate_rank = gate_weight_->output().get_shape().size();
    const auto up_rank = up_weight_->output().get_shape().size();
    const auto down_rank = down_expert_weights().output().get_shape().size();
    return gate_rank == 4 && up_rank == 4 && down_rank == 4;
}

size_t Qwen3_5SparseMoeBlock::infer_group_size() const {
    if (gate_weight_.has_value()) {
        const auto shape = gate_weight_->output().get_shape();
        if (shape.size() == 4 && shape[3] > 0) {
            return shape[3];
        }
    }
    return 128;
}

Tensor Qwen3_5SparseMoeBlock::routed_fused(const Tensor& flat_f32) const {
    OPENVINO_ASSERT(gate_weight_.has_value() && gate_scales_.has_value() && gate_zps_.has_value(),
                    "Gate expert weights not loaded (split-before-quantize)");
    OPENVINO_ASSERT(up_weight_.has_value() && up_scales_.has_value() && up_zps_.has_value(),
                    "Up expert weights not loaded (split-before-quantize)");
    OPENVINO_ASSERT(down_scales_.has_value() && down_zps_.has_value(),
                    "Down expert scales/zps not loaded");

    return ops::moe3gemm_fused_compressed(flat_f32,
                                          gate_weight(),
                                          *gate_weight_,
                                          *gate_scales_,
                                          *gate_zps_,
                                          *up_weight_,
                                          *up_scales_,
                                          *up_zps_,
                                          down_expert_weights(),
                                          *down_scales_,
                                          *down_zps_,
                                          hidden_size_,
                                          expert_intermediate_size_,
                                          num_experts_,
                                          top_k_,
                                          infer_group_size(),
                                          ov::element::f16);
}

Tensor Qwen3_5SparseMoeBlock::routed_fallback(const Tensor& flat_f32) const {
    auto* op_ctx = flat_f32.context();

    auto logits = ops::linear(flat_f32, gate_weight().to(ov::element::f32));
    auto scores = logits.softmax(1);

    auto k_node = ops::const_scalar(op_ctx, static_cast<int64_t>(top_k_));
    auto topk = std::make_shared<ov::op::v11::TopK>(scores.output(),
                                                    k_node,
                                                    -1,
                                                    ov::op::v11::TopK::Mode::MAX,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    ov::element::i64);

    Tensor topk_vals(topk->output(0), op_ctx);
    Tensor topk_idx(topk->output(1), op_ctx);
    if (norm_topk_prob_) {
        auto reduce_axis = ops::const_vec(op_ctx, std::vector<int64_t>{-1});
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(topk_vals.output(), reduce_axis, true);
        topk_vals = topk_vals / Tensor(sum, op_ctx);
    }

    auto zeros = shape::broadcast_to(Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx), shape::of(scores));
    auto scatter_axis = ops::const_scalar(op_ctx, static_cast<int64_t>(1));
    auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(zeros.output(),
                                                                         topk_idx.output(),
                                                                         topk_vals.output(),
                                                                         scatter_axis);
    Tensor routing(scatter, op_ctx);  // [T, E]
    auto perm = ops::const_vec(op_ctx, std::vector<int64_t>{1, 0});
    auto routing_t = Tensor(std::make_shared<ov::op::v1::Transpose>(routing.output(), perm), op_ctx);
    auto routing_3d = routing_t.unsqueeze(-1);  // [E, T, 1]

    auto flat_3d = flat_f32.unsqueeze(0);
    auto tiled = ops::tensor::tile(flat_3d, {num_experts_, 1, 1});  // [E, T, H]

    const bool has_gate_quant = gate_weight_.has_value() && gate_scales_.has_value() && gate_zps_.has_value();
    const bool has_down_quant = down_scales_.has_value() && down_zps_.has_value();

    Tensor gate_exps_w, up_exps_w, down_exps_w;
    if (has_gate_quant && gate_weight_->output().get_shape().size() == 4) {
        gate_exps_w = dequantize_packed_moe_weight(*gate_weight_, *gate_scales_, *gate_zps_,
                                                   num_experts_, expert_intermediate_size_, hidden_size_);
        up_exps_w = dequantize_packed_moe_weight(*up_weight_, *up_scales_, *up_zps_,
                                                 num_experts_, expert_intermediate_size_, hidden_size_);
    } else if (gate_weight_.has_value()) {
        gate_exps_w = gate_weight_->to(ov::element::f32);
        up_exps_w = up_weight_->to(ov::element::f32);
    } else {
        OPENVINO_THROW("Gate/up expert weights not loaded for fallback path");
    }

    if (has_down_quant && down_expert_weights().output().get_shape().size() == 4) {
        down_exps_w = dequantize_packed_moe_weight(down_expert_weights(),
                                                   *down_scales_,
                                                   *down_zps_,
                                                   num_experts_,
                                                   hidden_size_,
                                                   expert_intermediate_size_);
    } else {
        down_exps_w = down_expert_weights().to(ov::element::f32);
    }

    auto gate_bmm = ops::matmul(tiled, gate_exps_w, false, true);
    auto up_bmm = ops::matmul(tiled, up_exps_w, false, true);
    auto swiglu = ops::silu(gate_bmm) * up_bmm;
    auto down_bmm = ops::matmul(swiglu, down_exps_w, false, true);

    auto weighted = down_bmm.to(ov::element::f32) * routing_3d;
    auto reduce_axis = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted.output(), reduce_axis, false);
    return Tensor(reduced, op_ctx);
}

Tensor Qwen3_5SparseMoeBlock::forward(const Tensor& hidden_states) const {
    auto* op_ctx = hidden_states.context();
    auto input_dtype = hidden_states.dtype();

    auto flat = hidden_states.reshape({-1, hidden_size_});
    auto flat_f32 = flat.to(ov::element::f32);

    auto routed_out = can_use_fused_path() ? routed_fused(flat_f32) : routed_fallback(flat_f32);

    auto shared_gate = ops::linear(flat_f32, shared_gate_proj_weight().to(ov::element::f32));
    auto shared_up = ops::linear(flat_f32, shared_up_proj_weight().to(ov::element::f32));
    auto shared_hidden = ops::silu(shared_gate) * shared_up;
    auto shared_out = ops::linear(shared_hidden, shared_down_proj_weight().to(ov::element::f32));

    auto shared_gate_logits = ops::linear(flat_f32, shared_expert_gate_weight().to(ov::element::f32));
    auto shared_gate_sigmoid = Tensor(std::make_shared<ov::op::v0::Sigmoid>(shared_gate_logits.output()), op_ctx);
    auto combined = routed_out + (shared_out * shared_gate_sigmoid);
    auto restored = combined.reshape(shape::of(hidden_states), false);
    return restored.to(input_dtype);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
