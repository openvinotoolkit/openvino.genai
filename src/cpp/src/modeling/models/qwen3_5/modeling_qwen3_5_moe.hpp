// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3_5TextModelConfig;

class Qwen3_5SparseMoeBlock : public Module {
public:
    Qwen3_5SparseMoeBlock(BuilderContext& ctx, const std::string& name, const Qwen3_5TextModelConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& gate_weight() const;
    const Tensor& down_expert_weights() const;
    const Tensor& shared_expert_gate_weight() const;
    const Tensor& shared_gate_proj_weight() const;
    const Tensor& shared_up_proj_weight() const;
    const Tensor& shared_down_proj_weight() const;

    Tensor routed_fallback(const Tensor& flat_f32) const;
    Tensor routed_fused(const Tensor& flat_f32) const;
    bool can_use_fused_path() const;
    size_t infer_group_size() const;

    int32_t hidden_size_ = 0;
    int32_t expert_intermediate_size_ = 0;
    int32_t shared_intermediate_size_ = 0;
    int32_t num_experts_ = 0;
    int32_t top_k_ = 1;
    bool norm_topk_prob_ = true;

    WeightParameter* gate_param_ = nullptr;
    WeightParameter* experts_gate_up_param_ = nullptr;  // maps to safetensor "gate_up_proj"
    WeightParameter* experts_down_param_ = nullptr;
    WeightParameter* shared_expert_gate_param_ = nullptr;
    WeightParameter* shared_gate_proj_param_ = nullptr;
    WeightParameter* shared_up_proj_param_ = nullptr;
    WeightParameter* shared_down_proj_param_ = nullptr;

    // Split gate/up weights (quantized independently from the fused gate_up_proj)
    std::optional<Tensor> gate_weight_;
    std::optional<Tensor> gate_scales_;
    std::optional<Tensor> gate_zps_;
    std::optional<Tensor> up_weight_;
    std::optional<Tensor> up_scales_;
    std::optional<Tensor> up_zps_;

    std::optional<Tensor> down_scales_;
    std::optional<Tensor> down_zps_;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
