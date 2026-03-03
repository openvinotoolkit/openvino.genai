// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/layers/vocab_embedding.hpp"

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>

#include "modeling/ops/ops.hpp"

namespace ov {
namespace genai {
namespace modeling {

VocabEmbedding::VocabEmbedding(const Tensor& weight) : Module(), weight_(weight) {}

VocabEmbedding::VocabEmbedding(BuilderContext& ctx, const std::string& name, Module* parent)
    : Module(name, ctx, parent) {
    weight_param_ = &register_parameter("weight");
}

WeightParameter& VocabEmbedding::weight_param() {
    if (!weight_param_) {
        OPENVINO_THROW("VocabEmbedding has no registered parameter");
    }
    return *weight_param_;
}

const WeightParameter& VocabEmbedding::weight_param() const {
    if (!weight_param_) {
        OPENVINO_THROW("VocabEmbedding has no registered parameter");
    }
    return *weight_param_;
}

const Tensor& VocabEmbedding::weight() const {
    if (weight_param_) {
        return weight_param_->value();
    }
    return weight_;
}

Tensor VocabEmbedding::forward(const Tensor& ids) const {
    auto ids_i32 = ids.to(ov::element::i32);
    return ops::gather(weight(), ids_i32, 0);
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov

