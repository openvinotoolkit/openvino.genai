// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/builder_context.hpp"

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

namespace {

void set_name(const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {

BuilderContext::BuilderContext(const OpPolicy& policy) : op_policy_(policy) {}

Tensor BuilderContext::parameter(const std::string& name,
                                 const ov::element::Type& type,
                                 const ov::PartialShape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    set_name(param, name);
    inputs_.push_back(param);
    return Tensor(param->output(0), &op_ctx_);
}

const ov::ParameterVector& BuilderContext::parameters() const {
    return inputs_;
}

std::shared_ptr<ov::Model> BuilderContext::build_model(const ov::OutputVector& outputs,
                                                       const ov::SinkVector& sinks) const {
    const ov::SinkVector& model_sinks = sinks.empty() ? sinks_ : sinks;
    return std::make_shared<ov::Model>(outputs, model_sinks, inputs_);
}

OpContext& BuilderContext::op_context() {
    return op_ctx_;
}

const OpContext& BuilderContext::op_context() const {
    return op_ctx_;
}

OpPolicy& BuilderContext::op_policy() {
    return op_policy_;
}

const OpPolicy& BuilderContext::op_policy() const {
    return op_policy_;
}

void BuilderContext::register_parameter(const std::string& full_name, WeightParameter* param) {
    if (params_by_name_.count(full_name)) {
        OPENVINO_THROW("Duplicate parameter name: ", full_name);
    }
    params_by_name_[full_name] = param;
    params_.push_back(param);
}

WeightParameter* BuilderContext::find_parameter(const std::string& full_name) const {
    auto it = params_by_name_.find(full_name);
    if (it == params_by_name_.end()) {
        return nullptr;
    }
    return it->second;
}

const std::vector<WeightParameter*>& BuilderContext::registered_parameters() const {
    return params_;
}

void BuilderContext::register_sink(const std::shared_ptr<ov::op::Sink>& sink) const {
    if (!sink) {
        return;
    }
    sinks_.push_back(sink);
}

const ov::SinkVector& BuilderContext::sinks() const {
    return sinks_;
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov
