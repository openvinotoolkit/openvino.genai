// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/op/sink.hpp>

#include "modeling/ops/context.hpp"
#include "modeling/ops/op_policy.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

class WeightParameter;

class BuilderContext {
public:
    BuilderContext() = default;
    explicit BuilderContext(const OpPolicy& policy);

    Tensor parameter(const std::string& name, const ov::element::Type& type, const ov::PartialShape& shape);

    const ov::ParameterVector& parameters() const;
    std::shared_ptr<ov::Model> build_model(const ov::OutputVector& outputs,
                                           const ov::SinkVector& sinks = {}) const;
    void register_sink(const std::shared_ptr<ov::op::Sink>& sink) const;
    const ov::SinkVector& sinks() const;

    OpContext& op_context();
    const OpContext& op_context() const;
    OpPolicy& op_policy();
    const OpPolicy& op_policy() const;

    void register_parameter(const std::string& full_name, WeightParameter* param);
    WeightParameter* find_parameter(const std::string& full_name) const;
    const std::vector<WeightParameter*>& registered_parameters() const;

private:
    OpContext op_ctx_;
    OpPolicy op_policy_;
    ov::ParameterVector inputs_;
    std::unordered_map<std::string, WeightParameter*> params_by_name_;
    std::vector<WeightParameter*> params_;
    mutable ov::SinkVector sinks_;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
