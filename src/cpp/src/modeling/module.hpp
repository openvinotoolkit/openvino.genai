// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/weights/weight_parameter.hpp"

namespace ov {
namespace genai {
namespace modeling {

struct PackedRule {
    std::string match;
    std::string replace;
    int shard_id = 0;
};

struct PackedMapping {
    std::vector<PackedRule> rules;
};

class Module {
public:
    Module() = default;
    Module(std::string name, BuilderContext& ctx, Module* parent = nullptr);
    virtual ~Module() = default;

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) = default;
    Module& operator=(Module&&) = default;

    const std::string& name() const;
    const std::string& full_path() const;

    WeightParameter& register_parameter(const std::string& name);
    WeightParameter& get_parameter(const std::string& full_name);
    void register_module(const std::string& name, Module* child);
    std::vector<std::pair<std::string, Module*>> named_modules(bool include_self = true) const;
    std::vector<std::pair<std::string, WeightParameter*>> named_parameters(bool recurse = true) const;

    BuilderContext& ctx();
    const BuilderContext& ctx() const;

    PackedMapping& packed_mapping();
    const PackedMapping& packed_mapping() const;

    void finalize_parameters();

protected:
    BuilderContext* ctx_ = nullptr;
    Module* parent_ = nullptr;
    std::string name_;
    std::string full_path_;
    std::vector<std::unique_ptr<WeightParameter>> parameters_;
    std::vector<std::pair<std::string, Module*>> children_;
    PackedMapping packed_mapping_;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
