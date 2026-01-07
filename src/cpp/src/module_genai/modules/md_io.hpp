// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
namespace ov {
namespace genai {
namespace module {

class ParameterModule : public IBaseModule {
    DeclareModuleConstructor(ParameterModule);

public:
    void run(ov::AnyMap& inputs);
};

REGISTER_MODULE_CONFIG(ParameterModule);

class ResultModule : public IBaseModule {
    DeclareModuleConstructor(ResultModule);

public:
    void run(ov::AnyMap& outputs);
};

REGISTER_MODULE_CONFIG(ResultModule);
}  // namespace module
}  // namespace genai
}  // namespace ov
