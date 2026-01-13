// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <thread>

#include "module_genai/module.hpp"
#include "module_genai/module_base.hpp"
#include "module_genai/module_desc.hpp"
#include "module_genai/module_type.hpp"

namespace ov::genai::module {

#define FAKE_MODULE_CONSTRUCTOR(FM_NAME)                    \
    class FM_NAME : public ov::genai::module::IBaseModule { \
        DeclareModuleConstructor(FM_NAME);                  \
                                                            \
    public:                                                 \
    };

FAKE_MODULE_CONSTRUCTOR(FakeModuleA);
FAKE_MODULE_CONSTRUCTOR(FakeModuleB);
FAKE_MODULE_CONSTRUCTOR(FakeModuleC);
FAKE_MODULE_CONSTRUCTOR(FakeModuleD);
}  // namespace ov::genai::module