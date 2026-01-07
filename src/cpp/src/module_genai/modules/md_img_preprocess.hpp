// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"

#include "visual_language/qwen2vl/classes.hpp"

namespace ov {
namespace genai {
namespace module {
class ImagePreprocessModule : public IBaseModule {
    DeclareModuleConstructor(ImagePreprocessModule);

private:
    std::shared_ptr<VisionEncoderQwen2VL> encoder_ptr = nullptr;
};

REGISTER_MODULE_CONFIG(ImagePreprocessModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
