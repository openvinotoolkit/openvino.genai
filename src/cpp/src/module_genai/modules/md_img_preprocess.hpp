// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>
#include <variant>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "model/qwen3_5/qwen3_5preprocessor.hpp"
#include "visual_language/qwen2vl/classes.hpp"

namespace ov {
namespace genai {
namespace module {
class ImagePreprocessModule : public IBaseModule {
    DeclareModuleConstructor(ImagePreprocessModule);

private:
    std::variant<std::shared_ptr<VisionEncoderQwen2VL>, std::shared_ptr<Qwen3_5Preprocessor>> encoder_ptr;
};

REGISTER_MODULE_CONFIG(ImagePreprocessModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
