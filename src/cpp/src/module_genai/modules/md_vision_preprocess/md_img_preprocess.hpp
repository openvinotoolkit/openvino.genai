// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>
#include <variant>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "module_genai/modules/models/qwen3_5/qwen3_5preprocessor.hpp"
#include "visual_language/qwen2vl/classes.hpp"
// include vision_preprocess.hpp
#include "vision_preprocess.hpp"

namespace ov {
namespace genai {
namespace module {
class ImagePreprocessModule : public IBaseModule {
    DeclareModuleConstructor(ImagePreprocessModule);

private:
    VLMModelType _model_type;
    VisionPreprocess::PTR _vision_preprocess_ptr = nullptr;
    VisionEncoder::Ptr _encoder_ptr = nullptr;
    void run_image(const bool& has_images_input);
};

REGISTER_MODULE_CONFIG(ImagePreprocessModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
