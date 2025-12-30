// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <unordered_map>
#include <openvino/runtime/properties.hpp>

namespace ov::genai {

enum class ImageGenerationModelType {
    UNKNOWN,
    ZIMAGE
};

inline ImageGenerationModelType to_image_generation_model_type(const std::string &value) {
    static const std::unordered_map<std::string, ImageGenerationModelType> model_types_map = {
        {"zimage", ImageGenerationModelType::ZIMAGE},
    };

    auto it = model_types_map.find(value);
    if (it != model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' image generation model type");
}

}