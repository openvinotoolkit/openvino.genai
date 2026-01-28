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

static const std::unordered_map<std::string, ImageGenerationModelType> image_generation_model_types_map = {
        {"zimage", ImageGenerationModelType::ZIMAGE},
};

inline ImageGenerationModelType to_image_generation_model_type(const std::string &value) {
    auto it = image_generation_model_types_map.find(value);
    if (it != image_generation_model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' image generation model type");
}

inline bool is_image_generation_model(const std::string &value) {
    return image_generation_model_types_map.find(value) != image_generation_model_types_map.end();
}

}