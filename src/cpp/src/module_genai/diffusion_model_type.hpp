// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <unordered_map>
#include <openvino/runtime/properties.hpp>

namespace ov::genai {

enum class DiffusionModelType {
    UNKNOWN,
    ZIMAGE,
    WAN_2_1
};

static const std::unordered_map<std::string, DiffusionModelType> diffusion_model_types_map = {
        {"zimage", DiffusionModelType::ZIMAGE},
        {"wan2.1", DiffusionModelType::WAN_2_1},
};

inline DiffusionModelType to_diffusion_model_type(const std::string &value) {
    auto it = diffusion_model_types_map.find(value);
    if (it != diffusion_model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' diffusion model type");
}

inline bool is_image_generation_model(const std::string &value) {
    return to_diffusion_model_type(value) == DiffusionModelType::ZIMAGE;
}

inline bool is_video_generation_model(const std::string &value) {
    return to_diffusion_model_type(value) == DiffusionModelType::WAN_2_1;
}

inline std::string diffusion_model_type_to_string(DiffusionModelType model_type) {
    for (const auto& pair : diffusion_model_types_map) {
        if (pair.second == model_type) {
            return pair.first;
        }
    }
    return "unknown";
}

}