// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <unordered_map>
#include <openvino/runtime/properties.hpp>

namespace ov::genai {

enum class VideoGenerationModelType {
    UNKNOWN,
    WAN_2_1
};

static const std::unordered_map<std::string, VideoGenerationModelType> video_generation_model_types_map = {
        {"wan2.1", VideoGenerationModelType::WAN_2_1},
};

inline VideoGenerationModelType to_video_generation_model_type(const std::string &value) {
    auto it = video_generation_model_types_map.find(value);
    if (it != video_generation_model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' video generation model type");
}

inline bool is_video_generation_model(const std::string &value) {
    return video_generation_model_types_map.find(value) != video_generation_model_types_map.end();
}

}