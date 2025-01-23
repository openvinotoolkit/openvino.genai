// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>

#include "openvino/genai/visibility.hpp"
#include <openvino/core/except.hpp>

namespace ov::genai {

enum class VLMModelType {
    MINICPM,
    LLAVA,
    LLAVA_NEXT,
    INTERNVL_CHAT,
    PHI3_V,
    QWEN2_VL,
};

inline VLMModelType to_vlm_model_type(const std::string& value) {
    static const std::unordered_map<std::string, VLMModelType> model_types_map = {
        {"minicpmv", VLMModelType::MINICPM},
        {"llava", VLMModelType::LLAVA},
        {"llava_next", VLMModelType::LLAVA_NEXT},
        {"internvl_chat", VLMModelType::INTERNVL_CHAT},
        {"phi3_v", VLMModelType::PHI3_V},
        {"qwen2_vl", VLMModelType::QWEN2_VL}
    };

    auto it = model_types_map.find(value);
    if (it != model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' VLM model type");
}
}