// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <map>
#include <openvino/runtime/tensor.hpp>
#include <openvino/core/except.hpp>

namespace ov::genai {

using VLMModelWeightsPair = std::pair<std::string, ov::Tensor>;
using VLMModelsMap = std::map<std::string, VLMModelWeightsPair>;

inline const VLMModelWeightsPair& get_model_weights_pair(const VLMModelsMap& models_map, const std::string& key) {
    auto it = models_map.find(key);
    if (it != models_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Model with key '", key, "' not found in VLM models map.");
}

} // namespace ov::genai
