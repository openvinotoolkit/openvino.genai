// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "visual_language/processor_config.hpp"
#include "visual_language/video_processor_config.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

/// @brief In-memory models in the optimum-intel layout, keyed by ModelsMap names, with their configs.
struct VLMModels {
    std::map<std::string, std::shared_ptr<ov::Model>> models;
    VLMConfig config;
    ProcessorConfig processor_config;
    VideoProcessorConfig video_processor_config;

    std::shared_ptr<ov::Model> find(const std::string& name) const {
        const auto it = models.find(name);
        return it == models.end() ? nullptr : it->second;
    }

    const std::shared_ptr<ov::Model>& at(const std::string& name) const {
        const auto it = models.find(name);
        OPENVINO_ASSERT(it != models.end() && it->second, "VLM model '", name, "' is missing");
        return it->second;
    }
};

}  // namespace ov::genai
