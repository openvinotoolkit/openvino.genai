// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

#include "visual_language/continuous_batching_adapter.hpp"

namespace ov {
namespace genai {
namespace module {

/// @brief Unified generation module supporting multiple modalities
/// Supports: LLM, VLM
class LLMInferenceModule : public IBaseModule {
    DeclareModuleConstructor(LLMInferenceModule);

private:
    std::shared_ptr<ov::Model> m_ov_model_embed = nullptr;
    bool initialize();
    bool load_generation_config(const std::filesystem::path& config_path);

    // Pipeline instances (only one will be initialized based on model type)
    std::shared_ptr<ov::genai::VLMPipeline::VLMContinuousBatchingAdapter> m_cb_pipeline;

    // Generation configurations
    ov::genai::GenerationConfig m_generation_config;
};

REGISTER_MODULE_CONFIG(LLMInferenceModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
