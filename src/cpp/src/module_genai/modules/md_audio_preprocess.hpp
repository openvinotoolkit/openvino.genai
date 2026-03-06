// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>
#include <variant>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "module_genai/modules/models/whisper/feature_extraction_whisper.hpp"

namespace ov {
namespace genai {
namespace module {
class AudioPreprocessModule : public IBaseModule {
    DeclareModuleConstructor(AudioPreprocessModule);

private:
    VLMModelType _model_type;
    std::shared_ptr<WhisperFeatureExtractor> m_feature_extractor_ptr = nullptr;

    void preprocess_audio(const bool& has_audios_input);
};

REGISTER_MODULE_CONFIG(AudioPreprocessModule);
}  // namespace module
}  // namespace genai
}  // namespace ov
