// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "automatic_speech_recognition/models/sensevoice/input_limits.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"
#include "config.hpp"
#include "feature_extractor.hpp"
#include "openvino/runtime/infer_request.hpp"

namespace ov::genai {

class SenseVoiceSmall : public ASRPipelineImplBase {
public:
    SenseVoiceSmall(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    ASRDecodedResults generate(const AudiosView& audios,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer = nullptr) override;

    void set_generation_config(const ASRGenerationConfig& config) override;

private:
    ASRGenerationConfig resolve_generation_config(const std::optional<ASRGenerationConfig>& generation_config) const;
    void validate_generation_config(const ASRGenerationConfig& config) const;

    // Runs the model on `count` audios starting at `begin` and returns the inference duration in microseconds.
    float run_model(const std::vector<ov::Tensor>& features,
                    size_t begin,
                    size_t count,
                    int64_t language_id,
                    int64_t textnorm_id);

    SenseVoiceSmallConfig m_config;
    SenseVoiceSmallFeatureExtractor m_feature_extractor;
    ov::InferRequest m_request;
    SenseVoiceInputLimits m_limits;
};

}  // namespace ov::genai
