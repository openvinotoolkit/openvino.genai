// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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

    SenseVoiceSmallConfig m_config;
    SenseVoiceSmallFeatureExtractor m_feature_extractor;
    ov::InferRequest m_request;
    // Limits of the compiled model inputs. Audio longer than `m_max_window_samples` is split into windows.
    size_t m_max_batch = std::numeric_limits<size_t>::max();
    size_t m_padded_frames = 0;
    size_t m_max_window_samples = std::numeric_limits<size_t>::max();
};

}  // namespace ov::genai
