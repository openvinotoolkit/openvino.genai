// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "automatic_speech_recognition/models/qwen3-asr/split/pipeline.hpp"

namespace ov::genai {

class Qwen3ASRNPUVLMPipelineAdapter final : public Qwen3ASRVLMPipelineAdapter {
public:
    Qwen3ASRNPUVLMPipelineAdapter(const std::filesystem::path& models_path, const ov::AnyMap& properties);

private:
    void validate_generation_config(const ASRGenerationConfig& config) const override;
};

}  // namespace ov::genai
