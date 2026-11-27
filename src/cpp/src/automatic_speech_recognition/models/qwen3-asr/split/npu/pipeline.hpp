// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "automatic_speech_recognition/models/qwen3-asr/split/pipeline.hpp"

namespace ov::genai {

class Qwen3ASRNPUVLMPipelineAdapter final : public Qwen3ASRVLMPipelineAdapter {
public:
    Qwen3ASRNPUVLMPipelineAdapter(const std::filesystem::path& models_path, const ov::AnyMap& properties);
};

}  // namespace ov::genai
