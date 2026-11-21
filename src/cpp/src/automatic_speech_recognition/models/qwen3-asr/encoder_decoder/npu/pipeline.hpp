// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pipeline.hpp"

namespace ov::genai {

class Qwen3EncoderDecoderASRNPU final : public Qwen3EncoderDecoderASR {
public:
    Qwen3EncoderDecoderASRNPU(const std::filesystem::path& models_path, const ov::AnyMap& properties);
};

std::unique_ptr<ASRPipelineImplBase> create_qwen3_asr_encoder_decoder_pipeline(const std::filesystem::path& models_path,
                                                                               const std::string& device,
                                                                               const ov::AnyMap& properties);

}  // namespace ov::genai
