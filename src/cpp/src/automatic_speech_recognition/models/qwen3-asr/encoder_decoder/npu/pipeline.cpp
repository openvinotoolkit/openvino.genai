// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include "decoder.hpp"
#include "encoder.hpp"

namespace ov::genai {

Qwen3EncoderDecoderASRNPU::Qwen3EncoderDecoderASRNPU(const std::filesystem::path& models_path,
                                                     const ov::AnyMap& properties)
    : Qwen3EncoderDecoderASR(models_path, properties) {
    auto properties_copy = properties;
    erase_allowed_asr_ctor_properties(properties_copy);
    m_max_asr_input_seconds = 60;
    auto encoder = std::make_unique<Qwen3ASREncoderNPU>(models_path, properties_copy);
    const auto max_encoder_len = encoder->get_max_encoder_len(m_max_asr_input_seconds + CHUNK_BOUNDARY_SEARCH_SECONDS,
                                                              m_feature_extractor.sampling_rate,
                                                              m_feature_extractor.hop_length);
    m_encoder = std::move(encoder);
    m_decoder = std::make_unique<Qwen3ASRDecoderNPU>(models_path, properties_copy, max_encoder_len);
    m_decoder->set_seed(m_generation_config.rng_seed);
}

std::unique_ptr<ASRPipelineImplBase> create_qwen3_asr_encoder_decoder_pipeline(const std::filesystem::path& models_path,
                                                                               const std::string& device,
                                                                               const ov::AnyMap& properties) {
    if (device == "NPU") {
        return std::make_unique<Qwen3EncoderDecoderASRNPU>(models_path, properties);
    }
    return std::make_unique<Qwen3EncoderDecoderASR>(models_path, device, properties);
}

}  // namespace ov::genai
