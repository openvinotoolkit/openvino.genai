// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <unordered_map>

#include "automatic_speech_recognition/models/qwen3-asr/decoder.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"
#include "encoder.hpp"
#include "feature_extractor.hpp"

namespace ov::genai {

class FunASR : public ASRPipelineImplBase {
public:
    FunASR(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    ASRDecodedResults generate(const AudioInputs& audio_inputs,
                               const std::optional<ASRGenerationConfig>& generation_config,
                               const std::shared_ptr<StreamerBase> streamer = nullptr) override;

private:
    struct TokenizedInstructions {
        ov::Tensor prefix_ids;
        ov::Tensor suffix_ids;
    };

    FunASRFeatureExtractor m_feature_extractor;
    std::unique_ptr<FunASREncoder> m_encoder;
    std::unique_ptr<Qwen3ASRDecoder> m_decoder;
    std::mutex m_tokenized_instructions_mutex;
    std::unordered_map<std::string, TokenizedInstructions> m_tokenized_instructions;

    ov::Tensor build_input_ids(size_t num_audio_tokens, const ASRGenerationConfig& config);
    TokenizedInstructions get_tokenized_instructions(const ASRGenerationConfig& config);

    ASRGenerationConfig resolve_generation_config(const std::optional<ASRGenerationConfig>& generation_config) const;

    void validate_generation_config(const ASRGenerationConfig& config) const;
};

}  // namespace ov::genai
