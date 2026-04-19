// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "models/decoder.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "sampling/sampler.hpp"
#include "whisper/config.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/whisper.hpp"

namespace ov {
namespace genai {

// Qwen3-ASR generate reuses WhisperGenerateResult for pipeline compatibility
WhisperGenerateResult qwen3_asr_generate(const ov::genai::WhisperGenerationConfig& config,
                                          const ov::genai::WhisperConfig& model_config,
                                          const RawSpeechInput& raw_speech,
                                          ov::InferRequest& encoder,
                                          std::shared_ptr<WhisperDecoder> decoder,
                                          WhisperFeatureExtractor& feature_extractor,
                                          const std::shared_ptr<StreamerBase> streamer,
                                          Sampler& sampler,
                                          Tokenizer& tokenizer);

}  // namespace genai
}  // namespace ov
