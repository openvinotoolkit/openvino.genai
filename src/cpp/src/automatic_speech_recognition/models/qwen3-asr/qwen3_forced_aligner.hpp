// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "encoder.hpp"
#include "forced_aligner_config.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

class Qwen3ForcedAligner {
public:
    Qwen3ForcedAligner(const std::filesystem::path& models_path,
                       const std::string& device,
                       const ov::AnyMap& properties);

    // Returns word timestamps relative to the start of the supplied audio.
    std::vector<ASRDecodedResultChunk> align(const std::vector<float>& audio,
                                             const std::string& transcript,
                                             const ov::AnyMap& properties = {});

private:
    std::string resolve_language(const std::string& language) const;
    std::string build_marker_input(const std::vector<std::string>& units, size_t audio_frames) const;

    WhisperFeatureExtractor m_feature_extractor;
    Tokenizer m_tokenizer;
    Qwen3ForcedAlignerConfig m_config;
    std::unique_ptr<Qwen3ASREncoder> m_encoder;
    ov::InferRequest m_decoder;

    std::set<std::string> m_supported_languages;
};

}  // namespace ov::genai
