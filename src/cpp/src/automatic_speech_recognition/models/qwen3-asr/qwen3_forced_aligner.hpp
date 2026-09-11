// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "qwen3_forced_aligner_encoder.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

// Normalizes a recognized language name or code to the canonical name used by the forced aligner.
std::string normalize_alignment_language(const std::string& language);

// Segments transcript text into the language-specific units expected by the Qwen forced aligner.
std::vector<std::string> segment_alignment_units(const std::string& text, const std::string& canonical_language);

// Repairs non-monotonic timestamp predictions using the Qwen forced-aligner reference algorithm.
std::vector<int64_t> fix_timestamp(const std::vector<int64_t>& data);

class Qwen3ForcedAligner {
public:
    Qwen3ForcedAligner(const std::filesystem::path& models_path,
                       const std::string& device,
                       const ov::AnyMap& properties);

    // Aligns the transcript to the supplied audio and returns word-level timestamps relative to the start of that audio.
    std::vector<ASRDecodedResultChunk> align(const std::vector<float>& audio,
                                             const std::string& transcript,
                                             const std::string& language);

private:
    std::string resolve_language(const std::string& language) const;
    std::string build_marker_input(const std::vector<std::string>& units, size_t audio_frames) const;

    WhisperFeatureExtractor m_feature_extractor;
    Tokenizer m_tokenizer;
    std::unique_ptr<Qwen3ForcedAlignerEncoder> m_encoder;
    ov::InferRequest m_decoder;

    int64_t m_timestamp_token_id;
    float m_timestamp_segment_ms;
    std::set<std::string> m_supported_languages;
};

}  // namespace ov::genai
