// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "forced_aligner_config.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "split/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Qwen3 forced aligner for the audio encoder + text embeddings + language model export.
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

    Qwen3ForcedAlignerConfig m_config;
    Tokenizer m_tokenizer;
    InputsEmbedderQwen3ASR m_embedder;
    ov::InferRequest m_language_model;

    std::set<std::string> m_supported_languages;
};

}  // namespace ov::genai
