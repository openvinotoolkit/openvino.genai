// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "openvino/core/core.hpp"
#include "openvino/runtime/runtime.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

// Audio encoder for the Qwen3 forced aligner. It uses the forced aligner's own model and configuration,
// independently of the base Qwen3-ASR encoder and its exported-model layout.
class Qwen3ForcedAlignerEncoder {
public:
    Qwen3ForcedAlignerEncoder(const std::filesystem::path& models_path,
                              const std::string& device,
                              const ov::AnyMap& properties);

    ov::Tensor encode(const WhisperFeatures& features);

private:
    InferRequest m_request;

    // Number of mel frames per encoder chunk. The exported aligner encoder expects n_window * 2 frames per chunk,
    // with positional embeddings applied independently to each chunk. n_window comes from the aligner's config.json.
    const size_t m_encoder_chunk_frames;

    ov::Tensor chunk_mel_features(const WhisperFeatures& features) const;

    size_t get_remainder_output_tokens(size_t remainder_frames, size_t tokens_per_full_chunk) const;

    ov::Tensor merge_chunked_encoder_output(const ov::Tensor& chunked_output, size_t remainder_frames) const;
};

}  // namespace ov::genai
