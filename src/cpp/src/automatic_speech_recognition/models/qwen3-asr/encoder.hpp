// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "config.hpp"
#include "openvino/core/core.hpp"
#include "openvino/runtime/runtime.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

class Qwen3ASREncoder {
public:
    Qwen3ASREncoder(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    ov::Tensor encode(const WhisperFeatures& features);

private:
    InferRequest m_request;
    Qwen3ASRConfig m_model_config;

    // The original Qwen3-ASR encoder processes mel spectrograms in chunks of N_WINDOW*2=200 frames,
    // applies positional embeddings per-chunk (positions 0..24), and uses windowed attention.
    const size_t m_encoder_chunk_frames = m_model_config.n_window * 2;

    ov::Tensor chunk_mel_features(const WhisperFeatures& features);

    size_t get_remainder_output_tokens(const size_t remainder_frames, const size_t tokens_per_full_chunk);

    ov::Tensor merge_chunked_encoder_output(const ov::Tensor& chunked_output, const size_t remainder_frames);
};

}  // namespace ov::genai
