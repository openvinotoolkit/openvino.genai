// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

    // The original Qwen3-ASR encoder processes mel spectrograms in chunks of N_WINDOW*2=200 frames,
    // applies positional embeddings per-chunk (positions 0..24), and uses windowed attention.
    static constexpr size_t ENCODER_CHUNK_FRAMES = 200;

    ov::Tensor chunk_mel_features(const WhisperFeatures& features);

    static size_t infer_output_frames(const size_t input_frames, const size_t full_chunk_output_frames);

    ov::Tensor merge_chunked_encoder_output(const ov::Tensor& chunked_output, const size_t remainder_frames);
};

}  // namespace ov::genai
