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

    Tensor encode(std::vector<WhisperFeatures> features);

private:
    InferRequest m_request;

    // The original Qwen3-ASR encoder processes mel spectrograms in chunks of N_WINDOW*2=200 frames,
    // applies positional embeddings per-chunk (positions 0..24), and uses windowed attention.
    // The patched OV export removes chunking and uses full-sequence attention with global positional
    // embeddings — which degrades quality for long audio because positions beyond ~25 are untrained
    // and full attention is not what the model learned.
    //
    // Workaround: split mel features into 200-frame chunks, feed them as a batch to the encoder.
    // Each chunk gets correct per-chunk positional embeddings (0..24) and intra-chunk attention.
    static constexpr size_t ENCODER_CHUNK_FRAMES = 200;

    ov::Tensor chunk_mel_features(const WhisperFeatures& feat);

    static size_t conv_output_frames(const size_t input_frames);

    ov::Tensor merge_chunked_encoder_output(const ov::Tensor& chunked_output, const size_t remainder_frames);
};

}  // namespace ov::genai
