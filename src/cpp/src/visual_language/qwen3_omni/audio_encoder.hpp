// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "whisper/feature_extractor.hpp"
#include "circular_buffer_queue.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

/// @brief Audio encoder for Qwen3-Omni. Converts raw PCM audio into
/// audio feature embeddings at the thinker's hidden dimension.
class AudioEncoderQwen3Omni {
public:
    /// @brief Maximum accepted audio length in samples (30 minutes at 16 kHz).
    /// Inputs above this are rejected at the boundary so that downstream allocations
    /// (mel spectrogram buffers, padded chunks) have a known upper bound and cannot
    /// overflow on attacker-controlled tensor shapes.
    static constexpr size_t MAX_AUDIO_SAMPLES = 480'000'000;

    AudioEncoderQwen3Omni(const std::filesystem::path& model_dir,
                          const VLMConfig& config,
                          const std::string& device,
                          const ov::AnyMap& properties);

    /// @brief Reject an audio shape/element-type combination that would overflow
    /// downstream allocations. Static so it can be called before tensor allocation
    /// (used by tests to exercise the size cap without OOMing).
    static void validate_audio_shape(const ov::Shape& shape, ov::element::Type element_type);

    /// @brief Validate that @p audio_raw is a 1-D float32 tensor in
    /// [1, MAX_AUDIO_SAMPLES] samples. Throws ov::Exception otherwise.
    /// Called at the start of preprocess_audio() — exposed publicly so callers
    /// (and tests) can validate inputs before incurring inference cost.
    static void validate_audio_input(const ov::Tensor& audio_raw);

    /// @brief Check whether the audio encoder model was found and loaded.
    /// The audio encoder is optional — the model works as text+vision VLM without it.
    /// Callers should check this before calling encode().
    bool is_available() const {
        return m_ireq_queue != nullptr;
    }

    /// @brief Encode raw PCM audio into audio feature embeddings.
    /// @param audio_raw 1-D float32 tensor of PCM samples at 16kHz.
    /// @return Audio features tensor [total_tokens, output_dim].
    ov::Tensor encode(const ov::Tensor& audio_raw);

private:
    VLMConfig m_config;
    WhisperFeatureExtractor m_feature_extractor;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue;

    /// @brief Preprocess audio: mel spectrogram -> chunk into windows -> pad.
    /// @return Tuple of (padded_feature, padded_mask_after_cnn, aftercnn_lens, cu_seqlens).
    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> preprocess_audio(const ov::Tensor& audio_raw);

    /// @brief Compute CNN output length after 8x total downsampling (three stride-2 Conv2d stages).
    static constexpr size_t get_feat_extract_output_length(size_t input_length) {
        // Three Conv2d layers with kernel=3, stride=2, padding=1: out = ceil(in / 2)
        size_t len = input_length;
        len = (len + 1) / 2;
        len = (len + 1) / 2;
        len = (len + 1) / 2;
        return len;
    }
};

}  // namespace ov::genai
