// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "circular_buffer_queue.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

/// @brief Computes mel spectrogram from raw PCM audio for Qwen3-Omni audio encoder.
/// Adapts the Whisper mel spectrogram logic for Qwen3-Omni's parameters.
class MelSpectrogramExtractor {
public:
    /// @param num_mel_bins Number of mel filter bins (128 for Qwen3-Omni).
    /// @param sampling_rate Audio sampling rate in Hz (16000).
    /// @param n_fft FFT window size (400 for 25ms at 16kHz).
    /// @param hop_length Hop between frames (160 for 10ms at 16kHz).
    MelSpectrogramExtractor(size_t num_mel_bins = 128,
                            size_t sampling_rate = 16000,
                            size_t n_fft = 400,
                            size_t hop_length = 160);

    /// @brief Extract mel spectrogram from raw PCM float32 audio.
    /// @param raw_speech PCM samples at the configured sampling rate.
    /// @return Mel spectrogram with shape [num_mel_bins, n_frames], row-major.
    std::vector<float> extract(const std::vector<float>& raw_speech, size_t& n_frames) const;

private:
    const size_t m_num_mel_bins;
    const size_t m_sampling_rate;
    const size_t m_n_fft;
    const size_t m_hop_length;
    const std::vector<float> m_sin_vals;
    const std::vector<float> m_cos_vals;
    const std::vector<float> m_mel_filter;  // [num_mel_bins * (1 + n_fft/2)]
};

/// @brief Audio encoder for Qwen3-Omni. Converts raw PCM audio into
/// audio feature embeddings at the thinker's hidden dimension.
class AudioEncoderQwen3Omni {
public:
    AudioEncoderQwen3Omni(const std::filesystem::path& model_dir,
                          const VLMConfig& config,
                          const std::string& device,
                          const ov::AnyMap& properties);

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
    MelSpectrogramExtractor m_mel_extractor;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue;

    /// @brief Preprocess audio: mel spectrogram -> chunk into windows -> pad.
    /// @return Tuple of (padded_feature, padded_mask_after_cnn, aftercnn_lens, cu_seqlens).
    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> preprocess_audio(const ov::Tensor& audio_raw);

    /// @brief Compute CNN output length after 3x stride-2 conv2d downsampling (8x total).
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
