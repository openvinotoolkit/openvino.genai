// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "audio_utils.hpp"
#include "circular_buffer_queue.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

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
    audio_utils::MelSpectrogramExtractor m_mel_extractor;
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
