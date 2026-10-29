// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>

#include "circular_buffer_queue.hpp"
#include "visual_language/lfm3_vl/classes.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

/// @brief LFM3-Omni speech encoder: 16 kHz mono PCM -> log-mel features -> one embedding per 8 mel frames.
class AudioEncoderLFM3 {
public:
    AudioEncoderLFM3(const std::shared_ptr<ov::Model>& model,
                     const std::filesystem::path& config_dir_path,
                     const std::string& device,
                     const ov::AnyMap& properties);

    /// @param audio 1-D f32 tensor of PCM samples at 16 kHz.
    /// @return Audio embeddings [num_audio_tokens, hidden_size].
    ov::Tensor encode(const ov::Tensor& audio);

private:
    struct FeatureConfig {
        size_t feature_size = 128;
        size_t sampling_rate = 16000;
        size_t n_fft = 400;
        size_t hop_length = 160;
        float log_floor = -12.0f;
    };

    static FeatureConfig read_feature_config(const std::filesystem::path& config_dir_path);

    AudioEncoderLFM3(const std::shared_ptr<ov::Model>& model,
                     const FeatureConfig& config,
                     const std::string& device,
                     const ov::AnyMap& properties);

    WhisperFeatureExtractor m_feature_extractor;
    float m_log_floor;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue;
};

class InputsEmbedderLFM3Omni : public InputsEmbedderLFM3VL {
public:
    InputsEmbedderLFM3Omni(const VLMConfig& vlm_config,
                           const std::filesystem::path& model_dir,
                           const Tokenizer& tokenizer,
                           const std::string& device,
                           const ov::AnyMap device_config);

    InputsEmbedderLFM3Omni(const VLMConfig& vlm_config,
                           const ModelsMap& models_map,
                           const Tokenizer& tokenizer,
                           const std::filesystem::path& config_dir_path,
                           const std::string& device,
                           const ov::AnyMap device_config);

    std::vector<EncodedAudio> encode_audios(const std::vector<ov::Tensor>& audios) override;

    using InputsEmbedderLFM3VL::normalize_prompt;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_image_id,
                                      size_t base_video_id,
                                      size_t base_audio_id,
                                      const std::vector<EncodedImage>& images,
                                      const std::vector<EncodedVideo>& videos,
                                      const std::vector<EncodedAudio>& audios) const override;

private:
    std::unique_ptr<AudioEncoderLFM3> m_audio_encoder;
};

}  // namespace ov::genai
