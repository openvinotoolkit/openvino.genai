// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {
VLMModels read_lfm3_saved_models(const std::filesystem::path& directory, const ModelsMap& models_map = {});

class VisionEncoderLFM3 : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;
    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map = {}) override;
    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames) override;

private:
    ov::Tensor encode_pair(const ov::Tensor& first, const ov::Tensor& second, size_t height, size_t width);
};

class InputsEmbedderLFM3 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLFM3(const VLMModels& models,
                       const Tokenizer& tokenizer,
                       const std::string& device,
                       const ov::AnyMap& properties);
    InputsEmbedderLFM3(const VLMConfig& config,
                       const std::filesystem::path& directory,
                       const Tokenizer& tokenizer,
                       const std::string& device,
                       const ov::AnyMap& properties);
    InputsEmbedderLFM3(const VLMConfig& config,
                       const ModelsMap& models_map,
                       const Tokenizer& tokenizer,
                       const std::filesystem::path& directory,
                       const std::string& device,
                       const ov::AnyMap& properties);
    void set_audio_encoder(AudioEncode encoder) override {
        m_audio_encoder = std::move(encoder);
    }
    std::vector<EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;
    std::vector<EncodedAudio> encode_audios(const std::vector<ov::Tensor>& audios) override;
    std::vector<EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos,
                                            const std::vector<VideoMetadata>& metadata) override;
    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base,
                                      const std::vector<EncodedImage>& images) const override;
    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t image_base,
                                      size_t video_base,
                                      size_t audio_base,
                                      const std::vector<EncodedImage>& images,
                                      const std::vector<EncodedVideo>& videos,
                                      const std::vector<EncodedAudio>& audios) const override;
    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<EncodedImage>& images,
                                 VLMPerfMetrics& metrics,
                                 bool recalculate,
                                 const std::vector<size_t>& sequence) override;
    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<EncodedImage>& images,
                                 const std::vector<EncodedVideo>& videos,
                                 const std::vector<EncodedAudio>& audios,
                                 VLMPerfMetrics& metrics,
                                 bool recalculate,
                                 const std::vector<size_t>& images_sequence,
                                 const std::vector<size_t>& videos_sequence,
                                 const std::vector<size_t>& audios_sequence,
                                 const std::vector<std::pair<size_t, size_t>>& history_vision_count) override;

private:
    int64_t m_image_id = -1, m_video_id = -1, m_audio_id = -1;
    std::string m_vision_start, m_vision_end;
};
}  // namespace ov::genai
