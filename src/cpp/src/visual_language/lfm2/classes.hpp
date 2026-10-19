// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderLFM2 : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;
    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map = {}) override;
};

class InputsEmbedderLFM2Vision : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLFM2Vision(const VLMModels& models,
                             const Tokenizer& tokenizer,
                             const std::string& device,
                             const ov::AnyMap& properties);
    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base,
                                      const std::vector<EncodedImage>& images) const override;
    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<EncodedImage>& images,
                                 VLMPerfMetrics& metrics,
                                 bool recalculate,
                                 const std::vector<size_t>& sequence) override;

private:
    int64_t m_image_token_id;
    std::string m_image_placeholder;
};

class InputsEmbedderLFM2Audio : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLFM2Audio(const VLMModels& models,
                            const Tokenizer& tokenizer,
                            const std::string& device,
                            const ov::AnyMap& properties);

    void set_audio_encoder(AudioEncode encoder) override {
        m_audio_encoder = std::move(encoder);
    }
    std::vector<EncodedAudio> encode_audios(const std::vector<ov::Tensor>& audios) override;
    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t image_base_id,
                                      const std::vector<EncodedImage>& images) const override {
        return normalize_prompt(prompt, image_base_id, 0, 0, images, {}, {});
    }
    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t image_base_id,
                                      size_t video_base_id,
                                      size_t audio_base_id,
                                      const std::vector<EncodedImage>& images,
                                      const std::vector<EncodedVideo>& videos,
                                      const std::vector<EncodedAudio>& audios) const override;
    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<EncodedImage>& images,
                                 VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings,
                                 const std::vector<size_t>& image_sequence) override;
    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<EncodedImage>& images,
                                 const std::vector<EncodedVideo>& videos,
                                 const std::vector<EncodedAudio>& audios,
                                 VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings,
                                 const std::vector<size_t>& image_sequence,
                                 const std::vector<size_t>& videos_sequence,
                                 const std::vector<size_t>& audios_sequence,
                                 const std::vector<std::pair<size_t, size_t>>& history_vision_count) override;

private:
    int64_t m_audio_token_id;
};

}  // namespace ov::genai
