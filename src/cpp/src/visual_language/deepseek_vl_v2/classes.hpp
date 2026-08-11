// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vision_encoder.hpp"

namespace ov::genai {

class VisionEncoderDeepseekVLV2 : public VisionEncoder {
public:
    VisionEncoderDeepseekVLV2(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderDeepseekVLV2(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

private:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_encoder_tiles;
    VLMConfig m_vlm_config;
};

class InputsEmbedderDeepseekVLV2 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderDeepseekVLV2(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderDeepseekVLV2(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<ov::genai::EncodedImage>& images,
                                 ov::genai::VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {}) override;

    ov::Tensor apply_chat_template_tokenize(const std::string& prompt,
                                            ov::genai::VLMPerfMetrics& metrics) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_id,
                                      const std::vector<EncodedImage>& images) const override;

private:
    int64_t m_image_token_id = -1;
};

}  // namespace ov::genai
