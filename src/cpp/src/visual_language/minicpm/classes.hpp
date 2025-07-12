// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderMiniCPM : public VisionEncoder {
private:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> create_ireq(ov::CompiledModel& compiled_model);
    // VLM config
    VLMConfig m_vlm_config;

public:
    VisionEncoderMiniCPM(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderMiniCPM(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);
    
    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderMiniCPM : public InputsEmbedder::IInputsEmbedder {
protected:
    // A model for merging image embeddings with text embeddings
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_embeddings_merger;
    
    ov::Tensor m_merged_image_embeddings;
    
    virtual ov::Tensor run_image_embeddings_merger(
        const std::vector<EncodedImage>& images,
        const std::vector<size_t>& images_sequence);
    
    ov::Tensor create_position_ids(
        const ov::Tensor& input_ids_tensor,
        const std::vector<size_t>& images_sequence,
        const int64_t vision_start_token_id);

public:
    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    std::pair<std::string, std::vector<size_t>> normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;
};

namespace minicpm_utils {

std::pair<std::vector<ov::Tensor>, std::vector<ImageSize>> reorder_image_embeds_and_sizes(
    const std::vector<EncodedImage>& encoded_images,
    const std::vector<size_t>& images_sequence
);

ov::Tensor concatenate_image_embeds(const std::vector<ov::Tensor>& reordered_image_embeds);

} // namespace minicpm_utils

} // namespace ov::genai