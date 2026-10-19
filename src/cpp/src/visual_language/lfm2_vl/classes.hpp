// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderLFM2VL : public VisionEncoder {
public:
    VisionEncoderLFM2VL(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderLFM2VL(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

private:
    // Static position-embedding table submodel (openvino_vision_embeddings_pos_model).
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_pos;
    // Vision-to-text projector submodel (openvino_multi_modal_projector_model).
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_projector;
    VLMConfig m_vlm_config;
};

class InputsEmbedderLFM2VL : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLFM2VL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderLFM2VL(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;
};

} // namespace ov::genai
