// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/**
 * @class VisionEncoderPhi4MM
 * @brief A specialized vision encoder for the Phi4MM model.
 *
 * This class is responsible for encoding images into a format suitable for
 * multimodal processing in the Phi4MM model. It supports initialization
 * with model directories or preloaded models and provides an interface
 * for encoding images.
 */
class VisionEncoderPhi4MM : public VisionEncoder {
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_image_preprocessors;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_separator_inserters;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_projection;
    VLMConfig m_vlm_config;

public:
    VisionEncoderPhi4MM(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderPhi4MM(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderPhi4MM : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderPhi4MM(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config
    );

    InputsEmbedderPhi4MM(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& image_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    void update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    NormlizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

private:
    std::vector<size_t> m_tokens_per_images;
    std::vector<size_t> m_prev_tokens_per_images;
};

} // namespace ov::genai
