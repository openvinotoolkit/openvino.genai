// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

namespace phi_utils {

std::string normalize_prompt(
    const std::string& prompt, size_t base_id, size_t n_images, const std::regex& native_pattern, void(*write_native)(std::ostream& os, size_t idx)
);
std::vector<std::variant<ov::Tensor, size_t>> split_tokenize(const std::string& text, ov::genai::Tokenizer& tokenizer, const std::regex& native_pattern);
ov::Tensor insert_image_placeholders(const std::vector<std::variant<ov::Tensor, size_t>>& chunks, const std::vector<size_t>& tokens_per_images);
std::vector<std::variant<ov::Tensor, size_t>> drop_image_placeholders(const ov::Tensor& tokens);

}

class VisionEncoderPhi3V : public VisionEncoder {
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_hd_feature_transformer;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_projection;
    VLMConfig m_vlm_config;
public:
    VisionEncoderPhi3V(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderPhi3V(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderPhi3V : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderPhi3V(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config
    );

    InputsEmbedderPhi3V(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

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
