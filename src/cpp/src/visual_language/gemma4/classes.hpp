// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

class VisionEncoderGemma4 : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderGemma4 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderGemma4(const VLMConfig& vlm_config,
                         const std::filesystem::path& model_dir,
                         const std::string& device,
                         const ov::AnyMap device_config);

    InputsEmbedderGemma4(const VLMConfig& vlm_config,
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

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_id,
                                      const std::vector<EncodedImage>& images) const override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size,
                                                                   const size_t history_size) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(const size_t inputs_embeds_size,
                                                                                    const size_t history_size,
                                                                                    int64_t rope_delta) override;

    const std::unordered_map<std::string, ov::Tensor>& get_lm_extra_inputs() const override;

    CircularBufferQueue<ov::InferRequest>* get_per_layer_embeddings_queue() const override {
        return m_per_layer_embeddings_requests.get();
    }

private:
    // Per-layer text embeddings model (Gemma4-specific)
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_per_layer_embeddings_requests;

    // Extra inputs to pass to the language model
    std::unordered_map<std::string, ov::Tensor> m_lm_extra_inputs;

    // Compute per-layer embeddings from input_ids
    ov::Tensor compute_per_layer_embeddings(const ov::Tensor& input_ids);
};

}  // namespace ov::genai
