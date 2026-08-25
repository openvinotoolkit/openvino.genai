// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/**
 * @brief Vision encoder for HunYuanVL (e.g. tencent/HunyuanOCR).
 *
 * HunYuanVL exports the whole vision tower (patch embedding, learned positional grid, transformer
 * blocks and the spatial patch merger with begin/newline/end tokens) as a single OpenVINO model.
 * The graph takes the flattened image patches, an additive attention mask and a grid-shaped tensor
 * that only communicates the patch grid (height, width) to the graph. The output already contains
 * the merged, LLM-space image embeddings (one row per LLM image token).
 */
class VisionEncoderHunyuanVL : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderHunyuanVL : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderHunyuanVL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderHunyuanVL(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

private:
    // Number of multimodal RoPE axes consumed by the text backbone (config.text_config.rope_parameters.mrope_section).
    // HunYuanVL uses the last three axes for (width, height, image_index); the remaining leading axis keeps the
    // plain 1D sequence position.
    size_t m_num_mrope_axes = 4;
    // Special token ids used to build and merge the expanded image placeholder run.
    int64_t m_image_token_id = -1;
    int64_t m_image_start_token_id = -1;
    int64_t m_image_end_token_id = -1;

    void init_token_ids();

    std::pair<ov::Tensor, int64_t> create_position_ids(
        const ov::Tensor& input_ids_tensor,
        const std::vector<std::array<size_t, 2>>& images_grid_hw,
        const std::vector<size_t>& images_sequence
    ) const;
};

} // namespace ov::genai
