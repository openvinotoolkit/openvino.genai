// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <mutex>

#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

// MiniCPM-V-4.6 (model_type "minicpmv4_6").
//
// Architecture differs fundamentally from the classic MiniCPM-V (2.x/o) family:
//  - Vision tower is a NaViT-style SigLIP encoder with window/merge downsampling.
//    The exported openvino_vision_embeddings_model consumes NaViT-patchified
//    ``pixel_values`` together with ``position_ids``/``window_index``/``merge_index``
//    and directly returns the merged visual tokens (no separate resampler).
//  - Language backbone is a Qwen3.5 text model that uses sequential position ids
//    ([1, tokens] for MiniCPM-V-4.6, repeated over three axes for MiniCPM-OCR)
//    and merges visual tokens by replacing the ``<|image_pad|>`` placeholder ids
//    with the vision features (masked scatter), like LLaVA / InternVL.
class VisionEncoderMiniCPMV4_6 : public VisionEncoder {
public:
    VisionEncoderMiniCPMV4_6(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderMiniCPMV4_6(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

    void update_vision_config(const VLMConfig& vlm_config);
};

class InputsEmbedderMiniCPMV4_6 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderMiniCPMV4_6(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderMiniCPMV4_6(
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

    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(
        const size_t inputs_embeds_size,
        const size_t history_size
    ) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

private:
    std::string build_image_placeholder(const EncodedImage& image, size_t image_id) const;
    void encode_vision_token_ids();

    std::once_flag m_vision_token_ids_once_flag;
    int64_t m_image_token_id = -1;
};

} // namespace ov::genai
