// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <filesystem>

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
//  - Language backbone is a Qwen3.5 text model that uses plain 1D position ids
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

private:
    void load_vision_params(const std::filesystem::path& config_dir_path);

    size_t m_num_patches_per_side = 70;  // vision_config.image_size / patch_size
    size_t m_patch_size = 14;
    size_t m_scale_resolution = 448;
    size_t m_max_slice_nums = 9;
    size_t m_window_kh = 2;
    size_t m_window_kw = 2;
    size_t m_merge_kh = 2;
    size_t m_merge_kw = 2;
    size_t m_token_divisor = 16;  // 16x downsample -> grid_h*grid_w/16 tokens
    bool m_slice_mode = true;
    std::array<float, 3> m_image_mean{0.5f, 0.5f, 0.5f};
    std::array<float, 3> m_image_std{0.5f, 0.5f, 0.5f};
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

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

private:
    std::string build_image_placeholder(const EncodedImage& image, size_t image_id) const;
};

} // namespace ov::genai
