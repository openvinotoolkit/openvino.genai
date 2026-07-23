// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"

namespace ov::genai {

// Youtu-VL uses a SigLIP2 "naflex" packed vision front-end that produces
// pre-patchified pixel_values of shape [num_patches, patch_size*patch_size*channels]
// together with per-image spatial_shapes (in patches). Downstream of the
// vision_embeddings model, the window-attention merger is identical to Qwen2.5-VL,
// so this encoder only overrides the preprocessing + vision_embeddings inference.
class VisionEncoderYoutuVL : public VisionEncoder {
public:
    explicit VisionEncoderYoutuVL(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties);
    explicit VisionEncoderYoutuVL(const ModelsMap& models_map, const std::filesystem::path& config_dir_path, const std::string& device, const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

// Youtu-VL reuses the Qwen2.5-VL vision merger path but its language model uses
// plain 1D position ids (no 3D mRoPE). Only position-id generation is overridden.
class InputsEmbedderYoutuVL : public InputsEmbedderQwen2_5_VL {
public:
    InputsEmbedderYoutuVL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderYoutuVL(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) override;

protected:
    std::pair<ov::Tensor, int64_t> create_position_ids(
        const ov::Tensor& input_ids_tensor,
        const std::vector<std::array<size_t, 3>>& images_grid_thw,
        const std::vector<size_t>& images_sequence,
        const size_t image_id,
        const std::vector<std::array<size_t, 3>>& videos_grid_thw,
        const std::vector<size_t>& videos_sequence,
        const size_t video_id,
        const int64_t vision_start_token_id,
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
    ) override;
};

namespace youtu_vl_utils {

// SigLIP2 naflex resize: scale image so that num_patches <= max_num_patches, with
// both dimensions rounded up to a multiple of patch_size*2.
ImageSize get_image_size_for_patches(size_t image_height, size_t image_width, size_t patch_size, size_t max_num_patches);

} // namespace youtu_vl_utils

} // namespace ov::genai
