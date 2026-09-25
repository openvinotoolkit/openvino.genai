// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

// Preprocessing parameters for the LFM2-VL naflex SigLIP2 vision tower.
// Parsed from processor_config.json / preprocessor_config.json.
struct Lfm2VlPreprocessConfig {
    size_t downsample_factor = 2;
    size_t encoder_patch_size = 16;
    size_t tile_size = 512;
    size_t min_image_tokens = 64;
    size_t max_image_tokens = 256;
    size_t min_tiles = 2;
    size_t max_tiles = 10;
    bool do_image_splitting = true;
    bool use_thumbnail = true;
    float max_pixels_tolerance = 2.0f;
    std::array<float, 3> image_mean{0.5f, 0.5f, 0.5f};
    std::array<float, 3> image_std{0.5f, 0.5f, 0.5f};
    // Number of learned positional-embedding patches of the vision tower
    // (num_patches from vision_config); source grid is sqrt(num) x sqrt(num).
    size_t vision_num_patches = 256;

    size_t max_num_patches() const {
        size_t max_thumbnail_patches = max_image_tokens * downsample_factor * downsample_factor;
        size_t tile_patches = do_image_splitting ? (tile_size / encoder_patch_size) * (tile_size / encoder_patch_size) : 0;
        return std::max(max_thumbnail_patches, tile_patches);
    }
};

class VisionEncoderLFM2VL : public VisionEncoder {
public:
    VisionEncoderLFM2VL(const std::filesystem::path& model_dir,
                        const std::string& device,
                        const ov::AnyMap properties);

    VisionEncoderLFM2VL(const ModelsMap& models_map,
                        const std::filesystem::path& config_dir_path,
                        const std::string& device,
                        const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

protected:
    // Infer requests queue for the multi_modal_projector model.
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_projector;
    Lfm2VlPreprocessConfig m_preprocess_config;

    void load_preprocess_config(const std::filesystem::path& config_dir_path);
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

    // The LFM2 language model handles token positions internally (hybrid short-conv +
    // attention) and its exported IR does not expose a position_ids input. Return an
    // empty position_ids tensor so the LM encoding loop skips setting that tensor.
    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size) override;
};

} // namespace ov::genai
