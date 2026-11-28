// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

/// @brief PaddleOCR-VL vision encoder.
/// The patch embeddings model takes single-frame patches (temporal_patch_size == 1) in spatial-merge-block order
/// together with the bilinear position embeddings gather indices and weights. The resulting hidden states are merged
/// by the Qwen2-VL vision embeddings merger flow of InputsEmbedderPaddleOCRVL.
class VisionEncoderPaddleOCRVL : public VisionEncoderQwen2VL {
public:
    VisionEncoderPaddleOCRVL(const std::filesystem::path& model_dir,
                             const std::string& device,
                             const ov::AnyMap properties);
    VisionEncoderPaddleOCRVL(const ModelsMap& models_map,
                             const std::filesystem::path& config_dir_path,
                             const std::string& device,
                             const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames) override;

private:
    void compile(const std::shared_ptr<ov::Model>& model, const std::string& device, const ov::AnyMap& properties);

    // Side of the learned square position embeddings table, vision_config.image_size / vision_config.patch_size.
    size_t m_num_grid_per_side = 0;
};

/// @brief PaddleOCR-VL inputs embedder: reuses the Qwen2-VL image token expansion, vision embeddings merger
/// and 3D M-RoPE position ids with the PaddleOCR-VL vision tokens.
class InputsEmbedderPaddleOCRVL : public InputsEmbedderQwen2VL {
public:
    InputsEmbedderPaddleOCRVL(const VLMConfig& vlm_config,
                              const std::filesystem::path& model_dir,
                              const Tokenizer& tokenizer,
                              const std::string& device,
                              const ov::AnyMap device_config);

    InputsEmbedderPaddleOCRVL(const VLMConfig& vlm_config,
                              const ModelsMap& models_map,
                              const Tokenizer& tokenizer,
                              const std::filesystem::path& config_dir_path,
                              const std::string& device,
                              const ov::AnyMap device_config);

    std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos,
                                                       const std::vector<VideoMetadata>& videos_metadata = {}) override;

private:
    void validate_vision_token_ids() const;
};

namespace paddleocr_vl_utils {

/// @brief PaddleOCR-VL variant of Qwen2-VL smart_resize: images smaller than factor are upscaled to factor first
/// and the downscaled sides never drop below factor.
ImageSize smart_resize(size_t height, size_t width, size_t factor, size_t min_pixels, size_t max_pixels);

/// @brief Bilinear position embeddings gather indices [4, N] (i64) and weights [4, N] (f32) for a single-frame
/// grid_h x grid_w patch grid, in spatial-merge-block order.
std::pair<ov::Tensor, ov::Tensor> get_bilinear_indices_and_weights(size_t grid_h,
                                                                   size_t grid_w,
                                                                   size_t num_grid_per_side,
                                                                   size_t merge_size);

}  // namespace paddleocr_vl_utils

}  // namespace ov::genai
