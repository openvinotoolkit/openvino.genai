// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"
#include "circular_buffer_queue.hpp"

namespace ov::genai {

// PaddleOCR-VL vision encoder.
//
// Runs the Conv2d patch-embedding sub-model (openvino_vision_embeddings_model).
// Preprocessing (smart_resize -> bicubic resize -> rescale/normalize -> raster 14x14 patch
// extraction) is performed on the host and mirrors PaddleOCRVLImageProcessor. The resulting
// per-patch hidden states [num_patches, hidden] are stored in EncodedImage::resized_source and
// the (grid_h, grid_w) patch grid in EncodedImage::resized_source_size. Position-embedding
// interpolation, vision rotary embeddings, block-diagonal attention mask, the 2x2 spatial-merge
// permutation and the SigLIP encoder + projector are applied later in
// InputsEmbedderPaddleOCRVL::run_video_image_embeddings_merger.
class VisionEncoderPaddleOCRVL : public VisionEncoder {
public:
    explicit VisionEncoderPaddleOCRVL(const std::filesystem::path& model_dir,
                                      const std::string& device,
                                      const ov::AnyMap properties);
    explicit VisionEncoderPaddleOCRVL(const ModelsMap& models_map,
                                      const std::filesystem::path& config_dir_path,
                                      const std::string& device,
                                      const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

// PaddleOCR-VL inputs embedder.
//
// The language model uses the same 3D M-RoPE contract as Qwen2-VL (a vision_start token precedes
// the image placeholder run), so position-id construction and the text/vision embedding merge are
// reused verbatim from InputsEmbedderQwen2VL. Only the vision merger pipeline and prompt
// normalization differ and are overridden here.
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

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t image_base_id,
        size_t video_base_id,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos) const override;

protected:
    // Chat template emits <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|> for each image.
    inline static const std::string NATIVE_TAG_PADDLE = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>";

    // Position-embedding table sub-model (nn.Embedding). Output: [num_positions, hidden].
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_embeddings_pos;

    // Materialized [num_positions, hidden] position-embedding table, cached after first use.
    mutable ov::Tensor m_pos_embed_table;

    std::pair<ov::Tensor, ov::Tensor> run_video_image_embeddings_merger(
        const std::vector<EncodedImage>& images,
        const std::vector<size_t>& images_sequence,
        const std::vector<EncodedVideo>& videos,
        const std::vector<size_t>& videos_sequence) override;

    // Vision rotary embeddings with merge_size = 1 (raster (row, col) order); PaddleOCR merges
    // patches in the projector, not the encoder.
    ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) const override;

private:
    void init_pos_model(const std::shared_ptr<ov::Model>& pos_model,
                        const std::string& device,
                        const ov::AnyMap& device_config);

    // Materialize the full position-embedding table by running the pos sub-model over arange(num_positions).
    const ov::Tensor& get_pos_embed_table() const;

    // Adds bilinearly interpolated (align_corners=False) position embeddings into hidden_states in place.
    void add_interpolated_pos_embeds(const std::vector<std::array<size_t, 3>>& grids_thw,
                                     ov::Tensor& hidden_states) const;

    // Precomputes the raw-raster -> 2x2-block gather permutation consumed by the projector.
    ov::Tensor get_merge_index(const std::vector<std::array<size_t, 3>>& grids_thw) const;

    size_t m_num_grid_per_side = 0;  // image_size // patch_size
};

}  // namespace ov::genai
