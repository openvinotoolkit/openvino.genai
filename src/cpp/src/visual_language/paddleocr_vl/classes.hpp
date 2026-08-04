// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "circular_buffer_queue.hpp"

namespace ov::genai {

// PaddleOCR-VL vision encoder.
//
// The exported IR has two vision sub-models:
//   - openvino_vision_embeddings_model:
//       IN  pixel_values [N, 3, 14, 14] f32
//           interp_h     [grid_h, 27]   f32   (bilinear interpolation matrix for pos-embed height)
//           interp_w     [grid_w, 27]   f32   (bilinear interpolation matrix for pos-embed width)
//       OUT last_hidden_state [N, 1152]
//   - openvino_vision_embeddings_merger_model (SigLIP-variant encoder + 2x2 projector):
//       IN  hidden_states [N, 1152] f32
//           attention_mask [1, N, N] f32
//           rope_emb_cos  [N, 72] f32
//           rope_emb_sin  [N, 72] f32
//           merge_index   [N] i64
//       OUT last_hidden_state [N/4, 1024]
//
// encode() runs both sub-models and returns the merged image embeddings in
// EncodedImage::resized_source with shape [num_image_tokens, hidden_size].
class VisionEncoderPaddleOCRVL : public VisionEncoder {
public:
    explicit VisionEncoderPaddleOCRVL(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties);
    explicit VisionEncoderPaddleOCRVL(const ModelsMap& models_map, const std::filesystem::path& config_dir_path, const std::string& device, const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

private:
    void init_merger(const std::shared_ptr<ov::Model>& merger_model, const std::string& device, const ov::AnyMap& properties);

    // Merger sub-model infer request queue.
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_merger;

    size_t m_num_positions = 0;      // (image_size / patch_size)^2, positions of the learned pos-embed grid
    size_t m_num_positions_side = 0; // image_size / patch_size (e.g. 27)
    size_t m_vision_head_dim = 0;    // vision_config.hidden_size / vision_config.num_attention_heads
};

// PaddleOCR-VL inputs embedder.
//
// The language decoder is Qwen2-VL-like (3D mrope, GQA, image-token masked
// merge), so this class inherits InputsEmbedderQwen2VL to reuse the position-id
// logic. It overrides prompt normalization, image encoding and embedding merge
// because:
//   - the vision placeholder token strings differ from Qwen2-VL, and
//   - image embeddings are already merged by the vision encoder (no separate
//     merger call at the embedding stage), so the embedder only splices them at
//     <|IMAGE_PLACEHOLDER|> positions.
class InputsEmbedderPaddleOCRVL : public InputsEmbedderQwen2VL {
public:
    InputsEmbedderPaddleOCRVL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderPaddleOCRVL(
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

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<ov::genai::EncodedImage>& images,
                                 const std::vector<ov::genai::EncodedVideo>& videos,
                                 ov::genai::VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {},
                                 const std::vector<size_t>& videos_sequence = {},
                                 const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override {
        auto norm_prompt = normalize_prompt(prompt, base_id, 0, images, {});
        return {norm_prompt.unified_prompt, norm_prompt.images_sequence};
    }

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t image_base_id,
        size_t video_base_id,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos) const override;

private:
    // Chat template hardcodes the placeholder char sequence, matching Qwen2-VL's approach.
    inline static const std::string PADDLEOCR_NATIVE_TAG = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>";

    void init_paddleocr_tokens();
};

} // namespace ov::genai
