// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

// Preprocessing parameters for the JinaVLM (jvlm) Molmo-style image processor.
// Loaded from preprocessor_config.json.
struct JinaVLMProcParams {
    size_t patch_size = 14;
    size_t base_input_size = 378;   // both dims equal in known configs
    size_t max_crops = 12;
    size_t overlap_left = 4;
    size_t overlap_right = 4;
    size_t pooling_h = 2;
    size_t pooling_w = 2;
    size_t token_length_h = 14;
    size_t token_length_w = 14;
    size_t tokens_per_image = 196;
    bool use_column_tokens = true;
    std::array<float, 3> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
    std::array<float, 3> image_std = {0.26862954f, 0.26130258f, 0.27577711f};
    float image_min = -1.0f;
    float image_max = 1.0f;
    std::string normalization_method = "minmax";  // or "gaussian"
};

// Result of JinaVLM image preprocessing for a single image.
struct JinaVLMPreprocessed {
    // [n_crops, n_patches, patch*patch*3]
    ov::Tensor image_patches;
    // [n_crops, n_patches] (float mean of per-pixel mask, values in {-1, 1})
    ov::Tensor image_masks;
    // Number of crops (including the leading global thumbnail crop).
    size_t n_crops = 0;
    // Per-crop tiling of the non-thumbnail crops (rows, cols) used for prompt token layout.
    size_t tiling_rows = 1;
    size_t tiling_cols = 1;
    // Number of token rows/cols emitted for the tiled crops region.
    size_t crop_rows = 0;  // h // pooling_h
    size_t crop_cols = 0;  // w // pooling_w
};

class VisionEncoderJVLM : public VisionEncoder {
public:
    VisionEncoderJVLM(const std::filesystem::path& model_dir,
                      const std::string& device,
                      const ov::AnyMap properties);

    VisionEncoderJVLM(const ModelsMap& models_map,
                      const std::filesystem::path& config_dir_path,
                      const std::string& device,
                      const ov::AnyMap device_config);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

    const JinaVLMProcParams& get_jvlm_params() const { return m_params; }

private:
    void load_params(const std::filesystem::path& config_dir_path);
    JinaVLMProcParams m_params;
};

class InputsEmbedderJVLM : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderJVLM(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderJVLM(
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
        const std::vector<EncodedImage>& images) const override;

private:
    // JinaVLM-specific input-id construction.
    //
    // The JinaVLM chat template renders " User: <content> Assistant:" with a
    // leading space and relies on the tokenizer to prepend the BOS token
    // (<|endoftext|>, id from Tokenizer::get_bos_token_id). The exported
    // OpenVINO tokenizer neither preserves that leading space through
    // apply_chat_template nor adds the BOS token via add_special_tokens, so the
    // generic get_encoded_input_ids() path diverges from the reference
    // HF/optimum-intel tokenization. This helper reproduces the reference
    // tokenization exactly: it wraps the (already image-expanded) prompt in the
    // JinaVLM template when chat templating is requested, encodes with
    // add_special_tokens(false), and prepends the BOS token id.
    ov::Tensor build_jvlm_input_ids(const std::string& unified_prompt, ov::genai::VLMPerfMetrics& metrics);
};

}  // namespace ov::genai
