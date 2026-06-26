// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "circular_buffer_queue.hpp"

namespace ov::genai {

// Youtu-VL uses a windowed Siglip2 vision tower:
//   - openvino_vision_embeddings_model: patch-embedding (pixel_values [1, N, 768] -> hidden_states [N, vis_hidden]).
//   - openvino_vision_embeddings_merger_model: Qwen2.5-VL style windowed transformer + merger
//     (hidden_states [N, vis_hidden], attention_mask, window_attention_mask, window_index, rotary_pos_emb -> [N/merge^2, hidden]).
// The language model is a standard (2D position_ids) MLA decoder, so the default
// sequential position ids from IInputsEmbedder are reused (no Qwen-style mrope).
class VisionEncoderYoutuVL : public VisionEncoder {
public:
    explicit VisionEncoderYoutuVL(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties);
    explicit VisionEncoderYoutuVL(const ModelsMap& models_map, const std::filesystem::path& config_dir_path, const std::string& device, const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

private:
    void init_preprocess_model(const std::string& device, const ov::AnyMap& properties);
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_preprocess;
};

class InputsEmbedderYoutuVL : public InputsEmbedder::IInputsEmbedder {
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

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

protected:
    // Chat template hardcodes the char sequence, so NATIVE_TAG is hardcoded as well.
    inline static const std::string NATIVE_TAG = "<|vision_start|><|image_pad|><|vision_end|>";

    void init_merger(const std::shared_ptr<ov::Model>& merger_model, const std::string& device, const ov::AnyMap& device_config);

    // The exported OpenVINO tokenizer does not treat the Youtu-VL added/special tokens
    // (e.g. <|image_pad|>, <|vision_start|>, <|begin_of_text|>) as atomic ids; it splits
    // them into character-level subtokens. This breaks image-placeholder counting and the
    // language-model framing. To match the HuggingFace tokenizer we load the special tokens
    // from tokenizer.json and tokenize prompts span-by-span, splicing the correct ids back.
    void load_special_tokens(const std::filesystem::path& config_dir);
    ov::Tensor encode_with_special_tokens(const std::string& text, bool add_special_tokens);
    ov::Tensor apply_chat_template_tokenize(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) override;

    // Runs the windowed vision transformer/merger on a single image's patch embeddings.
    ov::Tensor run_image_embeddings_merger(const ov::Tensor& hidden_states, size_t grid_h, size_t grid_w);

    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_embeddings_merger;

    // Maps special-token text (e.g. "<|image_pad|>") to its token id from tokenizer.json.
    std::unordered_map<std::string, int64_t> m_special_tokens;

    size_t m_merge_size = 2;
    size_t m_patch_size = 16;
    size_t m_rope_dim = 16;       // (vision_hidden / num_heads) / 2
    int64_t m_image_pad_token_id = -1;
    int64_t m_bos_token_id = -1;
};

} // namespace ov::genai
