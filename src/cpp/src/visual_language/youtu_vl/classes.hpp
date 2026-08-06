// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "circular_buffer_queue.hpp"

namespace ov::genai {

// Vision encoder for tencent/Youtu-VL-4B-Instruct.
// The SigLIP2 windowed vision tower and the VLPatchMerger are exported as a single
// `openvino_vision_embeddings_model` that consumes materialized auxiliary tensors
// (attention_mask, window_attention_mask, window_index, rotary_pos_emb). These are
// recomputed here from the image spatial shape, mirroring optimum-intel's
// _OVYoutuVLForCausalLM runtime wrapper (see modeling_visual_language.py).
class VisionEncoderYoutuVL : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderYoutuVL : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderYoutuVL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderYoutuVL(
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

    // Youtu-VL text backbone uses plain 1-D position ids (base implementation), so
    // get_position_ids / get_generation_phase_position_ids are NOT overridden.

protected:
    // The chat template hardcodes the vision tag sequence, so NATIVE_TAG is hardcoded too.
    inline static const std::string NATIVE_TAG = "<|vision_start|><|image_pad|><|vision_end|>";

    void encode_vision_placeholder_tokens();

    // Number of merged image tokens for a (grid_h, grid_w) patch grid.
    size_t calc_tokens_num(size_t grid_h, size_t grid_w) const;

    std::map<std::string, int64_t> m_vision_token_ids;
    size_t m_merge_length = 4;  // spatial_merge_size ** 2
};

} // namespace ov::genai
