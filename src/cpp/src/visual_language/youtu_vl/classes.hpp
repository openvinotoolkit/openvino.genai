// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

// Youtu-VL couples a Siglip2 windowed vision encoder (Qwen2.5-VL-style windowing) plus a
// VLPatchMerger with a DeepSeek/MiniCPM3-style MLA text decoder. The vision IR is a single merged
// submodel: it consumes flattened Siglip2 patches plus host-computed bookkeeping
// (rotary_pos_emb, window_index, full/window attention masks) and outputs merged image tokens
// directly. The language model IR is generic (inputs_embeds/attention_mask/position_ids/beam_idx ->
// logits); the MLA internals live entirely inside the graph. Position ids are plain 1-D (no mRoPE),
// so the base IInputsEmbedder position-id logic is reused. This mirrors optimum-intel
// _OVYoutuVLForCausalLM.
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

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {}) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override;

protected:
    // The chat template hardcodes the vision tag sequence, so it is hardcoded here as well.
    inline static const std::string NATIVE_TAG = "<|vision_start|><|image_pad|><|vision_end|>";
};

} // namespace ov::genai
