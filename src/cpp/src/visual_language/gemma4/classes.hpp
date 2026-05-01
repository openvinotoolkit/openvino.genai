// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

class VisionEncoderGemma4 : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderGemma4 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderGemma4(const VLMConfig& vlm_config,
                         const std::filesystem::path& model_dir,
                         const std::string& device,
                         const ov::AnyMap device_config);

    InputsEmbedderGemma4(const VLMConfig& vlm_config,
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

    std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {}) override;

    bool has_token_type_ids() const override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_id,
                                      const std::vector<EncodedImage>& images) const override;

    const std::unordered_map<std::string, ov::Tensor>& get_lm_extra_inputs() const override;

    std::function<ov::Tensor(const ov::Tensor& new_input_ids)> get_per_layer_embeddings_callback() override {
        if (!has_per_layer_embeddings()) {
            // MOE LM models still have a `per_layer_inputs` Parameter input with hidden_size_per_layer == 0,
            // so a zero-element tensor with rank 4 must be supplied to satisfy the input port.
            return [this](const ov::Tensor& input_ids) {
                return make_empty_per_layer_inputs(input_ids);
            };
        }
        return [this](const ov::Tensor& input_ids) {
            return get_per_layer_embeddings(input_ids);
        };
    }

private:
    // Per-layer text embeddings model (Gemma4-specific)
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_per_layer_embeddings_requests = nullptr;

    // Extra inputs to pass to the language model
    std::unordered_map<std::string, ov::Tensor> m_lm_extra_inputs;

    ov::Tensor get_per_layer_embeddings(const ov::Tensor& input_ids);

    bool has_per_layer_embeddings() const {
        return m_vlm_config.hidden_size_per_layer_input > 0;
    }

    // Persistent non-null backing buffer for empty per_layer_inputs tensors used by MOE variants.
    std::array<float, 1> m_empty_per_layer_inputs_storage{};

    /// @brief Build a zero-element `per_layer_inputs` tensor matching the LM model's expected rank.
    /// Used for Gemma4 MOE variants whose LM model exposes the input with last dim == 0.
    /// A non-null backing buffer is provided to avoid potential null pointer dereferences in
    /// downstream consumers when the tensor has zero elements.
    ov::Tensor make_empty_per_layer_inputs(const ov::Tensor& input_ids) {
        const auto& shape = input_ids.get_shape();
        const size_t batch_size = shape.size() > 0 ? shape[0] : 1;
        const size_t seq_len = shape.size() > 1 ? shape[1] : 0;
        return ov::Tensor(ov::element::f32, {batch_size, seq_len, 1, 0}, m_empty_per_layer_inputs_storage.data());
    }

    /// @brief Compute merged text+image embeddings together with the encoded input_ids.
    /// Shared implementation behind get_inputs_embeds() and get_inputs_embeds_with_token_type_ids().
    /// @return A pair of (inputs_embeds, input_ids).
    std::pair<ov::Tensor, ov::Tensor> compute_inputs_embeds(const std::string& prompt,
                                                            const std::vector<EncodedImage>& images,
                                                            VLMPerfMetrics& metrics,
                                                            const std::vector<size_t>& images_sequence);

    ov::Tensor get_token_type_ids(const ov::Tensor& input_ids);

    int64_t m_image_token_id = -1;
    std::once_flag m_image_token_id_once_flag;

    void encode_image_token_id();
};

}  // namespace ov::genai
