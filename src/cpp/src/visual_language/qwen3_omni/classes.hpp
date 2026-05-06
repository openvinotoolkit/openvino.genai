// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/qwen3_omni/audio_encoder.hpp"
#include "visual_language/qwen3_vl/classes.hpp"

namespace ov::genai {

/// @brief Vision encoder for Qwen3-Omni.
/// Does NOT load the vision_embeddings model (it's merged with the merger in the new export format).
/// Only does image preprocessing (resize, normalize, create raw patches).
/// The merged vision model is loaded and used by InputsEmbedderQwen3Omni.
class VisionEncoderQwen3Omni : public VisionEncoderQwen3VL {
public:
    explicit VisionEncoderQwen3Omni(const std::filesystem::path& model_dir,
                                    const std::string& device,
                                    const ov::AnyMap properties);
    explicit VisionEncoderQwen3Omni(const ModelsMap& models_map,
                                    const std::filesystem::path& config_dir_path,
                                    const std::string& device,
                                    const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames, const ov::AnyMap& config_map) override;

private:
    /// @brief Preprocess a set of frames into raw flattened patches without running inference.
    /// Writes flattened patches [total_patches, patch_dim] to out_tensor and grid dims to out_rsz_size.
    void preprocess_to_patches(const std::vector<ov::Tensor>& images,
                               const ProcessorConfig& config,
                               ov::Tensor& out_tensor,
                               ImageSize& out_rsz_size,
                               size_t frame_num,
                               size_t frame_id);
};

/// @brief InputsEmbedder for Qwen3-Omni. Extends Qwen3-VL with audio encoding support.
class InputsEmbedderQwen3Omni : public InputsEmbedderQwen3VL {
public:
    InputsEmbedderQwen3Omni(const VLMConfig& vlm_config,
                            const std::filesystem::path& model_dir,
                            const std::string& device,
                            const ov::AnyMap device_config);

    InputsEmbedderQwen3Omni(const VLMConfig& vlm_config,
                            const ModelsMap& models_map,
                            const Tokenizer& tokenizer,
                            const std::filesystem::path& config_dir_path,
                            const std::string& device,
                            const ov::AnyMap device_config);

    /// @brief Override to merge audio embeddings into the input embeds
    /// alongside image/video embeddings.
    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        const std::vector<ov::genai::EncodedVideo>& videos,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {},
        const std::vector<size_t>& videos_sequence = {},
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

    /// @brief Encode raw audio tensors and cache the embeddings.
    /// Must be called before get_inputs_embeds() if audio is present.
    void encode_audios(const std::vector<ov::Tensor>& audios) override;

    /// @brief Check if audio encoder model was loaded and is ready for inference.
    bool has_audio_encoder() const {
        return m_audio_encoder != nullptr && m_audio_encoder->is_available();
    }

    /// @brief Override to inject audio placeholder tokens into the prompt
    /// when audio embeddings are available.
    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_id,
                                      const std::vector<EncodedImage>& images) const override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t image_base_id,
                                      size_t video_base_id,
                                      const std::vector<EncodedImage>& images,
                                      const std::vector<EncodedVideo>& videos) const override;

    void start_chat(const std::string& system_message) override;
    void finish_chat() override;

protected:
    /// @brief Check if the merged vision model was loaded and is ready for inference.
    /// The merged vision model is optional — it is only needed when images or videos are provided.
    bool has_merged_vision_model() const {
        return m_ireq_queue_merged_vision != nullptr;
    }

    /// @brief Override to use the merged vision model (patch_embed + transformer + merger in one).
    std::pair<ov::Tensor, ov::Tensor> run_video_image_embeddings_merger(
        const std::vector<EncodedImage>& images,
        const std::vector<size_t>& images_sequence,
        const std::vector<EncodedVideo>& videos,
        const std::vector<size_t>& videos_sequence) override;

    /// @brief Override to use merged vision model for rotary dim instead of separate merger.
    ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) const override;

private:
    std::unique_ptr<AudioEncoderQwen3Omni> m_audio_encoder;
    // Cached audio embeddings from last encode_audios() call
    ov::Tensor m_audio_embeddings;
    // Audio token ID used to identify audio placeholder positions
    int64_t m_audio_token_id = -1;

    // Merged vision model (patch_embed + transformer + merger)
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_merged_vision;
    // Cached rotary embedding dimension from merged vision model (avoids queue lock in get_rotary_pos_emb)
    size_t m_rotary_dim = 0;

    /// @brief Replace audio token positions in input_embeds with audio features.
    void merge_audio_embeddings(ov::Tensor& input_embeds, const std::vector<int64_t>& input_ids);
};

}  // namespace ov::genai
