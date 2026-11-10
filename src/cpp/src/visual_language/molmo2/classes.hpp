// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Vision encoder for the Molmo2 (allenai/MolmoWeb-4B) architecture.
///
/// Molmo2 preprocessing is data-dependent: the input image is decomposed into a low-resolution
/// (global) crop plus a set of overlapping high-resolution crops, each split into 14x14 patches.
/// A pooling-index tensor describes which patches are averaged into a single image token. Both the
/// patchified crops ("images") and the pooling indices ("pooled_patches_idx") are consumed by the
/// exported vision backbone, which returns one embedding row per image token.
class VisionEncoderMolmo2 : public VisionEncoder {
public:
    /// @brief Molmo2ImageProcessor / Molmo2VideoProcessor options the generic ProcessorConfig does not hold.
    struct PreprocessingOptions {
        /// @brief Maximum number of overlapping high-resolution crops per image (image processor only).
        size_t max_crops = 8;
        /// @brief Overlap margins (in patches) removed on each side when tiling crops (image processor only).
        size_t overlap_margin_left = 4;
        size_t overlap_margin_right = 4;
        /// @brief Vision-adapter pooling window: the patches pooled into one image token.
        size_t pooling_h = 2;
        size_t pooling_w = 2;
    };

    VisionEncoderMolmo2(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties);

    VisionEncoderMolmo2(const ModelsMap& models_map,
                        const std::filesystem::path& config_dir_path,
                        const std::string& device,
                        const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

    /// @brief Encodes already sampled video frames: every frame is resized to a single base-size crop and pooled
    /// with the video pooling window (video_preprocessor_config.json), mirroring Molmo2VideoProcessor.
    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames) override;

private:
    /// @brief Reads the options from the image and video processor configs VisionEncoder resolves.
    void read_preprocessing_options(const std::filesystem::path& config_dir);

    PreprocessingOptions m_image_options;
    PreprocessingOptions m_video_options;
};

/// @brief Inputs embedder for the Molmo2 architecture.
///
/// Molmo2 expands the "<|image|>" placeholder into a run of image special tokens whose layout
/// depends on the image grid (a low-resolution section followed by a high-resolution section).
/// Vision embeddings are *added* to the learned "<im_patch>" placeholder embeddings (Molmo2 uses an
/// additive merge rather than a scatter-replace). Image placeholder tokens receive token_type_ids == 1
/// so the language model can apply bidirectional attention between them at prefill.
class InputsEmbedderMolmo2 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderMolmo2(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderMolmo2(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<ov::genai::EncodedImage>& images,
                                 const std::vector<ov::genai::EncodedVideo>& videos,
                                 ov::genai::VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {},
                                 const std::vector<size_t>& videos_sequence = {},
                                 const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

    /// @brief token_type_ids of the last embedded sequence: image and frame special tokens are marked with 1, so the
    /// language model attends between them bidirectionally at prefill.
    const std::unordered_map<std::string, ov::Tensor>& get_lm_extra_inputs() const override;

    std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos,
                                                       const std::vector<VideoMetadata>& videos_metadata = {}) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_image_id,
                                      size_t base_video_id,
                                      const std::vector<EncodedImage>& images,
                                      const std::vector<EncodedVideo>& videos) const override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const override;

protected:
    /// @brief Finishes the Molmo2 prompt layout once the prompt formatting is known (normalize_prompt runs before
    /// it is set), tokenizes it and prepends the BOS token before get_encoded_input_ids registers the sequence. In a
    /// text rendered by the chat template (a chat history or a prompt that needs the template), the visual tokens are
    /// moved in front of the conversation the way the checkpoint chat template places the media; raw prompts keep
    /// them in place. The pipelines run Molmo2 chats with the full history on every turn, so the whole text is embedded.
    ov::Tensor apply_chat_template_tokenize(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) override;

private:
    /// @brief Build the run of image special tokens for a single image grid, mirroring
    /// Molmo2Processor.get_image_tokens (low-resolution section followed by high-resolution section).
    std::string build_image_tokens(size_t resized_h, size_t resized_w, size_t high_h, size_t high_w) const;

    /// @brief Build the frame tokens of a video, mirroring Molmo2Processor.get_video_string: for every frame its
    /// timestamp ("%.1f") followed by the frame start token, the pooled patch tokens and the frame end token.
    std::string build_video_tokens(const EncodedVideo& video) const;

    /// @brief Read the Molmo2 processor (processor_config.json) and video sampling
    /// (video_preprocessor_config.json) options.
    void read_molmo2_configs(const std::filesystem::path& config_dir);

    /// @brief Whether a token is an image or frame special token (token_type_ids == 1).
    bool is_image_token(int64_t id) const;

    /// @brief Molmo2Processor options from processor_config.json and the video sampling options.
    bool m_image_use_col_tokens = true;
    int m_use_single_crop_col_tokens = -1;  // -1: follow m_image_use_col_tokens
    bool m_use_single_crop_start_token = false;
    bool m_video_use_col_tokens = false;
    bool m_use_frame_special_tokens = true;
    std::string m_frame_sample_mode = "uniform_last_frame";
    size_t m_num_frames = 4;
    float m_max_fps = 2.0f;
    float m_sampling_fps = 2.0f;

    std::unordered_map<std::string, ov::Tensor> m_lm_extra_inputs;

    /// @brief Types of the visual segments (images / videos) of the last tokenized text, in token order: the layout
    /// apply_chat_template_tokenize produced for get_inputs_embeds of the same request (raw prompts can interleave
    /// images and videos).
    std::vector<char> m_segment_types;
};

} // namespace ov::genai
