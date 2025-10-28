// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/vlm_config.hpp"
#include "visual_language/llava/classes.hpp"
#include "visual_language/llava_next/classes.hpp"

namespace ov::genai {

class VisionEncoderLLaVANextVideo : public VisionEncoderLLaVANext {
public:
    using VisionEncoderLLaVANext::VisionEncoderLLaVANext;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

    std::pair<std::vector<ov::Tensor>, size_t> preprocess_frames(const std::vector<ov::Tensor>& frames);

    VisionEncoderLLaVANextVideo(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties);

    VisionEncoderLLaVANextVideo(const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    CircularBufferQueueElementGuard<ov::InferRequest> get_multi_modal_projector() {
        return m_ireq_queue_multi_modal_projector.get();
    }

    CircularBufferQueueElementGuard<ov::InferRequest> get_vision_encoder() {
        return m_ireq_queue_vision_encoder.get();
    }

    CircularBufferQueueElementGuard<ov::InferRequest> get_vision_resampler() {
        return m_ireq_queue_vision_resampler.get();
    }

private:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_multi_modal_projector;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_resampler;
    size_t m_patch_size;
};

class InputsEmbedderLLaVANextVideo : public InputsEmbedderLLaVANext {
public:
    InputsEmbedderLLaVANextVideo(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderLLaVANextVideo(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);
        
    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        const std::vector<ov::genai::EncodedVideo>& videos,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings,
        const std::vector<size_t>& images_sequence,
        const std::vector<size_t>& videos_sequence) override;

    std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos) override;

    NormlizedPrompt normalize_prompt(
        const std::string& prompt,
            size_t base_image_id,
            size_t base_video_id,
            const std::vector<EncodedImage>& images,
            const std::vector<EncodedVideo>& videos) const override;


    NormlizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override;
};

} // namespace ov::genai
