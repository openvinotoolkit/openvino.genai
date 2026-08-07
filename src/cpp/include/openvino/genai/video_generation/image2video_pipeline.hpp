// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/genai/video_generation/text2video_pipeline.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS Image2VideoPipeline {
public:
    /**
     * Initializes image to video generation pipeline from a folder with models.
     * Requires a 'vae_encoder' subdirectory in addition to the standard model files.
     * @param models_path A models path to read models and config files from
     */
    explicit Image2VideoPipeline(const std::filesystem::path& models_path);

    /**
     * Initializes image to video pipeline from a folder with models and performs compilation after it.
     * @param models_path A models path to read models and config files from
     * @param device A single device used for all models
     * @param properties Properties to pass to 'compile_model' or other pipeline properties like LoRA adapters
     */
    Image2VideoPipeline(const std::filesystem::path& models_path,
                        const std::string& device,
                        const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Image2VideoPipeline(const std::filesystem::path& models_path,
                        const std::string& device,
                        Properties&&... properties)
        : Image2VideoPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * Method to clone the pipeline to be used in parallel by another thread.
     * Reuses underlying models and recreates scheduler and generation config.
     * @returns A new pipeline for concurrent usage
     */
    Image2VideoPipeline clone();

    /**
     * Returns default video generation config.
     */
    const VideoGenerationConfig& get_generation_config() const;

    /**
     * Sets video generation config.
     */
    void set_generation_config(const VideoGenerationConfig& generation_config);

    /**
     * Reshapes pipeline based on a given set of parameters.
     */
    void reshape(int64_t num_videos_per_prompt,
                 int64_t num_frames,
                 int64_t height,
                 int64_t width,
                 float guidance_scale);

    /**
     * Compiles image to video pipeline for a given device.
     */
    void compile(const std::string& device, const ov::AnyMap& properties = {});

    /**
     * Compiles image to video pipeline for given devices for text encoding, denoising, and vae encoding/decoding.
     */
    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> compile(const std::string& device,
                                                                Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> compile(const std::string& text_encode_device,
                                                                const std::string& denoise_device,
                                                                const std::string& vae_device,
                                                                Properties&&... properties) {
        return compile(text_encode_device,
                       denoise_device,
                       vae_device,
                       ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Generates video(s) conditioned on an input image.
     * Frame 0 of the output is anchored to the encoded conditioning image.
     * Requires a 'vae_encoder' in the model directory.
     * @param image Conditioning image as a uint8 tensor of shape [H, W, 3] or [1, H, W, 3] in NHWC layout.
     *              Resized internally to (height, width) if not already that size.
     * @param positive_prompt Prompt to guide video generation
     * @param properties Generation parameters (see VideoGenerationConfig).
     * @returns VideoGenerationResult with video tensor shaped as [num_videos_per_prompt, num_frames, height, width, 3]
     */
    VideoGenerationResult generate(const ov::Tensor& image,
                                   const std::string& positive_prompt,
                                   const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<VideoGenerationResult, Properties...> generate(
        const ov::Tensor& image,
        const std::string& positive_prompt,
        Properties&&... properties) {
        return generate(image, positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Performs latent video decoding.
     */
    VideoGenerationResult decode(const ov::Tensor& latent);

    VideoGenerationPerfMetrics get_performance_metrics();

    ~Image2VideoPipeline();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
    explicit Image2VideoPipeline(std::unique_ptr<Impl> impl);
    friend class Text2VideoPipeline;
};

}  // namespace ov::genai
