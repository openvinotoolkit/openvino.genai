// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "openvino/genai/video_generation/video_condition.hpp"

namespace ov::genai {

/**
 * Video-to-video generation with LTX-2 models (diffusers' LTX2ConditionPipeline). A source video can be edited,
 * conditions can place frames in the output, e.g. to prolong a video, or both:
 *  - Edit: set 'noise_scale' (the noise level the whole source video gets) and / or 'edit_num_steps' (how many of the
 *    last steps of the schedule run). The output takes the length and size of the source unless they are given.
 *  - Conditions: pass 'ov::genai::conditions', each a set of frames with a latent frame index and a strength. To
 *    prolong a video, place its last frames at index 0 and ask for a longer 'num_frames'.
 */
class OPENVINO_GENAI_EXPORTS Video2VideoPipeline {
public:
    /**
     * Initializes video to video generation pipeline from a folder with models.
     * Requires a 'vae_encoder' subdirectory in addition to the standard model files.
     * @param models_path A models path to read models and config files from
     */
    explicit Video2VideoPipeline(const std::filesystem::path& models_path);

    /**
     * Initializes video to video pipeline from a folder with models and performs compilation after it.
     * @param models_path A models path to read models and config files from
     * @param device A single device used for all models
     * @param properties Properties to pass to 'compile_model'
     */
    Video2VideoPipeline(const std::filesystem::path& models_path,
                        const std::string& device,
                        const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Video2VideoPipeline(const std::filesystem::path& models_path,
                        const std::string& device,
                        Properties&&... properties)
        : Video2VideoPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * Method to clone the pipeline to be used in parallel by another thread.
     * @returns A new pipeline for concurrent usage
     */
    Video2VideoPipeline clone();

    const VideoGenerationConfig& get_generation_config() const;

    void set_generation_config(const VideoGenerationConfig& generation_config);

    /**
     * Reshapes pipeline based on a given set of parameters. Conditions placed after the first frame add tokens to
     * the sequence, so a reshaped pipeline takes conditions at latent index 0 only.
     */
    void reshape(int64_t num_videos_per_prompt,
                 int64_t num_frames,
                 int64_t height,
                 int64_t width,
                 float guidance_scale);

    void compile(const std::string& device, const ov::AnyMap& properties = {});

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
     * Generates video(s) from a source video, edited and / or conditioned (see the class description).
     * @param video Source video as a uint8 tensor [F, H, W, 3]. It is the input of an edit.
     * @param positive_prompt Prompt describing the video to generate
     * @param properties Generation parameters (see VideoGenerationConfig) and 'ov::genai::conditions'.
     * @returns VideoGenerationResult with video tensor shaped as [num_videos_per_prompt, num_frames, height, width, 3]
     * and the generated audio
     */
    VideoGenerationResult generate(const ov::Tensor& video,
                                   const std::string& positive_prompt,
                                   const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<VideoGenerationResult, Properties...> generate(
        const ov::Tensor& video,
        const std::string& positive_prompt,
        Properties&&... properties) {
        return generate(video, positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    VideoGenerationResult decode(const ov::Tensor& latent);

    VideoGenerationPerfMetrics get_performance_metrics();

    ~Video2VideoPipeline();

private:
    std::shared_ptr<VideoPipeline> m_impl;
    explicit Video2VideoPipeline(std::shared_ptr<VideoPipeline> impl);
};

}  // namespace ov::genai
