// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"
#include "openvino/genai/image_generation/scheduler.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "openvino/genai/video_generation/autoencoder_kl_ltx_video.hpp"
#include "utils.hpp"

namespace ov::genai {
using VideoGenerationPerfMetrics = ImageGenerationPerfMetrics;

/**
 * Generation config used for Text2VideoPipeline.
 * Note, that not all values are applicable for all pipelines and models - please, refer
 * to corresponding properties for ImageGenerationConfig to understand a meaning and applicability for specific models.
 */
struct VideoGenerationConfig : public ImageGenerationConfig {
    /// guidance_rescale (`float`, *optional*, defaults to 0.0):
    /// Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
    /// Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
    /// [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
    /// Guidance rescale factor should fix overexposure when using zero terminal SNR.
    /// Mixes with the original results from guidance by factor guidance_rescale to avoid "plain looking" images.
    /// negative or 0.0 disables rescaling. NaN corresponds to model default which is 0.0 for LTX-Video thus disabled.
    double guidance_rescale = std::numeric_limits<double>::quiet_NaN();
    /// The number of video frames to generate. 0 corresponds to model default which is 161 for LTX-Video.
    size_t num_frames = 0;
    /// Video frame rate. Affects rope_interpolation_scale. Any value can be used although positive non-infinity makes the most sense. NaN corresponds to model default which is 25.0f for LTX-Video.
    float frame_rate = std::numeric_limits<float>::quiet_NaN();

    /**
     * Checks whether video generation config is valid, otherwise throws an exception.
     */
    void validate() const;

    /**
     * Updates generation config from a map of properties.
     * @param properties A map of properties
     */
    void update_generation_config(const ov::AnyMap& properties);

    /**
     * Updates generation config from properties. Calls AnyMap version of update_generation_config().
     * @param properties
     */
    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(ov::AnyMap{std::forward<Properties>(properties)...});
    }
};

class OPENVINO_GENAI_EXPORTS Text2VideoPipeline {
public:
    static Text2VideoPipeline ltx_video();
    Text2VideoPipeline(
        const std::filesystem::path& models_dir,
        const std::string& device,
        const AnyMap& properties = {}
    );
    /**
     * Generates image(s) based on prompt and other image generation parameters
     * @param positive_prompt Prompt to generate image(s) from
     * @param negative_prompt
     * @param properties Image generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns A tensor which has dimensions [num_images_per_prompt, height, width, 3]
     */
    ov::Tensor generate(
        const std::string& positive_prompt,
        const std::string& negative_prompt = "",
        const ov::AnyMap& properties = {}
    );

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
        const std::string& positive_prompt,
        const std::string& negative_prompt,
        Properties&&... properties
    ) {
        return generate(positive_prompt, negative_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    const VideoGenerationConfig& get_generation_config() const;
    void set_generation_config(const VideoGenerationConfig& generation_config);

    ~Text2VideoPipeline();

private:
    class LTXPipeline;
    std::unique_ptr<LTXPipeline> m_impl;
};

/// guidance_rescale (`float`, *optional*, defaults to 0.0):
/// Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
/// Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
/// [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
/// Guidance rescale factor should fix overexposure when using zero terminal SNR.
/// Mixes with the original results from guidance by factor guidance_rescale to avoid "plain looking" images.
/// 0.0 disables rescaling.
static constexpr ov::Property<double> guidance_rescale{"guidance_rescale"};
/// The number of video frames to generate.
static constexpr ov::Property<size_t> num_frames{"num_frames"};
/// Video frame rate.
static constexpr ov::Property<float> frame_rate{"guidance_rescale"};
}  // namespace ov::genai
