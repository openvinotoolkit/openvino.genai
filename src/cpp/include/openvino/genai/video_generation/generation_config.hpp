// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/image_generation/generation_config.hpp"

namespace ov::genai {
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
    /// Video frame rate. Affects rope_interpolation_scale. Any value can be used although positive
    /// non-infinity makes the most sense. NaN corresponds to model default which is 25.0f for LTX-Video.
    float frame_rate = std::numeric_limits<float>::quiet_NaN();

    size_t num_videos_per_prompt = 1;

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
static constexpr ov::Property<float> frame_rate{"frame_rate"};
/// The number of videos to generate.
static constexpr ov::Property<size_t> num_videos_per_prompt{"num_videos_per_prompt"};
}  // namespace ov::genai
