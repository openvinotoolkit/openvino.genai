// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/video_generation/generation_config.hpp"
#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"
#include "openvino/genai/image_generation/scheduler.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "openvino/genai/video_generation/autoencoder_kl_ltx_video.hpp"

namespace ov::genai {
using VideoGenerationPerfMetrics = ImageGenerationPerfMetrics;

struct VideoGenerationResult {
    ov::Tensor video;
    ov::genai::VideoGenerationPerfMetrics perfomance_stat;
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
    VideoGenerationResult generate(
        const std::string& positive_prompt,
        const std::string& negative_prompt = "",
        const ov::AnyMap& properties = {}
    );

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<VideoGenerationResult, Properties...> generate(
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
}  // namespace ov::genai
