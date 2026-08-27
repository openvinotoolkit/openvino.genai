// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <memory>

#include "generation_config_utils.hpp"
#include "openvino/genai/video_generation/generation_config.hpp"
#include "openvino/genai/video_generation/text2video_pipeline.hpp"

namespace ov::genai {

enum class VideoPipelineType {
    TEXT_2_VIDEO = 0,
    IMAGE_2_VIDEO = 1,
};

class VideoPipeline {
public:
    virtual VideoGenerationResult generate(const std::string& positive_prompt, const ov::AnyMap& properties) = 0;

    virtual VideoGenerationResult generate(const ov::Tensor& image,
                                           const std::string& positive_prompt,
                                           const ov::AnyMap& properties) {
        OPENVINO_THROW("Image-to-video generation is not supported by this pipeline");
    }

    virtual VideoGenerationResult decode(const ov::Tensor& latent) = 0;

    virtual void reshape(int64_t num_videos_per_prompt,
                         int64_t num_frames,
                         int64_t height,
                         int64_t width,
                         float guidance_scale) = 0;

    virtual void compile(const std::string& text_encode_device,
                         const std::string& denoise_device,
                         const std::string& vae_device,
                         const ov::AnyMap& properties) = 0;

    virtual void compile(const std::string& device, const ov::AnyMap& properties) {
        compile(device, device, device, properties);
    }

    virtual std::shared_ptr<VideoPipeline> clone() = 0;

    const VideoGenerationConfig& get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const VideoGenerationConfig& generation_config) {
        utils::validate_generation_config(generation_config);
        m_generation_config = generation_config;
        replace_defaults(m_generation_config);
    }

    VideoGenerationPerfMetrics get_performance_metrics() const {
        return m_perf_metrics;
    }

    void save_load_time(std::chrono::steady_clock::time_point start_time) {
        m_load_time += Ms{std::chrono::steady_clock::now() - start_time};
    }

    virtual ~VideoPipeline() = default;

protected:
    using Ms = std::chrono::duration<float, std::ratio<1, 1000>>;

    // Replace unset sentinel values with per-model defaults.
    virtual void replace_defaults(VideoGenerationConfig& generation_config) const = 0;

    VideoGenerationConfig m_generation_config;
    VideoGenerationPerfMetrics m_perf_metrics;
    Ms m_load_time{};
};

}  // namespace ov::genai
