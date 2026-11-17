// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <string>

#include "generation_config_utils.hpp"
#include "openvino/genai/video_generation/generation_config.hpp"
#include "openvino/genai/video_generation/text2video_pipeline.hpp"

namespace ov::genai {

/// Model specific implementation behind VideoToVideoPipeline.
class VideoToVideoPipelineImpl {
public:
    virtual ~VideoToVideoPipelineImpl() = default;

    virtual void compile(const std::string& device, const ov::AnyMap& properties) = 0;

    /// `total_frames` is the number of frames of the whole video when known up front (generate()), 0 otherwise.
    virtual void start_stream(const ov::AnyMap& properties, size_t total_frames) = 0;
    virtual ov::Tensor push_frames(const ov::Tensor& frames) = 0;
    virtual ov::Tensor end_stream() = 0;

    const VideoGenerationConfig& get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const VideoGenerationConfig& generation_config) {
        m_generation_config = generation_config;
        replace_defaults(m_generation_config);
    }

    VideoGenerationPerfMetrics get_performance_metrics() const {
        return m_perf_metrics;
    }

protected:
    using Ms = std::chrono::duration<float, std::ratio<1, 1000>>;

    virtual void replace_defaults(VideoGenerationConfig& generation_config) const = 0;

    VideoGenerationConfig merge_generation_config(const ov::AnyMap& properties) const {
        VideoGenerationConfig merged = m_generation_config;
        utils::update_generation_config(merged, properties);
        replace_defaults(merged);
        return merged;
    }

    VideoGenerationConfig m_generation_config;
    VideoGenerationPerfMetrics m_perf_metrics;
};

}  // namespace ov::genai
