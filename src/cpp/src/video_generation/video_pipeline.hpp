// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <memory>

#include "generation_config_utils.hpp"
#include "logger.hpp"
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

    virtual size_t get_audio_sample_rate() const {
        return 0;
    }

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

    virtual void replace_defaults(VideoGenerationConfig& generation_config) const = 0;

    virtual size_t get_transformer_expected_batch_size() const = 0;

    virtual void reshape_models(const VideoGenerationConfig& generation_config, size_t batch_size_multiplier) = 0;

    VideoGenerationConfig merge_generation_config(const ov::AnyMap& properties) const {
        VideoGenerationConfig merged_generation_config = m_generation_config;
        utils::update_generation_config(merged_generation_config, properties);
        replace_defaults(merged_generation_config);
        return merged_generation_config;
    }

    // Resolves the effective CFG batch-size multiplier against the reshape/compile state,
    // reshaping not-yet-compiled models when a larger multiplier is required
    size_t resolve_batch_size_multiplier(const VideoGenerationConfig& generation_config, bool cfg_requested) {
        size_t requested_batch_size_multiplier = cfg_requested ? 2 : 1;
        if (m_is_compiled) {
            const size_t expected_batch_size = get_transformer_expected_batch_size();
            if (expected_batch_size > 0) {
                OPENVINO_ASSERT(expected_batch_size % generation_config.num_videos_per_prompt == 0,
                                "Compiled batch size must be divisible by num_videos_per_prompt");
                requested_batch_size_multiplier = expected_batch_size / generation_config.num_videos_per_prompt;
            } else if (m_compiled_batch_size_multiplier > 0) {
                requested_batch_size_multiplier = m_compiled_batch_size_multiplier;
            }
            OPENVINO_ASSERT(!(requested_batch_size_multiplier > 1 && !cfg_requested),
                            "guidance_scale <= 1 requested, but the compiled model expects CFG (batch size multiplier = ",
                            requested_batch_size_multiplier, "). "
                            "Either set guidance_scale > 1, or reshape/compile the model with guidance_scale <= 1.");
        }
        const size_t batch_size_multiplier = std::max({requested_batch_size_multiplier,
                                                       m_reshape_batch_size_multiplier,
                                                       m_compiled_batch_size_multiplier});

        if (!m_is_compiled) {
            if (m_reshape_batch_size_multiplier == 0) {
                m_reshape_batch_size_multiplier = batch_size_multiplier;
            } else if (m_reshape_batch_size_multiplier < batch_size_multiplier) {
                reshape_models(generation_config, batch_size_multiplier);
            }
        }

        if (m_is_compiled && cfg_requested && batch_size_multiplier <= 1) {
            GENAI_WARN("guidance_scale > 1 requested, but the compiled model batch size does not allow CFG. "
                       "Run reshape/compile with guidance_scale > 1 to enable guidance.");
        }
        return batch_size_multiplier;
    }

    VideoGenerationConfig m_generation_config;
    VideoGenerationPerfMetrics m_perf_metrics;
    Ms m_load_time{};
    // Batch size multiplier from the last reshape() call (0 = not set, 1 = no CFG, 2 = CFG enabled)
    size_t m_reshape_batch_size_multiplier = 0;
    // Batch size multiplier used when model was compiled (0 = not compiled, 1 = no CFG, 2 = CFG enabled)
    size_t m_compiled_batch_size_multiplier = 0;
    bool m_is_compiled = false;
};

}  // namespace ov::genai
