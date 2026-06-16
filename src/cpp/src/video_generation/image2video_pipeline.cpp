// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/image2video_pipeline.hpp"
#include "video_generation/ltx_pipeline.hpp"

using namespace ov::genai;

Image2VideoPipeline::Image2VideoPipeline(const std::filesystem::path& model_path) {
    OPENVINO_ASSERT(std::filesystem::exists(model_path / "vae_encoder"),
                    "Image2VideoPipeline requires a 'vae_encoder' directory in ",
                    model_path,
                    ". For text-to-video generation without image conditioning, use Text2VideoPipeline.");
    m_impl = std::make_unique<LTXPipeline>(model_path, VideoPipelineType::IMAGE_2_VIDEO);
}

Image2VideoPipeline::Image2VideoPipeline(const std::filesystem::path& models_path,
                                          const std::string& device,
                                          const ov::AnyMap& properties) {
    OPENVINO_ASSERT(std::filesystem::exists(models_path / "vae_encoder"),
                    "Image2VideoPipeline requires a 'vae_encoder' directory in ",
                    models_path,
                    ". For text-to-video generation without image conditioning, use Text2VideoPipeline.");
    m_impl = std::make_unique<LTXPipeline>(models_path, device, properties, VideoPipelineType::IMAGE_2_VIDEO);
}

Image2VideoPipeline::Image2VideoPipeline(std::unique_ptr<LTXPipeline> impl) : m_impl(std::move(impl)) {}

const VideoGenerationConfig& Image2VideoPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

void Image2VideoPipeline::set_generation_config(const VideoGenerationConfig& generation_config) {
    utils::validate_generation_config(generation_config);
    m_impl->m_generation_config = generation_config;
    replace_defaults(m_impl->m_generation_config);
}

void Image2VideoPipeline::reshape(int64_t num_videos_per_prompt,
                                   int64_t num_frames,
                                   int64_t height,
                                   int64_t width,
                                   float guidance_scale) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->reshape(num_videos_per_prompt, num_frames, height, width, guidance_scale);
    m_impl->save_load_time(start_time);

    auto config = m_impl->m_generation_config;
    config.num_videos_per_prompt = num_videos_per_prompt;
    config.num_frames = num_frames;
    config.height = height;
    config.width = width;
    config.guidance_scale = guidance_scale;
    set_generation_config(config);
}

void Image2VideoPipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(device, properties);
    m_impl->save_load_time(start_time);
}

void Image2VideoPipeline::compile(const std::string& text_encode_device,
                                   const std::string& denoise_device,
                                   const std::string& vae_device,
                                   const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(text_encode_device, denoise_device, vae_device, properties);
    m_impl->save_load_time(start_time);
}

VideoGenerationResult Image2VideoPipeline::generate(const ov::Tensor& image,
                                                     const std::string& positive_prompt,
                                                     const ov::AnyMap& properties) {
    return m_impl->generate(image, positive_prompt, properties);
}

VideoGenerationResult Image2VideoPipeline::decode(const ov::Tensor& latent) {
    return m_impl->decode(latent);
}

VideoGenerationPerfMetrics Image2VideoPipeline::get_performance_metrics() {
    return m_impl->get_performance_metrics();
}

Image2VideoPipeline Image2VideoPipeline::clone() {
    return Image2VideoPipeline(m_impl->clone());
}

Image2VideoPipeline::~Image2VideoPipeline() = default;
