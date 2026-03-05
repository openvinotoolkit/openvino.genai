// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "video_generation/ltx_pipeline.hpp"

using namespace ov::genai;

Text2VideoPipeline::Text2VideoPipeline(const std::filesystem::path& model_path)
    : m_impl{std::make_unique<ov::genai::Text2VideoPipeline::LTXPipeline>(model_path)} {}

Text2VideoPipeline::Text2VideoPipeline(const std::filesystem::path& models_dir,
                                       const std::string& device,
                                       const AnyMap& properties)
    : m_impl{std::make_unique<ov::genai::Text2VideoPipeline::LTXPipeline>(models_dir, device, properties)} {}

VideoGenerationResult Text2VideoPipeline::generate(const std::string& positive_prompt, const ov::AnyMap& properties) {
    return m_impl->generate(positive_prompt, properties);
}

const VideoGenerationConfig& Text2VideoPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

void Text2VideoPipeline::set_generation_config(const VideoGenerationConfig& generation_config) {
    utils::validate_generation_config(generation_config);
    m_impl->m_generation_config = generation_config;
    replace_defaults(m_impl->m_generation_config);
}

void Text2VideoPipeline::reshape(int64_t num_videos_per_prompt,
                                 int64_t num_frames,
                                 int64_t height,
                                 int64_t width,
                                 float guidance_scale) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->reshape(num_videos_per_prompt, num_frames, height, width, guidance_scale);
    m_impl->save_load_time(start_time);

    // update config with the specified parameters, so that the user doesn't need to explicitly pass these as properties
    // to generate()
    auto config = m_impl->m_generation_config;
    config.num_videos_per_prompt = num_videos_per_prompt;
    config.num_frames = num_frames;
    config.height = height;
    config.width = width;
    config.guidance_scale = guidance_scale;

    set_generation_config(config);
}

void Text2VideoPipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(device, properties);
    m_impl->save_load_time(start_time);
}

void Text2VideoPipeline::compile(const std::string& text_encode_device,
                                 const std::string& denoise_device,
                                 const std::string& vae_device,
                                 const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(text_encode_device, denoise_device, vae_device, properties);
    m_impl->save_load_time(start_time);
}

VideoGenerationResult Text2VideoPipeline::decode(const ov::Tensor& latent) {
    return m_impl->decode(latent);
}

Text2VideoPipeline::~Text2VideoPipeline() = default;
