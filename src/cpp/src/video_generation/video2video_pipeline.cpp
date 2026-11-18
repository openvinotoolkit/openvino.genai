// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/video2video_pipeline.hpp"
#include "json_utils.hpp"
#include "video_generation/ltx2_pipeline.hpp"

using namespace ov::genai;

namespace {

void check_video2video_model(const std::filesystem::path& models_path) {
    const std::string class_name = utils::get_class_name(models_path);
    OPENVINO_ASSERT(class_name == "LTX2Pipeline",
                    "Unsupported video to video generation pipeline '", class_name, "'. Expected an LTX-2 model.");
    OPENVINO_ASSERT(std::filesystem::exists(models_path / "vae_encoder"),
                    "Video2VideoPipeline requires a 'vae_encoder' directory in ", models_path, ".");
}

}  // namespace

Video2VideoPipeline::Video2VideoPipeline(const std::filesystem::path& models_path) {
    check_video2video_model(models_path);
    m_impl = std::make_shared<LTX2Pipeline>(VideoPipelineType::VIDEO_2_VIDEO, models_path);
}

Video2VideoPipeline::Video2VideoPipeline(const std::filesystem::path& models_path,
                                         const std::string& device,
                                         const ov::AnyMap& properties) {
    check_video2video_model(models_path);
    m_impl = std::make_shared<LTX2Pipeline>(VideoPipelineType::VIDEO_2_VIDEO, models_path, device, properties);
}

Video2VideoPipeline::Video2VideoPipeline(std::shared_ptr<VideoPipeline> impl) : m_impl(std::move(impl)) {}

const VideoGenerationConfig& Video2VideoPipeline::get_generation_config() const {
    return m_impl->get_generation_config();
}

void Video2VideoPipeline::set_generation_config(const VideoGenerationConfig& generation_config) {
    m_impl->set_generation_config(generation_config);
}

void Video2VideoPipeline::reshape(int64_t num_videos_per_prompt,
                                  int64_t num_frames,
                                  int64_t height,
                                  int64_t width,
                                  float guidance_scale) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->reshape(num_videos_per_prompt, num_frames, height, width, guidance_scale);
    m_impl->save_load_time(start_time);

    auto config = m_impl->get_generation_config();
    config.num_videos_per_prompt = num_videos_per_prompt;
    config.num_frames = num_frames;
    config.height = height;
    config.width = width;
    config.guidance_scale = guidance_scale;
    set_generation_config(config);
}

void Video2VideoPipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(device, properties);
    m_impl->save_load_time(start_time);
}

void Video2VideoPipeline::compile(const std::string& text_encode_device,
                                  const std::string& denoise_device,
                                  const std::string& vae_device,
                                  const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(text_encode_device, denoise_device, vae_device, properties);
    m_impl->save_load_time(start_time);
}

VideoGenerationResult Video2VideoPipeline::generate(const ov::Tensor& video,
                                                    const std::string& positive_prompt,
                                                    const ov::AnyMap& properties) {
    std::vector<VideoCondition> conditions;
    const auto conditions_iter = properties.find(ov::genai::conditions.name());
    if (conditions_iter != properties.end() && !conditions_iter->second.empty()) {
        conditions = conditions_iter->second.as<std::vector<VideoCondition>>();
    }
    return m_impl->generate(&video, conditions, positive_prompt, properties);
}

VideoGenerationResult Video2VideoPipeline::decode(const ov::Tensor& latent) {
    return m_impl->decode(latent);
}

VideoGenerationPerfMetrics Video2VideoPipeline::get_performance_metrics() {
    return m_impl->get_performance_metrics();
}

Video2VideoPipeline Video2VideoPipeline::clone() {
    return Video2VideoPipeline(m_impl->clone());
}

Video2VideoPipeline::~Video2VideoPipeline() = default;
