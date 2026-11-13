// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/video2video_pipeline.hpp"

#include <cstring>

#include "json_utils.hpp"
#include "video_generation/flashvsr_pipeline.hpp"

using namespace ov::genai;

namespace {

std::shared_ptr<VideoToVideoPipelineImpl> create_impl(const std::filesystem::path& models_path) {
    const std::string class_name = utils::get_class_name(models_path);
    OPENVINO_ASSERT(class_name == "FlashVSRPipeline",
                    "Unsupported video to video pipeline '",
                    class_name,
                    "'. Expected a FlashVSR model exported with genai-cli.");
    return std::make_shared<FlashVSRPipeline>(models_path);
}

}  // namespace

VideoToVideoPipeline::VideoToVideoPipeline(const std::filesystem::path& models_path) {
    const auto start_time = std::chrono::steady_clock::now();
    m_impl = create_impl(models_path);
    m_load_time_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_time).count();
}

VideoToVideoPipeline::VideoToVideoPipeline(const std::filesystem::path& models_path,
                                           const std::string& device,
                                           const ov::AnyMap& properties) {
    const auto start_time = std::chrono::steady_clock::now();
    m_impl = create_impl(models_path);
    m_impl->compile(device, properties);
    m_load_time_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_time).count();
}

const VideoGenerationConfig& VideoToVideoPipeline::get_generation_config() const {
    return m_impl->get_generation_config();
}

void VideoToVideoPipeline::set_generation_config(const VideoGenerationConfig& generation_config) {
    m_impl->set_generation_config(generation_config);
}

void VideoToVideoPipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    const auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(device, properties);
    m_load_time_ms += std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_time).count();
}

VideoGenerationResult VideoToVideoPipeline::generate(const ov::Tensor& video, const ov::AnyMap& properties) {
    const auto& shape = video.get_shape();
    OPENVINO_ASSERT(shape.size() == 4 && shape[3] == 3,
                    "VideoToVideoPipeline expects a video [num_frames, height, width, 3], got ",
                    shape);
    m_impl->start_stream(properties, shape[0]);
    const ov::Tensor head = m_impl->push_frames(video);
    const ov::Tensor tail = m_impl->end_stream();

    ov::Shape out_shape = tail.get_shape();
    const size_t frames = head.get_shape()[0] + out_shape[0];
    out_shape.insert(out_shape.begin(), 1);
    out_shape[1] = frames;
    VideoGenerationResult result;
    result.video = ov::Tensor(ov::element::u8, out_shape);
    auto* dst = static_cast<uint8_t*>(result.video.data());
    std::memcpy(dst, head.data(), head.get_byte_size());
    std::memcpy(dst + head.get_byte_size(), tail.data(), tail.get_byte_size());
    result.performance_stat = get_performance_metrics();
    return result;
}

void VideoToVideoPipeline::start_stream(const ov::AnyMap& properties) {
    m_impl->start_stream(properties, 0);
}

ov::Tensor VideoToVideoPipeline::push_frames(const ov::Tensor& frames) {
    return m_impl->push_frames(frames);
}

ov::Tensor VideoToVideoPipeline::end_stream() {
    return m_impl->end_stream();
}

VideoGenerationPerfMetrics VideoToVideoPipeline::get_performance_metrics() {
    auto metrics = m_impl->get_performance_metrics();
    metrics.load_time = m_load_time_ms;
    return metrics;
}

VideoToVideoPipeline::~VideoToVideoPipeline() = default;
