// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/text2video_pipeline.hpp"

namespace {
std::unique_ptr<ov::genai::LTXPipeline> create_LTXPipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const ov::AnyMap& properties
) {
    // TODO: move to common
    const std::string class_name = get_class_name(models_dir);
    auto start_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT("LTXPipeline" == class_name);
    std::unique_ptr<ov::genai::LTXPipeline> impl = std::make_unique<ov::genai::LTXPipeline>(models_dir, device, properties);
    impl->save_load_time(start_time);
    return impl;
}
}  // anonymous namespace

namespace ov::genai {
Text2VideoPipeline::Text2VideoPipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const AnyMap& properties
) : m_impl{create_LTXPipeline(models_dir, device, properties)} {}
}  // namespace ov::genai
