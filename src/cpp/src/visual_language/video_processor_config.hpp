// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/processor_config.hpp"
#include <fstream>
#include "json_utils.hpp"

namespace ov::genai {
/**
 * @brief A Configuration class passed to VisionEncoder and 
 * used to change VisionEncoder's behavior for video processing.
 * Corresponds to video_preprocessor_config.json
 */
class VideoProcessorConfig : public ProcessorConfig {
public:
    bool do_sample_frames = false;
    size_t max_frames = 0;
    size_t min_frames = 0;
    size_t num_frames = 0;
    // Target sampling rate in frames per second if do_sample_frames is true. Mutually exclusive with num_frames.
    // Used to compute the number of frames to extract from the original video.
    float fps = 0.0f;

    VideoProcessorConfig() = default;

    explicit VideoProcessorConfig(const std::filesystem::path& json_path)
        : ProcessorConfig(json_path)
    {
        std::ifstream stream(json_path);
        OPENVINO_ASSERT(stream.is_open(), "Failed to open '", json_path, "' with video processor config");
        nlohmann::json parsed = nlohmann::json::parse(stream);
        using ov::genai::utils::read_json_param;
        read_json_param(parsed, "do_sample_frames", do_sample_frames);
        read_json_param(parsed, "max_frames", max_frames);
        read_json_param(parsed, "min_frames", min_frames);
        read_json_param(parsed, "num_frames", num_frames);
        read_json_param(parsed, "fps", fps);
    }
};
}  // namespace ov::genai
