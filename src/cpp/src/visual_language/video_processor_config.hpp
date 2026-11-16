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
    // Target sampling rate in frames per second if do_sample_frames is true.
    // Generic video processors treat fps and the exact num_frames as mutually exclusive;
    // Model-specific maximum frame counts are stored in max_frames.
    float fps = 0.0f;

    // Muse Glimmer-specific config params.
    size_t max_video_frame_tokens = 144;

    // Molmo2-specific config params (Molmo2VideoProcessor).
    /// @brief Frame sampling mode, e.g. "uniform_last_frame" or "fps".
    std::string frame_sample_mode = "uniform_last_frame";
    /// @brief Highest frame rate sampled from a video; not positive: no limit.
    float max_fps = 2.0f;
    /// @brief Frame rate of a video whose frame rate is unknown.
    float sampling_fps = 2.0f;

    VideoProcessorConfig() = default;

    explicit VideoProcessorConfig(const nlohmann::json& parsed_json)
        : ProcessorConfig(parsed_json)
    {
        using ov::genai::utils::read_json_param;
        read_json_param(parsed_json, "do_sample_frames", do_sample_frames);
        read_json_param(parsed_json, "max_frames", max_frames);
        read_json_param(parsed_json, "min_frames", min_frames);
        read_json_param(parsed_json, "num_frames", num_frames);
        read_json_param(parsed_json, "fps", fps);
        read_json_param(parsed_json, "max_video_frame_tokens", max_video_frame_tokens);
        read_json_param(parsed_json, "frame_sample_mode", frame_sample_mode);
        read_json_param(parsed_json, "max_fps", max_fps);
        read_json_param(parsed_json, "sampling_fps", sampling_fps);
    }

    explicit VideoProcessorConfig(const std::filesystem::path& json_path)
        : VideoProcessorConfig([&json_path] {
            std::ifstream stream(json_path);
            OPENVINO_ASSERT(stream.is_open(), "Failed to open '", json_path, "' with video processor config");
            return nlohmann::json::parse(stream);
        }()) {}
};
}  // namespace ov::genai
