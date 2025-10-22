// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

/**
 * @brief Configuration for video processing parameters
 */
struct OPENVINO_GENAI_EXPORTS VideoProcessingConfig {
    // Input video parameters
    std::string input_file;
    
    // VPP (Video Post-Processing) parameters
    int target_width = 0;   // 0 means keep original
    int target_height = 0;  // 0 means keep original
    bool denoise = false;
    bool detail_enhance = false;
    
    // Output parameters
    std::string output_file;
    int output_format = 0; // 0 = NV12, 1 = RGB
};

/**
 * @brief Frame data structure for video frames
 */
struct OPENVINO_GENAI_EXPORTS VideoFrame {
    std::vector<uint8_t> data;
    int width;
    int height;
    int format; // 0 = NV12, 1 = RGB
    int64_t timestamp;
};

/**
 * @brief FFmpeg + oneVPL Video Processing Pipeline
 * 
 * This pipeline uses FFmpeg to decode video and oneVPL (Video Processing Library)
 * to perform video post-processing operations like scaling, denoising, and format conversion.
 */
class OPENVINO_GENAI_EXPORTS FFmpegVPLPipeline {
public:
    /**
     * @brief Construct a new FFmpeg VPL Pipeline
     * 
     * @param config Video processing configuration
     */
    explicit FFmpegVPLPipeline(const VideoProcessingConfig& config);
    
    /**
     * @brief Destroy the FFmpeg VPL Pipeline
     */
    ~FFmpegVPLPipeline();
    
    /**
     * @brief Process the video file
     * 
     * Decodes the input video using FFmpeg and processes it using oneVPL VPP,
     * then saves the result to the output file.
     * 
     * @return true if successful, false otherwise
     */
    bool process();
    
    /**
     * @brief Get the next processed frame
     * 
     * @param frame Output frame data
     * @return true if frame was retrieved, false if no more frames
     */
    bool get_next_frame(VideoFrame& frame);
    
    /**
     * @brief Get video metadata
     * 
     * @return Video metadata (width, height, fps, duration, etc.)
     */
    std::string get_metadata() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace genai
} // namespace ov
