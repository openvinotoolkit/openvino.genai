// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "module_genai/module.hpp"

namespace ov {
namespace genai {
namespace module {

/**
 * @brief SaveVideoModule - Saves video tensor to AVI file using MJPEG codec.
 *
 * This module accepts a 5D video tensor in [B, F, H, W, C] format:
 *   - B: Batch size (number of videos)
 *   - F: Frames (number of frames per video)
 *   - H: Height
 *   - W: Width
 *   - C: Channels (1=grayscale, 3=RGB, 4=RGBA)
 *
 * Supported DataTypes: uint8
 * Output format: AVI with MJPEG compression
 */
class SaveVideoModule : public IBaseModule {
    DeclareModuleConstructor(SaveVideoModule);

private:
    bool initialize();

    std::string m_filename_prefix;
    std::string m_output_folder;
    uint32_t m_fps;
    int m_quality;
    bool m_convert_bgr2rgb;
    std::atomic<size_t> m_sequence_number{0};

    std::string generate_filename();

    /**
     * @brief Save a video tensor to AVI file(s)
     * @param tensor Video tensor in [B, F, H, W, C] format
     * @param filepath Base filepath for output
     * @return Vector of saved file paths
     */
    std::vector<std::string> save_tensor_as_video(const ov::Tensor& tensor, const std::string& filepath);
};

REGISTER_MODULE_CONFIG(SaveVideoModule);

} // namespace module
} // namespace genai
} // namespace ov
