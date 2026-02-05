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
 * This module accepts video tensors in two formats:
 *   1. [B, F, H, W, C] format (channels-last, e.g., from VideoProcessor)
 *   2. [B, C, F, H, W] format (channels-first, e.g., from VAE decoder)
 *
 * Format detection:
 *   - If shape[1] <= 4 and shape[4] > 4: channels-first [B, C, F, H, W]
 *   - Otherwise: channels-last [B, F, H, W, C]
 *
 * For channels-first format with float data (typical VAE output), the module
 * automatically converts from [-1, 1] or [0, 1] range to uint8 [0, 255].
 *
 * Supported DataTypes: uint8, float32, float16
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
