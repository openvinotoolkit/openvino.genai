// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ov::genai {

/**
 * @brief Configuration struct for saving OpenVINO models generated from GGUF files.
 * 
 * This configuration controls whether and how to save the OpenVINO model after
 * loading from a GGUF file. Two model variants can be saved:
 * - Original GGUF quantization (small group size, for accuracy alignment with llama.cpp)
 * - Requantized model (larger group size, optimized for GPU performance)
 */
struct SaveOVModelConfig {
    enum class SaveMode {
        DISABLED,    // Do not save any model (default)
        ORIGINAL,    // Save model preserving original GGUF quantization
        OPTIMIZED    // Save model with optimized quantization for target device
    };

    /**
     * Which model variant to save.
     * - DISABLED: Do not save any model (default)
     * - ORIGINAL: Save model with original GGUF quantization (preserves Q4_K/Q6_K small group sizes)
     *             Output: <gguf_dir>/ov_model_original/openvino_model.xml
     * - OPTIMIZED: Save model with device-optimized quantization (e.g., Q4_0_128/Q8_0_C for GPU)
     *              Output: <gguf_dir>/ov_model_optimized/openvino_model.xml
     * 
     * Future extensions can add more modes like COMPRESSED, PERFORMANCE, etc.
     */
    SaveMode mode = SaveMode::DISABLED;

    bool operator==(const SaveOVModelConfig& other) const {
        return mode == other.mode;
    }
};

}  // namespace ov::genai
