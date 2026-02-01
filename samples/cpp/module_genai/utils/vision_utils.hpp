// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file utils_image.hpp
 * @brief Image utility functions for loading and saving ov::Tensor images
 */

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>
#include <filesystem>

#include <openvino/runtime/tensor.hpp>

namespace image_utils {

// ============================================================================
// Image Loading Functions
// ============================================================================

/**
 * @brief Load a single image from file
 * @param image_path Path to the image file
 * @return ov::Tensor with shape [1, H, W, 3] in RGB format
 */
ov::Tensor load_image(const std::filesystem::path& image_path);

/**
 * @brief Load multiple images from a directory or a single image file
 * @param input_path Path to a directory or single image file
 * @return Vector of ov::Tensor images
 */
std::vector<ov::Tensor> load_images(const std::filesystem::path& input_path);

/**
 * @brief Load images as a video tensor
 * @param input_path Path to directory containing video frames
 * @return ov::Tensor with shape [num_frames, H, W, 3]
 */
ov::Tensor load_video(const std::filesystem::path& input_path);

/**
 * @brief Create countdown frames for testing
 * @return ov::Tensor with shape [5, 240, 360, 3] containing countdown frames
 */
ov::Tensor create_countdown_frames();

// ============================================================================
// Image Saving Functions
// ============================================================================

/**
 * @brief Save ov::Tensor image to BMP file
 * @param filename Output file path
 * @param image Input tensor (HWC or NHWC format with 3 channels)
 * @param convert_rgb2bgr Whether to convert RGB to BGR (default: true)
 * @return true if successful, false otherwise
 */
bool save_image_bmp(const std::string& filename, const ov::Tensor& image, bool convert_rgb2bgr = true);

/**
 * @brief Generate unique filename with timestamp
 * @param prefix Filename prefix
 * @param suffix Filename suffix (default: ".bmp")
 * @return Generated filename with timestamp
 */
std::string generate_output_filename(const std::string& prefix, const std::string& suffix = ".bmp");

} // namespace image_utils
