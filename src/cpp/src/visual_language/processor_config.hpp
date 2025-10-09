// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include <openvino/runtime/properties.hpp>
#include <array>
#include <filesystem>

namespace ov::genai {
/// @brief A Configuration class passed to VisionEncoder and used to
/// change VisionEncoder's behavior. Corresponds to
/// preprocessor_config.json.
class ProcessorConfig {
public:
    size_t image_size = 980;
    /// @brief Dimensions of the smaller, non-overlapping patches that the
    /// input image is divided into before being fed into the
    /// transformer model. Used to divide image height and width.
    size_t patch_size = 14;
    /// @brief A recommended size to resize an input image.
    /// llava calls it crop_size[height, width].
    size_t scale_resolution = 448;
    /// @brief Maximum allowed number of input image slices.
    /// 0 disables slicing.
    /// llava has image_grid_pinpoints instead.
    size_t max_slice_nums = 0;
    /// @brief RGB values to be subtracted from image pixel values.
    /// Applied before norm_std.
    /// llava calls it image_mean.
    std::array<float, 3> norm_mean{0.0f, 0.0f, 0.0f};
    /// @brief RGB values to divide image pixel values.
    /// Applied after norm_mean.
    /// llava calls it image_std.
    std::array<float, 3> norm_std{1.0f, 1.0f, 1.0f};

    // A renamed version of norm_mean.
    std::array<float, 3> image_mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> image_std{1.0f, 1.0f, 1.0f};
    // llava specific config params
    size_t crop_size_height = 336;
    size_t crop_size_width = 336;
    size_t size_shortest_edge = 336;

    // llava-next specific config params
    std::vector<std::pair<int, int>> image_grid_pinpoints{{336, 672}, {672, 336}, {672, 672}, {1008, 336}, {336, 1008}};

    // gemma3-4b-it specific config params
    size_t size_height = 896;
    size_t size_width = 896;

    struct {
        size_t num_crops = 4;
        size_t num_img_tokens = 144;
    } phi3_v;
    // qwen2vl specific params
    size_t min_pixels = 3136;
    size_t max_pixels = 12845056;
    size_t temporal_patch_size = 2;
    size_t merge_size = 2;

    /// @brief Default constructor
    ProcessorConfig() = default;
    /// @brief Construct ProcessorConfig from values in json_path.
    /// Keys in the file must match the ProcessorConfig's members.
    /// @param json_path A path to a file to extract the values from.
    explicit ProcessorConfig(const std::filesystem::path& json_path);
};
}  // namespace ov::genai
