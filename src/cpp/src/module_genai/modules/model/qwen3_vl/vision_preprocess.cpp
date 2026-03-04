// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/modules/model/qwen3_vl/vision_preprocess.hpp"

#include <utility>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "openvino/core/except.hpp"

namespace ov::genai::module {

Qwen3VisionPreprocess::Qwen3VisionPreprocess(const std::filesystem::path& model_path, VLMModelType model_type)
    : VisionPreprocess(model_type) {
    (void)model_path;
}

PreprocessOutput Qwen3VisionPreprocess::preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) {
    OPENVINO_ASSERT(images.empty() || videos.empty(), "Qwen3VisionPreprocess: images and videos cannot both be non-empty");
    OPENVINO_THROW("Qwen3VisionPreprocess::preprocess is not implemented yet");
    return {};
}

namespace qwen3vl_utils {
/**
 * @brief Smartly resizes dimensions based on pixel constraints and alignment factors.
 * * @param num_frames Number of frames in the video sequence.
 * @param height Original height.
 * @param width Original width.
 * @param temporal_factor Alignment factor for time/frames (default 2).
 * @param factor Spatial alignment factor for height/width (default 32).
 * @param min_pixels Minimum allowed total pixels (num_frames * h * w).
 * @param max_pixels Maximum allowed total pixels (num_frames * h * w).
 * @return ResizeResult containing the new h_bar and w_bar.
 */
ImageSize smart_resize(int num_frames,
                       int height,
                       int width,
                       int temporal_factor,
                       int factor,
                       size_t min_pixels,
                       size_t max_pixels) {
    // 1. Validation Checks
    if (height < factor || width < factor) {
        throw std::invalid_argument("Height or width must be larger than the alignment factor.");
    }

    double aspect_ratio = static_cast<double>(std::max(height, width)) / std::min(height, width);
    if (aspect_ratio > 200.0) {
        throw std::invalid_argument("Absolute aspect ratio must be smaller than 200.");
    }

    // 2. Initial alignment
    // h_bar and w_bar are rounded to the nearest multiple of 'factor'
    int h_bar = static_cast<int>(std::round(static_cast<double>(height) / factor)) * factor;
    int w_bar = static_cast<int>(std::round(static_cast<double>(width) / factor)) * factor;

    // t_bar is the temporal dimension aligned to 'temporal_factor'
    int t_bar = static_cast<int>(std::ceil(static_cast<double>(num_frames) / temporal_factor)) * temporal_factor;

    // Use size_t for volume calculation to prevent 32-bit integer overflow
    size_t current_pixels = static_cast<size_t>(t_bar) * h_bar * w_bar;

    // 3. Scale adjustment based on pixel budget
    if (current_pixels > max_pixels) {
        // Calculate downscaling ratio (beta)
        double beta = std::sqrt(static_cast<double>(num_frames) * height * width / max_pixels);
        // Floor to ensure we stay under the max_pixels limit after factor alignment
        h_bar = std::max(factor, static_cast<int>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<int>(std::floor(width / beta / factor)) * factor);
    } else if (current_pixels < min_pixels) {
        // Calculate upscaling ratio (beta)
        double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(num_frames) * height * width));
        // Ceil to ensure we meet the min_pixels requirement
        h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int>(std::ceil(width * beta / factor)) * factor;
    }

    return {h_bar, w_bar};
}

ov::Tensor video_padding(const ov::Tensor& video, const size_t& pad) {
    OPENVINO_ASSERT(video.get_shape().size() == 4, "Input video tensor must have 4 dimensions [T, C, H, W]");
    OPENVINO_ASSERT(video.get_element_type() == ov::element::f32,
                    "Input video tensor must be of type f32 after resizing");

    auto resized_shape = video.get_shape();

    auto T = resized_shape[0];
    auto resized_channels = resized_shape[1];
    auto resized_h = resized_shape[2];
    auto resized_w = resized_shape[3];

    ov::Tensor padded_video(ov::element::f32, {T + pad, resized_channels, resized_h, resized_w});
    auto* dst = padded_video.data<float>();
    const auto* src = video.data<const float>();

    // Copy original video frames
    for (size_t t = 0; t < T; ++t) {
        std::copy(src + t * resized_channels * resized_h * resized_w,
                  src + (t + 1) * resized_channels * resized_h * resized_w,
                  dst + t * resized_channels * resized_h * resized_w);
    }

    // Copy padded frames (duplicate the last frame)
    dst = dst + (T * resized_channels * resized_h * resized_w);  // point to the start of padding region
    const auto* last_frame = src + (T - 1) * resized_channels * resized_h * resized_w;
    for (size_t t = 0; t < pad; ++t) {
        std::copy(last_frame,
                  last_frame + resized_channels * resized_h * resized_w,
                  dst + t * resized_channels * resized_h * resized_w);
    }

    return padded_video;
}

ov::Tensor ovtensor_view(ov::Tensor& input, const ov::Shape& target_shape) {
    OPENVINO_ASSERT(input.get_size() == ov::shape_size(target_shape), "Target shape size must match input tensor size for view operation.");
    input.set_shape(target_shape);
    return input;
}

ov::Tensor ovtensor_reshape(const ov::Tensor& input, const ov::Shape& target_shape, bool is_contiguous) {
    OPENVINO_ASSERT(input.get_size() == ov::shape_size(target_shape), "Target shape size must match input tensor size for reshape operation.");
    if (is_contiguous) {
        // If the tensor is contiguous, we can simply return a view with the new shape
        return ovtensor_view(const_cast<ov::Tensor&>(input), target_shape);
    } else {
        OPENVINO_THROW("Reshape with non-contiguous memory layout is not supported in this implementation.");
    }
}

ov::Tensor ovtensor_permute(const ov::Tensor& input, const std::vector<size_t>& order) {
    const auto& input_shape = input.get_shape();
    size_t rank = input_shape.size();
    OPENVINO_ASSERT(order.size() == rank, "Permutation order size must match input tensor rank.");

    // 1. Calculate new shape
    ov::Shape permuted_shape(rank);
    for (size_t i = 0; i < rank; ++i) {
        permuted_shape[i] = input_shape[order[i]];
    }

    // 2. Initialize output tensor
    // Fixed: get_element_type is a function call ()
    ov::Tensor permuted_tensor(input.get_element_type(), permuted_shape);

    const uint8_t* src_data = static_cast<const uint8_t*>(input.data());
    uint8_t* dst_data = static_cast<uint8_t*>(permuted_tensor.data());

    // 3. Handle data types by getting element size in bytes
    size_t elem_size = input.get_byte_size() / input.get_size();

    // 4. Compute strides
    std::vector<size_t> input_strides(rank);
    std::vector<size_t> output_strides(rank);

    auto compute_strides = [](const ov::Shape& shape, std::vector<size_t>& strides) {
        size_t s = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shape[i];
        }
    };

    compute_strides(input_shape, input_strides);
    compute_strides(permuted_shape, output_strides);

    // 5. Permute logic using byte-level copying
    size_t total_elements = permuted_tensor.get_size();

    for (size_t i = 0; i < total_elements; ++i) {
        size_t src_index = 0;
        size_t remaining = i;

        for (size_t j = 0; j < rank; ++j) {
            size_t coord = remaining / output_strides[j];
            remaining %= output_strides[j];
            src_index += coord * input_strides[order[j]];
        }

        // Copy the element byte-by-byte (handles FP32, INT64, etc.)
        std::memcpy(dst_data + i * elem_size, src_data + src_index * elem_size, elem_size);
    }

    return permuted_tensor;
}
}


}  // namespace ov::genai::module
