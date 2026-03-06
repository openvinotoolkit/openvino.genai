// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "module_genai/modules/md_vision_preprocess/vision_preprocess.hpp"
#include "module_genai/utils/vision_preprocess.hpp"

namespace ov::genai::module {

class Qwen3VisionPreprocess final : public VisionPreprocess {
public:
    Qwen3VisionPreprocess() = delete;
    Qwen3VisionPreprocess(const std::filesystem::path& model_path, VLMModelType model_type);

    PreprocessOutput preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) override;

    // void result_to_output(std::map<std::string, OutputModule>& output) const override;

private:
};

namespace qwen3vl_utils {

ImageSize smart_resize(int num_frames,
                       int height,
                       int width,
                       int temporal_factor = 2,
                       int factor = 32,
                       size_t min_pixels = 128 * 128,
                       size_t max_pixels = 16 * 16 * 2 * 2 * 2 * 6144);

// Pads the input video tensor on the temporal dimension by duplicating the last frame
// For example input shape [T, C, H, W] and pad=1, the output video will
// have shape [T+1, C, H, W], where the last frame is a duplicate of the original last frame.
ov::Tensor video_padding(const ov::Tensor& video, const size_t& pad);

ov::Tensor ovtensor_view(ov::Tensor& input, const ov::Shape& target_shape);

ov::Tensor ovtensor_permute(const ov::Tensor& input, const std::vector<size_t>& order);

ov::Tensor ovtensor_reshape(const ov::Tensor& input, const ov::Shape& target_shape, bool is_contiguous = true);

}  // namespace qwen3vl_utils

}  // namespace ov::genai::module
