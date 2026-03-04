// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <vector>

#include "module_genai/module_base.hpp"
#include "module_genai/utils/video_utils.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai::module {

using OutputModule = IBaseModule::OutputModule;

struct PreprocessOutput {
    ov::Tensor pixel_values;
    ov::Tensor grid_thw;
    ov::Tensor pos_embeds;
    ov::Tensor rotary_cos;
    ov::Tensor rotary_sin;

    // Video-specific outputs:
    ov::Tensor pixel_values_videos;
    ov::Tensor video_grid_thw;
};

// Vision preprocessing facade.
//
// The public API is intentionally model-agnostic so we can add more backends
// later. Currently, the factory returns a Qwen3_5VisionPreprocess instance
// (and QWEN3_VL is not yet implemented and returns nullptr).
class VisionPreprocess {
public:
    using PTR = std::shared_ptr<VisionPreprocess>;

    static PTR create(const std::filesystem::path& model_path, VLMModelType model_type = VLMModelType::QWEN3_VL);

    VisionPreprocess(const VisionPreprocess&) = delete;
    VisionPreprocess& operator=(const VisionPreprocess&) = delete;

    virtual ~VisionPreprocess() = default;

    // Preprocess images and videos.
    virtual PreprocessOutput preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) = 0;

private:
    VisionPreprocess() = delete;

protected:
    explicit VisionPreprocess(VLMModelType model_type) : _model_type(model_type) {}
    VLMModelType _model_type;
};

}  // namespace ov::genai::module
