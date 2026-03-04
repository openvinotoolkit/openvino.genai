// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/modules/model/qwen3_5/vision_preprocess.hpp"

#include <utility>

#include "openvino/core/except.hpp"
#include "module_genai/utils/tensor_utils.hpp"

namespace ov::genai::module {

Qwen3_5VisionPreprocess::Qwen3_5VisionPreprocess(const std::filesystem::path& model_path, VLMModelType model_type)
    : VisionPreprocess(model_type) {
    //   m_video_processor(std::make_unique<Qwen3_5VLVideoProcessor>(model_path)) {}
    m_preprocessor = std::make_shared<Qwen3_5Preprocessor>(model_path);
}

PreprocessOutput Qwen3_5VisionPreprocess::preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) {
    OPENVINO_ASSERT(images.empty() || videos.empty(), "Qwen3_5VisionPreprocess: images and videos cannot both be non-empty");
    OPENVINO_ASSERT(videos.size() == 1u || videos.empty(), "Qwen3_5VisionPreprocess: only a single video input is supported due to the complexity of handling variable-length videos and batching them together");

    ov::Tensor stack_images;
    PreprocessOutput preprocess_output;
    if (images.size() > 0) {
        if (images.size() > 1) {
            stack_images = tensor_utils::stack(images, 0);
        } else if (images.size() == 1) {
            stack_images = images[0];
        }

        auto output = m_preprocessor->preprocess(stack_images);
        preprocess_output.pixel_values = std::move(output.pixel_values);
        preprocess_output.grid_thw = std::move(output.grid_thw);
        preprocess_output.pos_embeds = std::move(output.pos_embeds);
        preprocess_output.rotary_cos = std::move(output.rotary_cos);
        preprocess_output.rotary_sin = std::move(output.rotary_sin);
    }
    else if (videos.size() == 1) {
        auto output = m_preprocessor->preprocess_video(videos[0]);
        preprocess_output.pixel_values_videos = std::move(output.pixel_values_videos);
        preprocess_output.video_grid_thw = std::move(output.video_grid_thw);
        preprocess_output.pos_embeds = std::move(output.pos_embeds);
        preprocess_output.rotary_cos = std::move(output.rotary_cos);
        preprocess_output.rotary_sin = std::move(output.rotary_sin);
    } else {
        OPENVINO_THROW("No valid input provided to Qwen3_5VisionPreprocess::preprocess. Please provide either images or a video.");
    }

    return preprocess_output;
}

// void Qwen3_5VisionPreprocess::result_to_output(std::map<std::string, OutputModule>& output) const {
//     output["pixel_values"].data = m_output.pixel_values;
//     output["grid_thw"].data = m_output.grid_thw;
//     output["pos_embeds"].data = m_output.pos_embeds;
//     output["rotary_cos"].data = m_output.rotary_cos;
//     output["rotary_sin"].data = m_output.rotary_sin;
// }

}  // namespace ov::genai::module
