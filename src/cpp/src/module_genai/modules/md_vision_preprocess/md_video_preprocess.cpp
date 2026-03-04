// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_video_preprocess.hpp"

#include "module_genai/module_factory.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include "module_genai/modules/model/qwen3_5/qwen3_5preprocessor.hpp"

#include <chrono>
#include <thread>

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(VideoPreprocessModule);

void VideoPreprocessModule::print_static_config() {
    std::cout << R"(
  video_preprocessor:           # Module Name
    type: "VideoPreprocessModule"
    device: "CPU"               # Optional, default to CPU
    description: "Video preprocessing."
    inputs:
      - name: "video"           # [optional] video frames
        type: "OVTensor"        # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "videos"          # [Optional] multiple videos
        type: "VecOVTensor"     # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "raw_data"        # Output port name, used by Qwen 2.5-VL
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "source_size"     # Output port name, used by Qwen 2.5-VL
        type: "VecInt"          # Support DataType: [VecInt]
      - name: "raw_datas"       # batch processed vision output, used by Qwen 2.5-VL
        type: "VecOVTensor"     # Support DataType: [VecOVTensor]
      - name: "source_sizes"    # Output port name, used by Qwen 2.5-VL
        type: "VecVecInt"       # Support DataType: [VecVecInt]
      - name: "pos_embeds"      # Output port name, used by Qwen 3.5
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "rotary_cos"      # Output port name, used by Qwen 3.5
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "rotary_sin"      # Output port name, used by Qwen 3.5
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "video_grid_thw"    # Output port name, used by Qwen 3.5 for video input
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "pixel_values_videos"    # Output port name, used by Qwen 3.5 for video input
        type: "OVTensor"        # Support DataType: [OVTensor]
    params:
      model_path: "models/openvino_vision_embeddings_model.xml"
    )" << std::endl;
}

VideoPreprocessModule::VideoPreprocessModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    std::string model_path = desc->get_full_path(desc->params["model_path"]);
    std::string device = desc->device;
    if (device.empty()) {
        device = "CPU";
    }

    _model_type = to_vlm_model_type(desc->model_type);

    _vision_preprocess_ptr = VisionPreprocess::create(model_path, _model_type);
    if (_vision_preprocess_ptr == nullptr) {
        _encoder_ptr = VisionEncoder::create(model_path, _model_type, device);
        OPENVINO_ASSERT(_encoder_ptr != nullptr,
                        "Failed to create VisionEncoder for VideoPreprocessModule: " + desc->name);
    }
}

VideoPreprocessModule::~VideoPreprocessModule() {}

void VideoPreprocessModule::run_video(const bool& has_videos_input) {
    std::vector<ov::Tensor> videos_data;
    if (has_videos_input) {
        videos_data = get_input("videos").as<std::vector<ov::Tensor>>();
    } else {
        videos_data.push_back(get_input("video").as<ov::Tensor>());
    }

    if (_vision_preprocess_ptr) {
        OPENVINO_ASSERT(!videos_data.empty(), "VideoPreprocessModule: empty video input");

        // Current preprocess backends only support a single video tensor.
        OPENVINO_ASSERT(videos_data.size() == 1u,
                        "VideoPreprocessModule: only a single video input is supported");

        auto output = _vision_preprocess_ptr->preprocess({}, videos_data);
        this->outputs["pixel_values_videos"].data = output.pixel_values_videos;
        this->outputs["video_grid_thw"].data = output.video_grid_thw;
        this->outputs["pos_embeds"].data = output.pos_embeds;
        this->outputs["rotary_cos"].data = output.rotary_cos;
        this->outputs["rotary_sin"].data = output.rotary_sin;
        return;
    }

    // Fallback to VisionEncoder path (legacy)
    OPENVINO_ASSERT(_encoder_ptr != nullptr, "VideoPreprocessModule: VisionEncoder is not initialized");
    auto encoded_video = _encoder_ptr->encode_frames(videos_data, ov::AnyMap{});

    // Keep output port names consistent with ImagePreprocessModule.
    if (has_videos_input) {
        this->outputs["raw_datas"].data = std::vector<ov::Tensor>{encoded_video.video_features};
        this->outputs["source_sizes"].data = std::vector<std::vector<int>>{
            {static_cast<int>(encoded_video.resized_source_size.height),
             static_cast<int>(encoded_video.resized_source_size.width)}};
    } else {
        this->outputs["raw_data"].data = encoded_video.video_features;
        this->outputs["source_size"].data = std::vector<int>{static_cast<int>(encoded_video.resized_source_size.height),
                                                             static_cast<int>(encoded_video.resized_source_size.width)};
    }
}

void VideoPreprocessModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    bool has_video_input = exists_input("video");
    bool has_videos_input = exists_input("videos");
    bool has_video = has_video_input || has_videos_input;
    if (has_video) {
        OPENVINO_ASSERT(
            !(has_video_input && has_videos_input),
            "VideoPreprocessModule: Both 'video' and 'videos' inputs exist. Please provide only one of them.");
    }

    if (has_video_input || has_videos_input) {
      run_video(has_videos_input);
    } else {
        OPENVINO_THROW("VideoPreprocessModule[" + module_desc->name +
                       "]: No valid input found. Please provide one of the following inputs: 'video', 'videos'.");
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov
