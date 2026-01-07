// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_img_preprocess.hpp"

#include <chrono>
#include <thread>

namespace ov {
namespace genai {
namespace module {

void ImagePreprocessModule::print_static_config() {
    std::cout << R"(
  image_preprocessor:           # Module Name
    type: "ImagePreprocessModule"
    device: "CPU"               # Optional, default to CPU
    description: "Image or Video preprocessing."
    inputs:
      - name: "image"           # [optional]
        type: "OVTensor"        # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "images"          # [Optional] multiple images
        type: "VecOVTensor"     # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "video"           # [optional] video frames
        type: "OVTensor"        # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "videos"          # [Optional] multiple videos
        type: "VecOVTensor"     # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "raw_data"        # Output port name
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "source_size"     # Output port name
        type: "VecInt"          # Support DataType: [VecInt]
      - name: "raw_datas"       # batch processed vision output
        type: "VecOVTensor"     # Support DataType: [VecOVTensor]
      - name: "source_sizes"    # Output port name
        type: "VecVecInt"       # Support DataType: [VecVecInt]
    params:
      target_resolution: [224, 224]   # optional
      mean: [0.485, 0.456, 0.406]     # optional
      std: [0.229, 0.224, 0.225]      # optional
      model_path: "models/openvino_vision_embeddings_model.xml"
    )" << std::endl;
}

ImagePreprocessModule::ImagePreprocessModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    std::string model_path = desc->get_full_path(desc->params["model_path"]);
    std::string device = desc->device;
    if (device.empty()) {
        device = "CPU";
    }

    VLMModelType model_type = to_vlm_model_type(desc->model_type);

    if (model_type == VLMModelType::QWEN2_VL || model_type == VLMModelType::QWEN2_5_VL) {
        encoder_ptr = std::make_shared<VisionEncoderQwen2VL>(std::filesystem::path(model_path), device, ov::AnyMap{});
    } else {
        GENAI_ERR("ImagePreprocessModule[" + desc->name + "]: Unsupported model type: " + desc->model_type);
    }
}

ImagePreprocessModule::~ImagePreprocessModule() {}

void ImagePreprocessModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    auto image1_data = this->inputs["image"].data.as<ov::Tensor>();
    auto encoded_img = encoder_ptr->encode(image1_data, ov::AnyMap{});

    this->outputs["raw_data"].data = encoded_img.resized_source;
    this->outputs["source_size"].data = std::vector<int>{encoded_img.resized_source_size.height, encoded_img.resized_source_size.width};
}

}  // namespace module
}  // namespace genai
}  // namespace ov
