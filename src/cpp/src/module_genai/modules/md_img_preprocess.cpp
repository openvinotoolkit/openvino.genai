// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_img_preprocess.hpp"

#include "module_genai/module_factory.hpp"
#include "module_genai/utils/tensor_utils.hpp"

#include <chrono>
#include <thread>

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(ImagePreprocessModule);

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
      - name: "raw_data"        # Output port name, used by Qwen 2.5-VL
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "source_size"     # Output port name, used by Qwen 2.5-VL
        type: "VecInt"          # Support DataType: [VecInt]
      - name: "raw_datas"       # batch processed vision output, used by Qwen 2.5-VL
        type: "VecOVTensor"     # Support DataType: [VecOVTensor]
      - name: "source_sizes"    # Output port name, used by Qwen 2.5-VL
        type: "VecVecInt"       # Support DataType: [VecVecInt]
      - name: "pixel_values"    # Output port name, used by Qwen 3.5
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "grid_thw"        # Output port name, used by Qwen 3.5
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "pos_embeds"      # Output port name, used by Qwen 3.5
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "rotary_cos"      # Output port name, used by Qwen 3.5
        type: "OVTensor"        # Support DataType: [OVTensor]
      - name: "rotary_sin"      # Output port name, used by Qwen 3.5
        type: "OVTensor"        # Support DataType: [OVTensor]
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
    } else if (model_type == VLMModelType::QWEN3_5) {
        encoder_ptr = std::make_shared<Qwen3_5Preprocessor>(std::filesystem::path(model_path));
    } else {
        OPENVINO_THROW("ImagePreprocessModule[" + desc->name + "]: Unsupported model type: " + desc->model_type);
    }
}

ImagePreprocessModule::~ImagePreprocessModule() {}

void ImagePreprocessModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    VLMModelType model_type = to_vlm_model_type(module_desc->model_type);

    if (exists_input("images")) {
        auto images_data = get_input("images").as<std::vector<ov::Tensor>>();
        if (model_type == VLMModelType::QWEN2_VL || model_type == VLMModelType::QWEN2_5_VL) {
            std::vector<ov::Tensor> output_tensors;
            std::vector<ImageSize> output_sizes;
            for (size_t i = 0; i < images_data.size(); ++i) {
                auto encoded_img = std::get<std::shared_ptr<VisionEncoderQwen2VL>>(encoder_ptr)->encode(images_data[i], ov::AnyMap{});
                output_tensors.push_back(encoded_img.resized_source);
                output_sizes.push_back(encoded_img.resized_source_size);
            }
            this->outputs["raw_datas"].data = output_tensors;
            std::vector<std::vector<int>> sizes_vec;
            for (const auto& sz : output_sizes) {
                sizes_vec.push_back({static_cast<int>(sz.height), static_cast<int>(sz.width)});
            }
            this->outputs["source_sizes"].data = sizes_vec;
        } else if (model_type == VLMModelType::QWEN3_5) {
            ov::Tensor images = tensor_utils::stack(images_data, 0);
            Qwen3_5PreprocessorOutput output = std::get<std::shared_ptr<Qwen3_5Preprocessor>>(encoder_ptr)->preprocess(images);
            this->outputs["pixel_values"].data = output.pixel_values;
            this->outputs["grid_thw"].data = output.grid_thw;
            this->outputs["pos_embeds"].data = output.pos_embeds;
            this->outputs["rotary_cos"].data = output.rotary_cos;
            this->outputs["rotary_sin"].data = output.rotary_sin;
        }
    } else {
        auto image1_data = get_input("image").as<ov::Tensor>();
        if (model_type == VLMModelType::QWEN2_VL || model_type == VLMModelType::QWEN2_5_VL) {
            auto encoded_img = std::get<std::shared_ptr<VisionEncoderQwen2VL>>(encoder_ptr)->encode(image1_data, ov::AnyMap{});
            this->outputs["raw_data"].data = encoded_img.resized_source;
            this->outputs["source_size"].data =
                std::vector<int>{static_cast<int>(encoded_img.resized_source_size.height), static_cast<int>(encoded_img.resized_source_size.width)};
        } else if (model_type == VLMModelType::QWEN3_5) {
            Qwen3_5PreprocessorOutput output = std::get<std::shared_ptr<Qwen3_5Preprocessor>>(encoder_ptr)->preprocess(image1_data);
            this->outputs["pixel_values"].data = output.pixel_values;
            this->outputs["grid_thw"].data = output.grid_thw;
            this->outputs["pos_embeds"].data = output.pos_embeds;
            this->outputs["rotary_cos"].data = output.rotary_cos;
            this->outputs["rotary_sin"].data = output.rotary_sin;
        }
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov
