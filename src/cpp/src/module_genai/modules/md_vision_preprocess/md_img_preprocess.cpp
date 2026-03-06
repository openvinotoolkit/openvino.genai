// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_img_preprocess.hpp"

#include "module_genai/module_factory.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include "module_genai/modules/models/qwen3_5/qwen3_5preprocessor.hpp"

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
    description: "Image preprocessing."
    inputs:
      - name: "image"           # [optional]
        type: "OVTensor"        # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "images"          # [Optional] multiple images
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

    _model_type = to_vlm_model_type(desc->model_type);

    _vision_preprocess_ptr = VisionPreprocess::create(model_path, _model_type);
    if (_vision_preprocess_ptr == nullptr) {
        _encoder_ptr = VisionEncoder::create(model_path, _model_type, device);
        OPENVINO_ASSERT(_encoder_ptr != nullptr,
                        "Failed to create VisionEncoder for ImagePreprocessModule: " + desc->name);
    }
}

ImagePreprocessModule::~ImagePreprocessModule() {}

void ImagePreprocessModule::run_image(const bool& has_images_input) {
    std::vector<ov::Tensor> images_data;
    if (has_images_input) {
        images_data = get_input("images").as<std::vector<ov::Tensor>>();
    } else {
        images_data.push_back(get_input("image").as<ov::Tensor>());
    }

    if (_vision_preprocess_ptr) {
        auto output = _vision_preprocess_ptr->preprocess(images_data, {});
        this->outputs["pixel_values"].data = output.pixel_values;
        this->outputs["grid_thw"].data = output.grid_thw;
        this->outputs["pos_embeds"].data = output.pos_embeds;
        this->outputs["rotary_cos"].data = output.rotary_cos;
        this->outputs["rotary_sin"].data = output.rotary_sin;
    } else {
        std::vector<ov::Tensor> output_tensors;
        std::vector<ImageSize> output_sizes;
        for (size_t i = 0; i < images_data.size(); ++i) {
            auto encoded_img = _encoder_ptr->encode(images_data[i], ov::AnyMap{});
            output_tensors.push_back(encoded_img.resized_source);
            output_sizes.push_back(encoded_img.resized_source_size);
        }

        if (has_images_input) {
            this->outputs["raw_datas"].data = output_tensors;
            std::vector<std::vector<int>> sizes_vec;
            for (const auto& sz : output_sizes) {
                sizes_vec.push_back({static_cast<int>(sz.height), static_cast<int>(sz.width)});
            }
            this->outputs["source_sizes"].data = sizes_vec;
        } else {
            this->outputs["raw_data"].data = output_tensors[0];
            this->outputs["source_size"].data =
                std::vector<int>{static_cast<int>(output_sizes[0].height), static_cast<int>(output_sizes[0].width)};
        }
    }
}

void ImagePreprocessModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    bool has_images_input = exists_input("images");
    bool has_image_input = exists_input("image");
    bool has_image = has_images_input || has_image_input;
    if (has_image) {
        OPENVINO_ASSERT(
            !(has_images_input && has_image_input),
            "ImagePreprocessModule: Both 'image' and 'images' inputs exist. Please provide only one of them.");
    }

    if (has_images_input || has_image_input) {
      run_image(has_images_input);
    } else {
        OPENVINO_THROW("ImagePreprocessModule[" + module_desc->name +
                       "]: No valid input found. Please provide one of the following inputs: 'image', 'images'.");
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov
