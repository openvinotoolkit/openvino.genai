// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class TextEncoderModuleTest : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(TextEncoderModuleTest)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "qwen2_5_vl"
pipeline_modules:

  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "img1"
        type: "OVTensor"
      - name: "prompts_data"
        type: "String"
  image_preprocessor:       # Module Name
    type: "ImagePreprocessModule"
    device: "CPU"
    description: "Image or Video preprocessing."
    inputs:
      - name: "image"     # single image
        type: "OVTensor"        # Support DataType: [OVTensor, OVRemoteTensor]
        source: "pipeline_params.img1"
    outputs:
      - name: "raw_data"        # Output port name
        type: "OVTensor"        # Support DataType: [OVTensor, OVRemoteTensor]
      - name: "source_size"     # Output port name
        type: "VecInt"          # Support DataType: [VecInt]
    params:
      target_resolution: [224, 224]   # optional
      mean: [0.485, 0.456, 0.406]     # optional
      std: [0.229, 0.224, 0.225]      # optional
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"
  prompt_encoder:
    type: "TextEncoderModule"
    device: "GPU"
    inputs:
      - name: "prompts"
        type: "String"
        source: "pipeline_params.prompts_data"
      - name: "encoded_image"
        type: "OVTensor"
        source: "image_preprocessor.raw_data"
      - name: "source_size"
        type: "VecInt"
        source: "image_preprocessor.source_size"
    outputs:
      - name: "input_ids"
        type: "OVTensor"
      - name: "mask"
        type: "OVTensor"
      - name: "images_sequence"
        type: "VecInt"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "input_ids"
        type: "OVTensor"
        source: "prompt_encoder.input_ids"
      - name: "mask"
        type: "OVTensor"
        source: "prompt_encoder.mask"
      - name: "images_sequence"
        type: "VecInt"
        source: "prompt_encoder.images_sequence"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["prompts_data"] = std::vector<std::string>{"This is a sample prompt."};
        auto img1 = image_utils::load_image("ut_test_data/cat_120_100.png");
        CHECK(img1, "Failed to load test image: ut_test_data/cat_120_100.png");
        inputs["img1"] = img1;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("input_ids").as<ov::Tensor>();
        std::vector<int64_t> expected_input_ids = {151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645};

        CHECK(compare_big_tensor<int64_t>(output, expected_input_ids), "input_ids do not match expected values");

        auto mask = pipe.get_output("mask").as<ov::Tensor>();
        std::vector<int64_t> expected_mask = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        CHECK(compare_big_tensor<int64_t>(mask, expected_mask), "mask not match expected values");
        
        auto images_sequence = pipe.get_output("images_sequence").as<std::vector<int>>();
        std::vector<int> expected_images_sequence = {0};
        CHECK(images_sequence == expected_images_sequence, "images_sequence do not match expected values");
    }
};

REGISTER_MODULE_TEST(TextEncoderModuleTest);
