// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class VisionEncoderTest : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(VisionEncoderTest)

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
      - name: "images_sequence"
        type: "VecInt"
      - name: "input_ids"
        type: "OVTensor"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"
  vision_encoder:
    type: "VisionEncoderModule"
    device: "GPU"
    inputs:
      - name: "preprocessed_image"
        type: "OVTensor"
        source: "image_preprocessor.raw_data"
      - name: "source_size"
        type: "VecInt"
        source: "image_preprocessor.source_size"
      - name: "images_sequence"
        type: "VecInt"
        source: "prompt_encoder.images_sequence"
      - name: "input_ids"
        type: "OVTensor"
        source: "prompt_encoder.input_ids"
    outputs:
      - name: "image_embedding"
        type: "OVTensor"
      - name: "position_ids"
        type: "OVTensor"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"
      vision_start_token_id: 151652

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "image_embedding"
        type: "OVTensor"
        source: "vision_encoder.image_embedding"
      - name: "position_ids"
        type: "OVTensor"
        source: "vision_encoder.position_ids"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
      
        auto img1 = utils::load_image("ut_test_data/cat_120_100.png");
        CHECK(img1, "Failed to load test image: ut_test_data/cat_120_100.png");
        inputs["prompts_data"] = std::vector<std::string>{"This is a sample prompt."};
        inputs["img1"] = img1;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto image_embedding = pipe.get_output("image_embedding").as<ov::Tensor>();
        std::vector<float> expected_image_embedding = { 
            -1.03223, 0.269775, 0.316406, 1.99805, -1.7666, 1.14746, 1.60254, 1.89453, 3.13086, 0.59082
        };
        CHECK(compare_big_tensor<float>(image_embedding, expected_image_embedding, 0.2f), "image_embedding do not match expected values");

        auto position_ids = pipe.get_output("position_ids").as<ov::Tensor>();
        CHECK(compare_shape(position_ids.get_shape(), ov::Shape({3, 1, 43})), "position_ids shape do not match expected values");
    }
};

REGISTER_MODULE_TEST(VisionEncoderTest);
