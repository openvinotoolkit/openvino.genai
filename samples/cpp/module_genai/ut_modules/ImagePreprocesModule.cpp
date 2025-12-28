// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class ImagePreprocesModuleTest : public ModuleTestBase {
public:
  DEFINE_MODULE_TEST_CONSTRUCTOR(ImagePreprocesModuleTest)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "qwen2_5_vl"
pipeline_modules:

  # When only specify one module,
  # ParameterModule and ResultModule will be automatically inferred and added
  # so there is no need to specify the inputs' `source`.

  image_preprocessor:       # Module Name
    type: "ImagePreprocessModule"
    device: "CPU"
    description: "Image or Video preprocessing."
    inputs:
      - name: "image"           # single image
        type: "OVTensor"        # Support DataType: [OVTensor, OVRemoteTensor]
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
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
      
        auto img1 = utils::load_image("ut_test_data/cat_120_100.png");
        CHECK(img1, "Failed to load test image: ut_test_data/cat_120_100.png");
        inputs["image"] = img1;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto raw_data = pipe.get_output("raw_data").as<ov::Tensor>();
        std::vector<float> expected_input_ids = { 
            -0.0712891, 0.251953, 0.0825195, 0.078125, 0.122559, 0.0986328, 0.0844727, -0.0932617, 0.130859, -0.0274658
        };
        CHECK(compare_big_tensor(raw_data, expected_input_ids, 1e-2), "raw_data do not match expected values");
        CHECK(compare_shape(raw_data.get_shape(), ov::Shape{64,1280}), "raw_data's shape not match expected shape");

        auto source_size = pipe.get_output("source_size").as<std::vector<int>>();
        auto expected_source_size = std::vector<int>{8, 8};
        CHECK(source_size == expected_source_size, "source_size not match expected values");
    }
};

REGISTER_MODULE_TEST(ImagePreprocesModuleTest);