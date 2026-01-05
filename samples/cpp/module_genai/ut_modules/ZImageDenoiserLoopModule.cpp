// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class ZImageDenoiserLoopModule : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(ZImageDenoiserLoopModule)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:

  denoiser_loop:
    type: "ZImageDenoiserLoopModule"
    device: "CPU"
    inputs:
      - name: "prompt_embed"
        type: "OVTensor"
      - name: "num_inference_steps"               # [optional], default 10
        type: "Int"
      - name: "width"                             # [optional], default 512
        type: "Int"
      - name: "height"                            # [optional], default 512
        type: "Int"
    outputs:
      - name: "latents"
        type: "OVTensor"
    params:
      model_path: "./ut_pipelines/Z-Image-Turbo-fp16-ov/"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        auto prompt_embed = ut_randn_tensor(ov::Shape{101, 2560}, 42);
        inputs["prompt_embed"] = prompt_embed;
        inputs["num_inference_steps"] = 2;
        inputs["width"] = 128;
        inputs["height"] = 128;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("latents").as<ov::Tensor>();
        std::vector<float> expected_ouput = { 
          0.0279331, -0.0194968, -0.158097, 0.142582, -0.313633, -0.452601, 0.107033, 0.305759, -0.0610831, 0.136313
        };
        CHECK(compare_big_tensor(output, expected_ouput, 1e-2), "latent do not match expected values");
    }
};

REGISTER_MODULE_TEST(ZImageDenoiserLoopModule);
