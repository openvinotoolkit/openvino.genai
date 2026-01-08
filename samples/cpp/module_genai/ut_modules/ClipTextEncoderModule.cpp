// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class ClipTextEncoderModuleTest : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(ClipTextEncoderModuleTest)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:

  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "prompt"
        type: "String"
      - name: "guidance_scale"
        type: "Float"
      - name: "max_sequence_length"
        type: "Int"

  clip_text_encoder:
    type: "ClipTextEncoderModule"
    description: "Encode positive prompt and negative prompt"
    device: "GPU"
    inputs:
      - name: "prompt"
        type: "String"
        source: "pipeline_params.prompt"
      - name: "guidance_scale"
        type: "Float"
        source: "pipeline_params.guidance_scale"
      - name: "max_sequence_length"
        type: "Int"
        source: "pipeline_params.max_sequence_length"
    outputs:
      - name: "prompt_embeds"
        type: "VecOVTensor"
    params:
      model_path: "./ut_pipelines/Z-Image-Turbo-fp16-ov"

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "prompt_embeds"
        type: "OVTensor"
        source: "clip_text_encoder.prompt_embeds"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        inputs["prompt"] = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.";
        inputs["guidance_scale"] = 0.0;
        inputs["max_sequence_length"] = 512;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("prompt_embeds").as<std::vector<ov::Tensor>>();
        std::vector<float> expected_embeds = { 
          -5.8134e+02, 7.6859e-01, -3.0166e+01, 3.7192e+01, 1.3754e+04, -3.9269e+00, -8.7309e+00, 3.2020e+01, 6.0102e-01, 7.2419e+01
        };
        CHECK(compare_big_tensor<float>(output[0], expected_embeds, 1e+01), "embedding do not match expected values");
    }
};

REGISTER_MODULE_TEST(ClipTextEncoderModuleTest);
