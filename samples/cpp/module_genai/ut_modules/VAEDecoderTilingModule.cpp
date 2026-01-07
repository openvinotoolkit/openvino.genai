// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class VAEDecoderTilingModuleTest : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(VAEDecoderTilingModuleTest)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:
  vae_decoder_tiling:
    type: "VAEDecoderTilingModule"
    device: "CPU"
    inputs:
      - name: "latent"
        type: "OVTensor"
    outputs:
      - name: "image"
        type: "OVTensor"
    params:
      tile_overlap_factor: "0.25"
      model_path: "./ut_pipelines/Z-Image-Turbo-fp16-ov/"
      sub_module_name: "vae_decoder_submodule"

sub_modules:
  - name: "vae_decoder_submodule"
    vae_decoder:
      type: "VAEDecoderModule"
      device: "GPU"
      inputs:
        - name: "latents"
          type: "OVTensor"
      outputs:
        - name: "image"
          type: "OVTensor"
      params:
        model_path: "./ut_pipelines/Z-Image-Turbo-fp16-ov/"
        enable_postprocess: "false"   # Tiling decoder, don't need to do post-process
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        auto latent = ut_randn_tensor(ov::Shape{1, 16, 240, 240}, 42);
        inputs["latent"] = latent;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("image").as<ov::Tensor>();
        CHECK(output.get_element_type() == ov::element::u8, "Expect output data type is u8");

        std::vector<uint8_t> expected_ouput = {119, 113, 97, 107, 97, 81, 106, 93};
        CHECK(compare_big_tensor<uint8_t>(output, expected_ouput, 1), "latent do not match expected values");
    }
};

REGISTER_MODULE_TEST(VAEDecoderTilingModuleTest);
