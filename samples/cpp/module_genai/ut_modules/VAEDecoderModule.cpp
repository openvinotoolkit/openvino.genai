// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class VAEDecoderModule : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(VAEDecoderModule)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:

  vae_decoder:
    type: "VAEDecoderModule"
    device: "CPU"
    inputs:
      - name: "latents"
        type: "OVTensor"
        source: "pipeline_params.latent_input"
    outputs:
      - name: "image"
        type: "OVTensor"
    params:
      model_path: "./ut_pipelines/Z-Image-Turbo-fp16-ov/"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        // Latent shape for VAE decoder usually depends on the model config.
        // The model expects 16 channels (e.g. Z-Image-Turbo).
        // Using a small size for testing.
        auto latent = ut_randn_tensor(ov::Shape{1, 16, 64, 64}, 42);
        inputs["latent_input"] = latent;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto image = pipe.get_output("image").as<ov::Tensor>();
        CHECK(image.get_size() > 0, "VAE decoder output is empty");
        std::vector<uint8_t> expected_ouput = { 120, 150, 165, 104, 128, 145, 93, 108, 127, 92 };
        CHECK(compare_big_tensor<uint8_t>(image, expected_ouput), "decoder output do not match expected values");
    }
};

class VAEDecoderModuleSkipPostProcess : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(VAEDecoderModuleSkipPostProcess)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:

  vae_decoder:
    type: "VAEDecoderModule"
    device: "CPU"
    inputs:
      - name: "latents"
        type: "OVTensor"
        source: "pipeline_params.latent_input"
    outputs:
      - name: "image"
        type: "OVTensor"
    params:
      model_path: "./ut_pipelines/Z-Image-Turbo-fp16-ov/"
      enable_postprocess: "false"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        auto latent = ut_randn_tensor(ov::Shape{1, 16, 64, 64}, 42);
        inputs["latent_input"] = latent;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto image = pipe.get_output("image").as<ov::Tensor>();
        CHECK(image.get_size() > 0, "VAE decoder output is empty");
        std::vector<float> expected_ouput = {-0.0553406, -0.177907, -0.268292, -0.276685, -0.182945,
                                             0.0203716, 0.255036, 0.342477, 0.279524, 0.0960406};

        CHECK(compare_big_tensor<float>(image, expected_ouput, 1e-6), "decoder output do not match expected values");
    }
};

REGISTER_MODULE_TEST(VAEDecoderModule)
REGISTER_MODULE_TEST(VAEDecoderModuleSkipPostProcess)
