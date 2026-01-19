// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"

using namespace ov::genai::module;

struct VAEDecoderTestData {
    std::string test_name;
    ov::Tensor latent_input;
    bool enable_postprocess;
    std::vector<uint8_t> expected_output_u8;
    std::vector<float> expected_output_f32;
    float threshold;
};

namespace TEST_DATA {

VAEDecoderTestData vae_decoder_with_postprocess() {
    VAEDecoderTestData data;
    data.test_name = "WithPostProcess";
    data.latent_input = ModuleTestBase::ut_randn_tensor(ov::Shape{1, 16, 64, 64}, 42);
    data.enable_postprocess = true;
    data.expected_output_u8 = {120, 150, 165, 104, 128, 145, 93, 108, 127, 92};
    data.threshold = 0.0f;
    return data;
}

VAEDecoderTestData vae_decoder_skip_postprocess() {
    VAEDecoderTestData data;
    data.test_name = "SkipPostProcess";
    data.latent_input = ModuleTestBase::ut_randn_tensor(ov::Shape{1, 16, 64, 64}, 42);
    data.enable_postprocess = false;
    data.expected_output_f32 = {-0.0553406f, -0.177907f, -0.268292f, -0.276685f, -0.182945f,
                                 0.0203716f, 0.255036f, 0.342477f, 0.279524f, 0.0960406f};
    data.threshold = 1e-6f;
    return data;
}

}  // namespace TEST_DATA

using test_params = std::tuple<VAEDecoderTestData, std::string>;

class VAEDecoderModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    VAEDecoderTestData m_test_data;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& test_data = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        return test_data.test_name + "_" + device;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_test_data, m_device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "zimage";

        YAML::Node pipeline_modules = config["pipeline_modules"];
        YAML::Node vae_decoder;
        vae_decoder["type"] = "VAEDecoderModule";
        vae_decoder["device"] = m_device;

        YAML::Node inputs;
        YAML::Node latents;
        latents["name"] = "latents";
        latents["type"] = "OVTensor";
        latents["source"] = "pipeline_params.latent_input";
        inputs.push_back(latents);
        vae_decoder["inputs"] = inputs;

        YAML::Node outputs;
        YAML::Node image;
        image["name"] = "image";
        image["type"] = "OVTensor";
        outputs.push_back(image);
        vae_decoder["outputs"] = outputs;

        YAML::Node params;
        params["model_path"] = TEST_MODEL::ZImage_Turbo_fp16_ov();
        if (!m_test_data.enable_postprocess) {
            params["enable_postprocess"] = "false";
        }
        vae_decoder["params"] = params;

        pipeline_modules["vae_decoder"] = vae_decoder;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["latent_input"] = m_test_data.latent_input;
        return inputs;
    }

    void check_outputs(ModulePipeline& pipe) override {
        auto image = pipe.get_output("image").as<ov::Tensor>();
        EXPECT_GT(image.get_size(), 0) << "VAE decoder output is empty";

        if (m_test_data.enable_postprocess) {
            EXPECT_TRUE(compare_big_tensor<uint8_t>(image, m_test_data.expected_output_u8))
                << "decoder output does not match expected values";
        } else {
            // Use non-template version that supports threshold for float comparison
            EXPECT_TRUE(compare_big_tensor(image, m_test_data.expected_output_f32, m_test_data.threshold))
                << "decoder output does not match expected values";
        }
    }
};

TEST_P(VAEDecoderModuleTest, ModuleTest) {
    run();
}

namespace vae_decoder_test {

auto test_data = std::vector<VAEDecoderTestData>{
    TEST_DATA::vae_decoder_with_postprocess(),
    TEST_DATA::vae_decoder_skip_postprocess()
};
auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};

}  // namespace vae_decoder_test

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         VAEDecoderModuleTest,
                         ::testing::Combine(::testing::ValuesIn(vae_decoder_test::test_data),
                                            ::testing::ValuesIn(vae_decoder_test::test_devices)),
                         VAEDecoderModuleTest::get_test_case_name);
