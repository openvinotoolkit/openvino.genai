// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/load_image.hpp"

struct ZImageDenoiserLoopTestData {
    ov::Tensor latents;
    ov::Tensor prompt_embed;
    ov::Tensor init_latents;
    int num_inference_steps;
    float guidance_scale;
};

namespace TEST_DATA {

ZImageDenoiserLoopTestData z_image_denoiser_loop_test_data() {
    ZImageDenoiserLoopTestData data;
    data.latents = ModuleTestBase::ut_randn_tensor(ov::Shape{1, 16, 16, 16}, 42);
    data.prompt_embed = ModuleTestBase::ut_randn_tensor(ov::Shape{101, 2560}, 42);
    data.num_inference_steps = 2;
    data.guidance_scale = 0.0f;
    return data;
}

}

using test_params = std::tuple<ZImageDenoiserLoopTestData, std::string>;

class ZImageDenoiserLoopModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    ZImageDenoiserLoopTestData m_test_data;
    float m_threshold = 1e-1;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& device = std::get<1>(obj.param);
        return device;
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
        YAML::Node denoiser_loop;
        denoiser_loop["type"] = "ZImageDenoiserLoopModule";
        denoiser_loop["device"] = m_device;
        YAML::Node inputs;
        YAML::Node input_latents;
        input_latents["name"] = "latents";
        input_latents["type"] = "OVTensor";
        inputs.push_back(input_latents);
        YAML::Node prompt_embed;
        prompt_embed["name"] = "prompt_embed";
        prompt_embed["type"] = "OVTensor";
        inputs.push_back(prompt_embed);
        YAML::Node num_inference_steps;
        num_inference_steps["name"] = "num_inference_steps";
        num_inference_steps["type"] = "Int";
        inputs.push_back(num_inference_steps);
        YAML::Node guidance_scale;
        guidance_scale["name"] = "guidance_scale";
        guidance_scale["type"] = "Float";
        inputs.push_back(guidance_scale);
        denoiser_loop["inputs"] = inputs;
        YAML::Node outputs;
        YAML::Node latents;
        latents["name"] = "latents";
        latents["type"] = "OVTensor";
        outputs.push_back(latents);
        denoiser_loop["outputs"] = outputs;
        YAML::Node params;
        params["model_path"] = TEST_MODEL::ZImage_Turbo_fp16_ov();
        denoiser_loop["params"] = params;
        pipeline_modules["denoiser_loop"] = denoiser_loop;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["latents"] = m_test_data.latents;
        inputs["prompt_embed"] = m_test_data.prompt_embed;
        inputs["num_inference_steps"] = m_test_data.num_inference_steps;
        inputs["guidance_scale"] = m_test_data.guidance_scale;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto latents = pipe.get_output("latents").as<ov::Tensor>();
        std::vector<float> expected_latents = {
            0.515862, -0.0482551, 0.392357, 0.0667839, 0.445243, -0.077749, 0.373029, 0.0833834, 0.390022, 0.0221919
        };

        EXPECT_TRUE(compare_shape(latents.get_shape(), ov::Shape{1, 16, 16, 16}))
            << "latents shape does not match expected shape";
        EXPECT_TRUE(compare_big_tensor(latents, expected_latents, m_threshold))
            << "latents do not match expected values within threshold " << m_threshold;
    }
};

TEST_P(ZImageDenoiserLoopModuleTest, ModuleTest) {
    run();
}

namespace z_image_denoiser_loop_test {

auto test_data = std::vector<ZImageDenoiserLoopTestData> {TEST_DATA::z_image_denoiser_loop_test_data()};
auto test_devices = std::vector<std::string> {TEST_MODEL::get_device()};

}

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         ZImageDenoiserLoopModuleTest,
                         ::testing::Combine(::testing::ValuesIn(z_image_denoiser_loop_test::test_data),
                                            ::testing::ValuesIn(z_image_denoiser_loop_test::test_devices)),
                         ZImageDenoiserLoopModuleTest::get_test_case_name);
