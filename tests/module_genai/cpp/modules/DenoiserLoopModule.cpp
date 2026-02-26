// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/load_image.hpp"
#include "module_genai/diffusion_model_type.hpp"
#include <regex>

using namespace ov::genai::module;

struct DenoiserLoopTestData {
    ov::genai::DiffusionModelType model_type;
    std::string model_path;
    bool splitted_model = false;
    bool dynamic_load_model_weights = false;
    ov::Tensor latents;
    ov::Tensor prompt_embed;
    ov::Tensor negative_prompt_embed;
    ov::Tensor init_latents;
    int num_inference_steps;
    float guidance_scale;
    std::vector<float> expected_latents;
};

namespace TEST_DATA {

std::vector<DenoiserLoopTestData> denoiser_loop_test_data() {
    std::vector<DenoiserLoopTestData> datas;
    DenoiserLoopTestData z_image_data;
    z_image_data.model_type = ov::genai::DiffusionModelType::ZIMAGE;
    z_image_data.model_path = TEST_MODEL::ZImage_Turbo_fp16_ov();
    z_image_data.latents = ModuleTestBase::ut_randn_tensor(ov::Shape{1, 16, 16, 16}, 42);
    z_image_data.prompt_embed = ModuleTestBase::ut_randn_tensor(ov::Shape{101, 2560}, 42);
    z_image_data.num_inference_steps = 2;
    z_image_data.guidance_scale = 0.0f;
    z_image_data.expected_latents = {
        0.515862, -0.0482551, 0.392357, 0.0667839, 0.445243, -0.077749, 0.373029, 0.0833834, 0.390022, 0.0221919
    };
    datas.push_back(z_image_data);
    DenoiserLoopTestData wan_data;
    wan_data.model_type = ov::genai::DiffusionModelType::WAN_2_1;
    wan_data.model_path = TEST_MODEL::Wan_2_1();
    wan_data.latents = ModuleTestBase::ut_randn_tensor(ov::Shape{1, 16, 2, 16, 16}, 42);
    wan_data.prompt_embed = ModuleTestBase::ut_randn_tensor(ov::Shape{16, 4096}, 42);
    wan_data.negative_prompt_embed = ModuleTestBase::ut_randn_tensor(ov::Shape{16, 4096}, 42);
    wan_data.num_inference_steps = 2;
    wan_data.guidance_scale = 5.0f;
    wan_data.expected_latents = {
        0.201578, 0.797599, 0.624354, 0.365458, 0.454261, 0.781906, 0.362604, 0.654469, 0.047139, 0.549425,
        0.0369669, 0.292062, -0.0306935, 0.553844, 0.545802, 0.459735, 0.822431, 0.74533, 0.913414, 1.11607
    };
    datas.push_back(wan_data);

    // Split model for Wan
    DenoiserLoopTestData wan_data_splitted_model = wan_data;
    wan_data_splitted_model.splitted_model = true;
    datas.push_back(wan_data_splitted_model);

#ifdef ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS
    // Dynamic load weights for Split model
    DenoiserLoopTestData wan_data_dyn_weights = wan_data;
    wan_data_dyn_weights.splitted_model = true;
    wan_data_dyn_weights.dynamic_load_model_weights = true;
    datas.push_back(wan_data_dyn_weights);
#endif
    return datas;
}

}

using test_params = std::tuple<DenoiserLoopTestData, std::string>;

class DenoiserLoopModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    DenoiserLoopTestData m_test_data;
    float m_threshold = 1e-1;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        std::string result;
        auto test_data = std::get<0>(obj.param);
        auto device = std::get<1>(obj.param);

        result = std::regex_replace(diffusion_model_type_to_string(test_data.model_type), std::regex("\\."), "_");
        result = result + "_" + device;
        result = result + "_SplittedModel_" + (test_data.splitted_model ? "true" : "false");
        result = result + "_DynamicLoadWeights_" + (test_data.dynamic_load_model_weights ? "true" : "false");
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_test_data, m_device) = GetParam();

        bool is_gpu = m_device.find("GPU") != std::string::npos || m_device.find("gpu") != std::string::npos;
        if (!is_gpu && m_test_data.dynamic_load_model_weights) {
            GTEST_SKIP() << "dynamic_load_model_weights is only supported for GPU device.";
        }
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = diffusion_model_type_to_string(m_test_data.model_type);
        YAML::Node pipeline_modules = config["pipeline_modules"];
        YAML::Node denoiser_loop;
        denoiser_loop["type"] = "DenoiserLoopModule";
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
        params["model_path"] = m_test_data.model_path;
        params["splitted_model"] = m_test_data.splitted_model ? "true" : "false";
        params["dynamic_load_weights"] = m_test_data.dynamic_load_model_weights ? "true" : "false";
        if (m_test_data.dynamic_load_model_weights) {
            params["cache_dir"] = "./unittest_cache_dir_denoiserloop";
        }
        denoiser_loop["params"] = params;
        pipeline_modules["denoiser_loop"] = denoiser_loop;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["latents"] = m_test_data.latents;
        inputs["prompt_embed"] = m_test_data.prompt_embed;
        if (m_test_data.model_type == ov::genai::DiffusionModelType::WAN_2_1) {
            inputs["prompt_embed_negative"] = m_test_data.negative_prompt_embed;
        }
        inputs["num_inference_steps"] = m_test_data.num_inference_steps;
        inputs["guidance_scale"] = m_test_data.guidance_scale;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto latents = pipe.get_output("latents").as<ov::Tensor>();
        std::vector<float> &expected_latents = m_test_data.expected_latents;

        EXPECT_TRUE(compare_shape(latents.get_shape(), m_test_data.latents.get_shape()))
            << "latents shape does not match expected shape";
        EXPECT_TRUE(compare_big_tensor(latents, expected_latents, m_threshold))
            << "latents do not match expected values within threshold " << m_threshold;
    }
};

TEST_P(DenoiserLoopModuleTest, ModuleTest) {
    run();
}

namespace denoiser_loop_test {

auto test_data = TEST_DATA::denoiser_loop_test_data();
auto test_devices = std::vector<std::string> {TEST_MODEL::get_device()};

}

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         DenoiserLoopModuleTest,
                         ::testing::Combine(::testing::ValuesIn(denoiser_loop_test::test_data),
                                            ::testing::ValuesIn(denoiser_loop_test::test_devices)),
                         DenoiserLoopModuleTest::get_test_case_name);
