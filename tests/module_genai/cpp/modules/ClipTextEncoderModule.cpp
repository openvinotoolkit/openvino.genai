// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"
#include "module_genai/diffusion_model_type.hpp"
#include <regex>

using namespace ov::genai::module;

struct ClipTextEncoderPositiveTestDataCommon {
    ov::genai::DiffusionModelType model_type;
    std::string model_path;
    ov::Shape expected_embeds_shape;
    ov::Shape expected_negative_embeds_shape;
    std::vector<float> expected_embeds;
    std::vector<float> expected_negative_embeds;
};

struct ClipTextEncoderPositiveTestData : public ClipTextEncoderPositiveTestDataCommon {
    std::string prompt;
    float guidance_scale;
};

struct ClipTextEncoderNegativeTestData : public ClipTextEncoderPositiveTestDataCommon {
    std::string negative_prompt;
    float guidance_scale;
};

struct ClipTextEncoderPosNegTestData : public ClipTextEncoderPositiveTestDataCommon {
    std::string prompt;
    std::string negative_prompt;
    float guidance_scale;
};

namespace TEST_DATA {

ClipTextEncoderPositiveTestData z_image_clip_text_encoder_1_test_data() {
    ClipTextEncoderPositiveTestData data;
    data.model_type = ov::genai::DiffusionModelType::ZIMAGE;
    data.model_path = TEST_MODEL::ZImage_Turbo_fp16_ov();
    data.prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.";
    data.guidance_scale = 0.0f;
    data.expected_embeds_shape = ov::Shape{101, 2560};
    data.expected_embeds = { 
        -603.058, -6.29294, -24.5433, 33.7109, 13672.3, -8.1542, -6.41789, 25.8192, 6.84497, 67.1992
    };
    return data;
}

ClipTextEncoderNegativeTestData z_image_clip_text_encoder_2_test_data() {
    ClipTextEncoderNegativeTestData data;
    data.model_type = ov::genai::DiffusionModelType::ZIMAGE;
    data.model_path = TEST_MODEL::ZImage_Turbo_fp16_ov();
    data.negative_prompt = "blurry ugly bad";
    data.guidance_scale = 2.0f;
    data.expected_negative_embeds_shape = ov::Shape{12, 2560};
    data.expected_negative_embeds = { 
        -603.058, -6.29294, -24.5433, 33.7109, 13672.3, -8.1542, -6.41789, 25.8192, 6.84497, 67.1992
    };
    return data;
}

ClipTextEncoderPosNegTestData z_image_clip_text_encoder_3_test_data() {
    ClipTextEncoderPosNegTestData data;
    data.model_type = ov::genai::DiffusionModelType::ZIMAGE;
    data.model_path = TEST_MODEL::ZImage_Turbo_fp16_ov();
    data.prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.";
    data.negative_prompt = "blurry ugly bad";
    data.guidance_scale = 2.0f;
    data.expected_embeds_shape = ov::Shape{101, 2560};
    data.expected_embeds = { 
        -603.058, -6.29294, -24.5433, 33.7109, 13672.3, -8.1542, -6.41789, 25.8192, 6.84497, 67.1992
    };
    data.expected_negative_embeds_shape = ov::Shape{12, 2560};
    data.expected_negative_embeds = { 
        -603.058, -6.29294, -24.5433, 33.7109, 13672.3, -8.1542, -6.41789, 25.8192, 6.84497, 67.1992
    };
    return data;
}

ClipTextEncoderPosNegTestData wan_2_1_clip_text_encoder_1_test_data() {
    ClipTextEncoderPosNegTestData data;
    data.model_type = ov::genai::DiffusionModelType::WAN_2_1;
    data.model_path = TEST_MODEL::Wan_2_1();
    data.prompt = "A fantasy landscape with mountains and a river, vibrant colors, highly detailed, digital art";
    data.negative_prompt = "low quality, blurry, distorted";
    data.guidance_scale = 5.0f;
    data.expected_embeds_shape = ov::Shape{512,4096};
    data.expected_embeds = { 
        0.00152739, 0.0495682, -0.0664976, 0.125089, -0.00320548, 0.145041, 0.0220366, -0.00679807, -0.009486, 0.118701
    };
    data.expected_negative_embeds_shape = ov::Shape{512,4096};
    data.expected_negative_embeds = { 
        0.00205161, -0.0196467, 0.0319586, 0.117366, 0.0435993, 0.0243144, 0.0910006, -0.130668, -0.007705, -0.0799274
    };
    return data;
}

}

using ClipTextEncoderTestData = std::variant<ClipTextEncoderPositiveTestData,
                                             ClipTextEncoderNegativeTestData,
                                             ClipTextEncoderPosNegTestData>;
using test_params = std::tuple<ClipTextEncoderTestData, std::string>;

class ClipTextEncoderModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    ClipTextEncoderTestData m_test_data;
    float m_threshold = 1e+1;
    int max_sequence_length = 512;

public:
    static std::string test_data_name(const ClipTextEncoderTestData& data) {
        if (std::holds_alternative<ClipTextEncoderPositiveTestData>(data)) {
            std::string model_type = std::regex_replace(
                ov::genai::diffusion_model_type_to_string(std::get<ClipTextEncoderPositiveTestData>(data).model_type),
                std::regex("\\."), "_");
            return "Positive_Prompt_" + model_type;
        } else if (std::holds_alternative<ClipTextEncoderNegativeTestData>(data)) {
            std::string model_type = std::regex_replace(
                ov::genai::diffusion_model_type_to_string(std::get<ClipTextEncoderNegativeTestData>(data).model_type),
                std::regex("\\."), "_");
            return "Negative_Prompt_" + model_type;
        } else if (std::holds_alternative<ClipTextEncoderPosNegTestData>(data)) {
            std::string model_type = std::regex_replace(
                ov::genai::diffusion_model_type_to_string(std::get<ClipTextEncoderPosNegTestData>(data).model_type),
                std::regex("\\."), "_");
            return "Positive_Negative_Prompt_" + model_type;
        }
        return "Unknown";
    }

    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& data = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        return test_data_name(data) + "_" + device;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_test_data, m_device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        ov::genai::DiffusionModelType model_type;
        std::string model_path;
        if (auto* pos = std::get_if<ClipTextEncoderPositiveTestData>(&m_test_data)) {
            model_type = pos->model_type;
            model_path = pos->model_path;
        } else if (auto* neg = std::get_if<ClipTextEncoderNegativeTestData>(&m_test_data)) {
            model_type = neg->model_type;
            model_path = neg->model_path;
        } else if (auto* pos_neg = std::get_if<ClipTextEncoderPosNegTestData>(&m_test_data)) {
            model_type = pos_neg->model_type;
            model_path = pos_neg->model_path;
        }
        config["global_context"]["model_type"] = ov::genai::diffusion_model_type_to_string(model_type);
        YAML::Node pipeline_modules = config["pipeline_modules"];
        YAML::Node clip_text_encoder;
        clip_text_encoder["type"] = "ClipTextEncoderModule";
        clip_text_encoder["device"] = m_device;
        YAML::Node inputs;
        inputs.push_back(input_node("prompt", "String"));
        inputs.push_back(input_node("negative_prompt", "String"));
        inputs.push_back(input_node("guidance_scale", "Float"));
        inputs.push_back(input_node("max_sequence_length", "Int"));
        clip_text_encoder["inputs"] = inputs;
        YAML::Node outputs;
        outputs.push_back(output_node("prompt_embeds", "VecOVTensor"));
        outputs.push_back(output_node("negative_prompt_embeds", "VecOVTensor"));
        clip_text_encoder["outputs"] = outputs;
        YAML::Node params;
        params["model_path"] = model_path;
        clip_text_encoder["params"] = params;
        pipeline_modules["clip_text_encoder"] = clip_text_encoder;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        if (auto* pos = std::get_if<ClipTextEncoderPositiveTestData>(&m_test_data)) {
            inputs["prompt"] = pos->prompt;
            inputs["negative_prompt"] = "";
            inputs["guidance_scale"] = pos->guidance_scale;
        } else if (auto* neg = std::get_if<ClipTextEncoderNegativeTestData>(&m_test_data)) {
            inputs["prompt"] = "";
            inputs["negative_prompt"] = neg->negative_prompt;
            inputs["guidance_scale"] = neg->guidance_scale;
        } else if (auto* pos_neg = std::get_if<ClipTextEncoderPosNegTestData>(&m_test_data)) {
            inputs["prompt"] = pos_neg->prompt;
            inputs["negative_prompt"] = pos_neg->negative_prompt;
            inputs["guidance_scale"] = pos_neg->guidance_scale;
        }
        inputs["max_sequence_length"] = max_sequence_length;
        return inputs;
    }

    void check_outputs_input_1(ov::genai::module::ModulePipeline& pipe) {
        auto output = pipe.get_output("prompt_embeds").as<std::vector<ov::Tensor>>();
        const auto &test_data = std::get<ClipTextEncoderPositiveTestData>(m_test_data);

        EXPECT_TRUE(compare_shape(output[0].get_shape(), test_data.expected_embeds_shape))
            << "prompt_embeds shape does not match expected shape";
        EXPECT_TRUE(compare_big_tensor(output[0], test_data.expected_embeds, m_threshold))
            << "negative_prompt_embeds do not match expected values within threshold " << m_threshold;
    }

    void check_outputs_input_2(ov::genai::module::ModulePipeline& pipe) {
        auto output = pipe.get_output("negative_prompt_embeds").as<std::vector<ov::Tensor>>();
        const auto &test_data = std::get<ClipTextEncoderNegativeTestData>(m_test_data);

        EXPECT_TRUE(compare_shape(output[0].get_shape(), test_data.expected_negative_embeds_shape))
            << "negative_prompt_embeds shape does not match expected shape";
        EXPECT_TRUE(compare_big_tensor(output[0], test_data.expected_negative_embeds, m_threshold))
            << "negative_prompt_embeds do not match expected values within threshold " << m_threshold;
    }

    void check_outputs_input_3(ov::genai::module::ModulePipeline& pipe) {
        auto pos_output = pipe.get_output("prompt_embeds").as<std::vector<ov::Tensor>>();
        auto neg_output = pipe.get_output("negative_prompt_embeds").as<std::vector<ov::Tensor>>();
        const auto &test_data = std::get<ClipTextEncoderPosNegTestData>(m_test_data);

        EXPECT_TRUE(compare_shape(pos_output[0].get_shape(), test_data.expected_embeds_shape))
            << "positive_prompt_embeds shape does not match expected shape";
        EXPECT_TRUE(compare_big_tensor(pos_output[0], test_data.expected_embeds, m_threshold))
            << "positive_prompt_embeds do not match expected values within threshold " << m_threshold;
        EXPECT_TRUE(compare_shape(neg_output[0].get_shape(), test_data.expected_negative_embeds_shape))
            << "negative_prompt_embeds shape does not match expected shape";
        EXPECT_TRUE(compare_big_tensor(neg_output[0], test_data.expected_negative_embeds, m_threshold))
            << "negative_prompt_embeds do not match expected values within threshold " << m_threshold;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        if (auto* pos = std::get_if<ClipTextEncoderPositiveTestData>(&m_test_data)) {
            check_outputs_input_1(pipe);
        } else if (auto* neg = std::get_if<ClipTextEncoderNegativeTestData>(&m_test_data)) {
            check_outputs_input_2(pipe);
        } else if (auto* pos_neg = std::get_if<ClipTextEncoderPosNegTestData>(&m_test_data)) {
            check_outputs_input_3(pipe);
        }
    }
};

TEST_P(ClipTextEncoderModuleTest, ModuleTest) {
    run();
}

namespace clip_text_encoder_test {
    std::vector<ClipTextEncoderTestData> all_test_data = {
        TEST_DATA::z_image_clip_text_encoder_1_test_data(),
        TEST_DATA::z_image_clip_text_encoder_2_test_data(),
        TEST_DATA::z_image_clip_text_encoder_3_test_data(),
        TEST_DATA::wan_2_1_clip_text_encoder_1_test_data()
    };
    auto test_devices = std::vector<std::string> {TEST_MODEL::get_device()};
}

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         ClipTextEncoderModuleTest,
                         ::testing::Combine(::testing::ValuesIn(clip_text_encoder_test::all_test_data),
                                            ::testing::ValuesIn(clip_text_encoder_test::test_devices)),
                         ClipTextEncoderModuleTest::get_test_case_name);
