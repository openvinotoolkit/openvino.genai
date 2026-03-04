// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/load_image.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"

// Define test parameters:
// std::string: device;
using test_params = std::tuple<std::string>;
using namespace ov::genai::module;

class TextEncoderModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& device = std::get<0>(obj.param);
        std::string result;
        result += device;
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(_device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "qwen2_5_vl";
        YAML::Node pipeline_modules = config["pipeline_modules"];

        std::string pipeline_params_name = "pipeline_params";
        {
            YAML::Node cur_node;
            cur_node["type"] = "ParameterModule";
            cur_node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["outputs"].push_back(output_node("img1", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("prompts_data", to_string(DataType::String)));
            pipeline_modules[pipeline_params_name] = cur_node;
        }

        std::string image_preprocessor_name = "image_preprocessor";
        {
            YAML::Node cur_node;
            cur_node["type"] = "ImagePreprocessModule";
            cur_node["device"] = _device;
            cur_node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["inputs"].push_back(
                input_node("image", to_string(DataType::OVTensor), pipeline_params_name + ".img1"));
            cur_node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["outputs"].push_back(output_node("raw_data", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("source_size", to_string(DataType::VecInt)));
            cur_node["params"] = YAML::Node();
            cur_node["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
            pipeline_modules[image_preprocessor_name] = cur_node;
        }

        std::string prompt_encoder_name = "prompt_encoder";
        {
            YAML::Node cur_node;
            cur_node["type"] = "TextEncoderModule";
            cur_node["device"] = _device;
            cur_node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["inputs"].push_back(
                input_node("prompts", to_string(DataType::String), pipeline_params_name + ".prompts_data"));
            cur_node["inputs"].push_back(
                input_node("encoded_image", to_string(DataType::OVTensor), image_preprocessor_name + ".raw_data"));
            cur_node["inputs"].push_back(
                input_node("source_size", to_string(DataType::VecInt), image_preprocessor_name + ".source_size"));
            cur_node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["outputs"].push_back(output_node("input_ids", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("mask", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("images_sequence", to_string(DataType::VecInt)));
            cur_node["params"] = YAML::Node();
            cur_node["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
            pipeline_modules[prompt_encoder_name] = cur_node;
        }

        std::string pipeline_results_name = "pipeline_results";
        {
            YAML::Node cur_node;
            cur_node["type"] = "ResultModule";
            cur_node["device"] = "CPU";
            cur_node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["inputs"].push_back(
                input_node("input_ids", to_string(DataType::OVTensor), prompt_encoder_name + ".input_ids"));
            cur_node["inputs"].push_back(
                input_node("mask", to_string(DataType::OVTensor), prompt_encoder_name + ".mask"));
            cur_node["inputs"].push_back(
                input_node("images_sequence", to_string(DataType::VecInt), prompt_encoder_name + ".images_sequence"));
            pipeline_modules[pipeline_results_name] = cur_node;
        }

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["prompts_data"] = std::vector<std::string>{"This is a sample prompt."};
        auto img1 = utils::load_image(TEST_DATA::img_cat_120_100());
        EXPECT_TRUE(img1) << "Failed to load test image: " + TEST_DATA::img_cat_120_100();
        inputs["img1"] = img1;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("input_ids").as<ov::Tensor>();
        const std::vector<int64_t> expected_input_ids = {151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645};

        EXPECT_TRUE(compare_big_tensor<int64_t>(output, expected_input_ids)) << "input_ids do not match expected values";

        auto mask = pipe.get_output("mask").as<ov::Tensor>();
        std::vector<int64_t> expected_mask = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        EXPECT_TRUE(compare_big_tensor<int64_t>(mask, expected_mask)) << "mask not match expected values";

        auto images_sequence = pipe.get_output("images_sequence").as<std::vector<int>>();
        std::vector<int> expected_images_sequence = {0};
        EXPECT_TRUE(images_sequence == expected_images_sequence) << "images_sequence do not match expected values";
    }
};

TEST_P(TextEncoderModuleTest, ModuleTest) {
    run();
}

static auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         TextEncoderModuleTest,
                         ::testing::Combine(::testing::ValuesIn(test_devices)),
                         TextEncoderModuleTest::get_test_case_name);


class Qwen3_5TextEncoderModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& device = std::get<0>(obj.param);
        std::string result {"Qwen3_5_"};
        result += device;
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(_device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "qwen3_5";
        YAML::Node pipeline_modules = config["pipeline_modules"];

        std::string pipeline_params_name = "pipeline_params";
        {
            YAML::Node cur_node;
            cur_node["type"] = "ParameterModule";
            cur_node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["outputs"].push_back(output_node("image", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("prompts_data", to_string(DataType::String)));
            pipeline_modules[pipeline_params_name] = cur_node;
        }

        std::string image_preprocessor_name = "image_preprocessor";
        {
            YAML::Node cur_node;
            cur_node["type"] = "ImagePreprocessModule";
            cur_node["device"] = _device;
            cur_node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["inputs"].push_back(
                input_node("image", to_string(DataType::OVTensor), pipeline_params_name + ".image"));
            cur_node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["outputs"].push_back(output_node("pixel_values", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("grid_thw", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("pos_embeds", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("rotary_cos", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("rotary_sin", to_string(DataType::OVTensor)));
            cur_node["params"] = YAML::Node();
            cur_node["params"]["model_path"] = TEST_MODEL::Qwen3_5_0_8B();
            pipeline_modules[image_preprocessor_name] = cur_node;
        }

        std::string prompt_encoder_name = "prompt_encoder";
        {
            YAML::Node cur_node;
            cur_node["type"] = "TextEncoderModule";
            cur_node["device"] = _device;
            cur_node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["inputs"].push_back(
                input_node("prompts", to_string(DataType::String), pipeline_params_name + ".prompts_data"));
            cur_node["inputs"].push_back(
                input_node("grid_thw", to_string(DataType::OVTensor), image_preprocessor_name + ".grid_thw"));
            cur_node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["outputs"].push_back(output_node("input_ids", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("mask", to_string(DataType::OVTensor)));
            cur_node["params"] = YAML::Node();
            cur_node["params"]["model_path"] = TEST_MODEL::Qwen3_5();
            pipeline_modules[prompt_encoder_name] = cur_node;
        }

        std::string pipeline_results_name = "pipeline_results";
        {
            YAML::Node cur_node;
            cur_node["type"] = "ResultModule";
            cur_node["device"] = "CPU";
            cur_node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["inputs"].push_back(
                input_node("input_ids", to_string(DataType::OVTensor), prompt_encoder_name + ".input_ids"));
            cur_node["inputs"].push_back(
                input_node("mask", to_string(DataType::OVTensor), prompt_encoder_name + ".mask"));
            pipeline_modules[pipeline_results_name] = cur_node;
        }
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["prompts_data"] = std::vector<std::string>{"Describe this picture"};
        auto image = utils::load_image(TEST_DATA::img_dog_120_120());
        EXPECT_TRUE(image) << "Failed to load test image: " + TEST_DATA::img_dog_120_120();
        inputs["image"] = image;
        return inputs;
    }

    std::vector<int64_t> expected_input_ids = {
        248045, 846, 198, 248053, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056
    };
    ov::Shape expected_input_ids_shape = {1, 78};

    std::vector<int64_t> expected_mask = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };
    ov::Shape expected_mask_shape = {1, 78};

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto input_ids = pipe.get_output("input_ids").as<ov::Tensor>();
        auto mask = pipe.get_output("mask").as<ov::Tensor>();

        EXPECT_TRUE(compare_shape(input_ids.get_shape(), expected_input_ids_shape))
            << "input_ids's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(input_ids, expected_input_ids))
            << "input_ids do not match expected values";

        EXPECT_TRUE(compare_shape(mask.get_shape(), expected_mask_shape))
            << "mask's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(mask, expected_mask))
            << "mask do not match expected values";
    }
};

TEST_P(Qwen3_5TextEncoderModuleTest, ModuleTest) {
    run();
}

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         Qwen3_5TextEncoderModuleTest,
                         ::testing::Combine(::testing::ValuesIn(test_devices)),
                         Qwen3_5TextEncoderModuleTest::get_test_case_name);
