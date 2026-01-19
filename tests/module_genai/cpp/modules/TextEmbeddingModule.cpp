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

class TextEmbeddingModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;
    float _threshold = 1e-5;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& device = std::get<0>(obj.param);
        std::string result;
        result += "_" + device;
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

        std::string text_embedding_name = "text_embedding";
        {
            YAML::Node text_embedding;
            text_embedding["type"] = "TextEmbeddingModule";
            text_embedding["device"] = _device;
            text_embedding["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            text_embedding["inputs"].push_back(
                input_node("input_ids", to_string(DataType::OVTensor), "pipeline_params.input_ids"));
            text_embedding["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            text_embedding["outputs"].push_back(output_node("input_embedding", to_string(DataType::OVTensor)));
            text_embedding["params"] = YAML::Node();
            text_embedding["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
            text_embedding["params"]["scale_emb"] = "1.0";
            pipeline_modules[text_embedding_name] = text_embedding;
        }

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        auto input_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 6});
        int64_t* data_ptr = input_ids.data<int64_t>();
        std::vector<int64_t> values = {1986, 374, 264, 6077, 9934, 13};
        std::copy(values.begin(), values.end(), data_ptr);

        inputs["input_ids"] = input_ids;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("input_embedding").as<ov::Tensor>();
        const std::vector<float> expected_text_embeds = { 
            0.0129318, 0.000862122, 0.0021553, 0, -0.0133667, 0.0168152, 0.00387955, 0.0021553, -0.0375061, -0.0241394
        };
        EXPECT_TRUE(compare_big_tensor(output, expected_text_embeds, _threshold)) << "input_embedding do not match expected values within threshold " << _threshold;
        EXPECT_TRUE(compare_shape(output.get_shape(), ov::Shape{1, 6, 2048})) << "input_embedding's shape not match expected shape";
    }
};

TEST_P(TextEmbeddingModuleTest, ModuleTest) {
    run();
}

static auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         TextEmbeddingModuleTest,
                         ::testing::Combine(::testing::ValuesIn(test_devices)),
                         TextEmbeddingModuleTest::get_test_case_name);
