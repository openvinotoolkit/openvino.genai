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

class EmbeddingMergerModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;
    float _threshold = 1e-2;

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

        std::string pipeline_params_name = "pipeline_params";
        {
            YAML::Node pipeline_params;
            pipeline_params["type"] = "ParameterModule";
            pipeline_params["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            pipeline_params["outputs"].push_back(output_node("input_ids", to_string(DataType::OVTensor)));
            pipeline_params["outputs"].push_back(output_node("input_embedding", to_string(DataType::OVTensor)));
            pipeline_params["outputs"].push_back(output_node("image_embedding", to_string(DataType::OVTensor)));
            pipeline_params["outputs"].push_back(output_node("video_embedding", to_string(DataType::OVTensor)));
            pipeline_modules[pipeline_params_name] = pipeline_params;
        }

        std::string embedding_merger_name = "embedding_merger";
        {
            YAML::Node embedding_merger;
            embedding_merger["type"] = "EmbeddingMergerModule";
            embedding_merger["device"] = _device;
            embedding_merger["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            embedding_merger["inputs"].push_back(
                input_node("input_ids", to_string(DataType::OVTensor), pipeline_params_name + ".input_ids"));
            embedding_merger["inputs"].push_back(
                input_node("input_embedding", to_string(DataType::OVTensor), pipeline_params_name + ".input_embedding"));
            embedding_merger["inputs"].push_back(
                input_node("image_embedding", to_string(DataType::OVTensor), pipeline_params_name + ".image_embedding"));
            embedding_merger["inputs"].push_back(
                input_node("video_embedding", to_string(DataType::OVTensor), pipeline_params_name + ".video_embedding"));
            embedding_merger["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            embedding_merger["outputs"].push_back(output_node("merged_embedding", to_string(DataType::OVTensor)));
            embedding_merger["params"] = YAML::Node();
            embedding_merger["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
            pipeline_modules[embedding_merger_name] = embedding_merger;
        }

        {
            YAML::Node pipeline_results;
            pipeline_results["type"] = "ResultModule";
            pipeline_results["device"] = "CPU";
            pipeline_results["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            pipeline_results["inputs"].push_back(
                input_node("merged_embedding", to_string(DataType::OVTensor), embedding_merger_name + ".merged_embedding"));
            pipeline_modules["pipeline_results"] = pipeline_results;
        }

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        auto input_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 6});
        int64_t* data_ptr = input_ids.data<int64_t>();
        std::vector<int64_t> values = {1986, 374, 264, 6077, 9934, 13};
        std::copy(values.begin(), values.end(), data_ptr);

        ov::Tensor input_embedding = ut_randn_tensor(ov::Shape{1, 6, 2048}, 42);
        ov::Tensor image_embedding = ut_randn_tensor(ov::Shape{16, 2048}, 43);
        ov::Tensor video_embedding = ut_randn_tensor(ov::Shape{32, 2048}, 44);

        inputs["input_ids"] = input_ids;
        inputs["input_embedding"] = input_embedding;
        inputs["image_embedding"] = image_embedding;
        inputs["video_embedding"] = video_embedding;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("merged_embedding").as<ov::Tensor>();

        const std::vector<float> expected_merged_embeds =
            {0.37454, 0.796543, 0.950714, 0.183435, 0.731994, 0.779691, 0.598659, 0.59685, 0.156019, 0.445833};
        EXPECT_TRUE(compare_big_tensor(output, expected_merged_embeds, _threshold))
            << "merged_embedding do not match expected values within threshold " << _threshold;
        EXPECT_TRUE(compare_shape(output.get_shape(), ov::Shape{1, 6, 2048}))
            << "merged_embedding's shape not match expected shape";
    }
};

TEST_P(EmbeddingMergerModuleTest, ModuleTest) {
    run();
}

static auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         EmbeddingMergerModuleTest,
                         ::testing::Combine(::testing::ValuesIn(test_devices)),
                         EmbeddingMergerModuleTest::get_test_case_name);
