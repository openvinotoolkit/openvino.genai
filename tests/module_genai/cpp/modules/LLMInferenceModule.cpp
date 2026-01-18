// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/load_image.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "utils.hpp"

// Parameters for test
// bool: true: model is from param; false: model is from models map
// bool: true: use batch input; false: use single input
// string: device
using test_params = std::tuple<bool, bool, std::string>;
using namespace ov::genai::module;

class LLMInferenceModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    bool _is_model_from_param = true;
    bool _use_batch_input = true;
    std::string _device;

    std::string _module_name = "cb_llm_infer";
    float _threshold = 1e-2;
    std::string _position_ids_path = get_data_path() + "/cb_llm/position_ids_0.bin";
    std::string _embeds_path = get_data_path() + "/cb_llm/embeds_tensor_0.bin";

    std::shared_ptr<ov::Model> _ov_llm_model = nullptr;
    std::shared_ptr<ov::Model> _ov_vlm_embed_merger_model = nullptr;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& is_model_from_param = std::get<0>(obj.param);
        const auto& use_batch_input = std::get<1>(obj.param);
        const auto& device = std::get<2>(obj.param);
        std::string result;
        result += "Device_" + device;
        result += std::string("_ModelFromParam") + (is_model_from_param ? "_True" : "_False");
        result += use_batch_input ? "_BatchInput" : "_SingleInput";
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(_is_model_from_param, _use_batch_input, _device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "qwen2_5_vl";

        YAML::Node pipeline_modules = config["pipeline_modules"];

        YAML::Node llm_inference;
        llm_inference["type"] = "LLMInferenceModule";
        llm_inference["device"] = _device;
        llm_inference["description"] = "LLM Inference Module.";
        YAML::Node inputs;
        if (_use_batch_input) {
            inputs.push_back(input_node("embeds_list", "VecOVTensor"));
            inputs.push_back(input_node("position_ids_list", "VecOVTensor"));
            inputs.push_back(input_node("rope_delta_list", "VecInt"));
        } else {
            inputs.push_back(input_node("embeds", "OVTensor"));
            inputs.push_back(input_node("position_ids", "VecOVTensor"));
            inputs.push_back(input_node("rope_delta", "Int"));
        }
        llm_inference["inputs"] = inputs;

        YAML::Node outputs;
        if (_use_batch_input) {
            outputs.push_back(output_node("generated_texts", "VecString"));
        } else {
            outputs.push_back(output_node("generated_text", "String"));
        }
        llm_inference["outputs"] = outputs;

        YAML::Node params;
        if (_is_model_from_param) {
            params["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        } else {
            params["model_cfg_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
            _ov_llm_model = ov::genai::utils::singleton_core().read_model(
                TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4() + "/openvino_language_model.xml");
            _ov_vlm_embed_merger_model = ov::genai::utils::singleton_core().read_model(
                TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4() + "/openvino_vision_embeddings_merger_model.xml");
            m_models_map[_module_name]["ov_model"] = _ov_llm_model;
            m_models_map[_module_name]["ov_model_embed"] = _ov_vlm_embed_merger_model;
        }
        params["max_new_tokens"] = "16";
        params["do_sample"] = "false";
        params["top_p"] = "1.0";
        params["top_k"] = "50";
        params["temperature"] = "1.0";
        params["repetition_penalty"] = "1.0";
        llm_inference["params"] = params;

        pipeline_modules[_module_name] = llm_inference;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        std::vector<ov::Tensor> input_embeds_list;
        load_test_data_input_embeds_list(input_embeds_list);
        EXPECT_GT(input_embeds_list.size(), 0) << "Failed to load input embeds list data";


        std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> input_position_ids_list;
        load_test_data_position_ids_list(input_position_ids_list);
        EXPECT_GT(input_position_ids_list.size(), 0) << "Failed to load position ids list data";
        std::vector<ov::Tensor> only_position_ids_list;
        std::vector<int> rope_delta_list;
        for (auto& pids : input_position_ids_list) {
            only_position_ids_list.push_back(pids.first);
            rope_delta_list.push_back(pids.second.has_value() ? pids.second.value() : 0);
        }

        if (_use_batch_input) {
            inputs["embeds_list"] = input_embeds_list;
            inputs["position_ids_list"] = only_position_ids_list;
            inputs["rope_delta_list"] = rope_delta_list;
        } else {
            inputs["embeds"] = input_embeds_list[0];
            inputs["position_ids"] = only_position_ids_list[0];
            inputs["rope_delta"] = rope_delta_list[0];
        }
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        if (_use_batch_input) {
            auto generated_texts = pipe.get_output("generated_texts").as<std::vector<std::string>>();
            for (const auto& text : generated_texts) {
                bool contains_white_cat = text.find("white cat") != std::string::npos;
                EXPECT_TRUE(contains_white_cat) << "can't find 'white cat' in generated text: " << text;
            }
        } else {
            auto generated_text = pipe.get_output("generated_text").as<std::string>();

            bool contains_white_cat = generated_text.find("white cat") != std::string::npos;
            EXPECT_TRUE(contains_white_cat) << "can't find 'white cat' in generated text: " << generated_text;
        }
    }

private:
    bool load_test_data_position_ids_list(
        std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>& position_ids_list) {
        ov::element::Type element_type = ov::element::i64;
        ov::Shape shape = {3, 1, 30};
        size_t byte_size = 720;
        bool has_rope_delta = true;
        int64_t rope_delta_value = -2;

        ov::Tensor tensor(element_type, shape);
        std::ifstream bin_file(_position_ids_path, std::ios::binary);
        EXPECT_TRUE(bin_file.is_open()) << "Failed to open position ids tensor file: " << _position_ids_path;
        bin_file.read(reinterpret_cast<char*>(tensor.data()), byte_size);
        bin_file.close();

        std::optional<int64_t> rope_delta = has_rope_delta ? std::optional<int64_t>(rope_delta_value) : std::nullopt;
        position_ids_list.emplace_back(std::move(tensor), rope_delta);

        return true;
    }

    bool load_test_data_input_embeds_list(std::vector<ov::Tensor>& input_embeds_list) {
        input_embeds_list.clear();

        ov::element::Type element_type = ov::element::f32;
        ov::Shape shape = {1, 30, 2048};
        size_t byte_size = 245760;

        ov::Tensor tensor(element_type, shape);
        std::ifstream bin_file(_embeds_path, std::ios::binary);
        EXPECT_TRUE(bin_file.is_open()) << "Failed to open embeds tensor file: " << _embeds_path;
        bin_file.read(reinterpret_cast<char*>(tensor.data()), byte_size);
        bin_file.close();

        input_embeds_list.push_back(std::move(tensor));

        return true;
    }
};

TEST_P(LLMInferenceModuleTest, ModuleTest) {
    run();
}

static std::vector<test_params> g_test_params = {
    {true, true, TEST_MODEL::get_device()},
    {true, false, TEST_MODEL::get_device()},
    {false, false, TEST_MODEL::get_device()},
};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         LLMInferenceModuleTest,
                         ::testing::ValuesIn(g_test_params),
                         LLMInferenceModuleTest::get_test_case_name);
