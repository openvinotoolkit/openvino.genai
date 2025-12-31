// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class LlmInferenceModuleTest : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(LlmInferenceModuleTest)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "qwen2_5_vl"
pipeline_modules:

  llm_inference:
    type: "LLMInferenceModule"
    description: "LLM module for Continuous Batch pipeline"
    device: "CPU"
    inputs:
      - name: "embeds"
        type: "OVTensor"
        source: "pipeline_params.embeds"
      - name: "position_ids"
        type: "VecOVTensor"
        source: "pipeline_params.position_ids"
      - name: "rope_delta"
        type: "Int"
        source: "pipeline_params.rope_delta"
    outputs:
      - name: "generated_text"
        type: "String"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"
      max_new_tokens: "16"
      do_sample: "false"
      top_p: "1.0"
      top_k: "50"
      temperature: "1.0"
      repetition_penalty: "1.0"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        std::vector<ov::Tensor> input_embeds_list;
        load_test_data_input_embeds_list(input_embeds_list);
        CHECK(input_embeds_list.size(), "Failed to load input embeds list data");
        inputs["embeds"] = input_embeds_list[0];

        std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> input_position_ids_list;
        load_test_data_position_ids_list(input_position_ids_list);
        CHECK(input_position_ids_list.size(), "Failed to load position ids list data");
        std::vector<ov::Tensor> only_position_ids_list;
        std::vector<int> rope_delta_list;
        for (auto& pids : input_position_ids_list) {
            only_position_ids_list.push_back(pids.first);
            rope_delta_list.push_back(pids.second.has_value() ? pids.second.value() : 0);
        }
        inputs["position_ids"] = only_position_ids_list[0];
        inputs["rope_delta"] = rope_delta_list[0];
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto generated_text = pipe.get_output("generated_text").as<std::string>();

        bool contains_white_cat = generated_text.find("white cat") != std::string::npos;
        CHECK(contains_white_cat, "llm inference module does not work as expected");
    }

    bool load_test_data_position_ids_list(
    	std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>& position_ids_list) {
        ov::element::Type element_type = ov::element::i64;
        ov::Shape shape = {3, 1, 30};
        size_t byte_size = 720;
        bool has_rope_delta = true;
        int64_t rope_delta_value = -2;

        ov::Tensor tensor(element_type, shape);
        std::string bin_path = "ut_test_data/position_ids_0.bin";
        std::ifstream bin_file(bin_path, std::ios::binary);
        if (bin_file.is_open()) {
            bin_file.read(reinterpret_cast<char*>(tensor.data()), byte_size);
            bin_file.close();
        } else {
            return false;
        }

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
        std::string bin_path = "ut_test_data/embeds_tensor_0.bin";
        std::ifstream bin_file(bin_path, std::ios::binary);
        if (bin_file.is_open()) {
            bin_file.read(reinterpret_cast<char*>(tensor.data()), byte_size);
            bin_file.close();
        } else {
            return false;
        }
        input_embeds_list.push_back(std::move(tensor));

        return true;
    }
};


class LlmInferenceModuleTest_Batch : public LlmInferenceModuleTest {
public:
    LlmInferenceModuleTest_Batch(const std::string& test_name) : LlmInferenceModuleTest(test_name) {}

protected:
    virtual std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "qwen2_5_vl"
pipeline_modules:

  llm_inference:
    type: "LLMInferenceModule"
    description: "LLM module for Continuous Batch pipeline"
    device: "CPU"
    inputs:
      - name: "embeds_list"
        type: "VecOVTensor"
        source: "pipeline_params.embeds_list"
      - name: "position_ids_list"
        type: "VecOVTensor"
        source: "pipeline_params.position_ids_list"
      - name: "rope_delta_list"
        type: "VecInt"
        source: "pipeline_params.rope_delta_list"
    outputs:
      - name: "generated_texts"
        type: "VecString"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"
      max_new_tokens: "16"
      do_sample: "false"
      top_p: "1.0"
      top_k: "50"
      temperature: "1.0"
      repetition_penalty: "1.0"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        std::vector<ov::Tensor> input_embeds_list;
        load_test_data_input_embeds_list(input_embeds_list);
        CHECK(input_embeds_list.size(), "Failed to load input embeds list data");
        inputs["embeds_list"] = input_embeds_list;

        std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> input_position_ids_list;
        load_test_data_position_ids_list(input_position_ids_list);
        CHECK(input_position_ids_list.size(), "Failed to load position ids list data");

        std::vector<ov::Tensor> only_position_ids_list;
        std::vector<int> rope_delta_list;
        for (auto& pids : input_position_ids_list) {
            only_position_ids_list.push_back(pids.first);
            rope_delta_list.push_back(pids.second.has_value() ? pids.second.value() : 0);
        }

        inputs["position_ids_list"] = only_position_ids_list;
        inputs["rope_delta_list"] = rope_delta_list;

        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto generated_texts = pipe.get_output("generated_texts").as<std::vector<std::string>>();

        bool contains_white_cat = generated_texts[0].find("white cat") != std::string::npos;
        CHECK(contains_white_cat, "llm inference module does not work as expected");
    }
};

REGISTER_MODULE_TEST(LlmInferenceModuleTest);
REGISTER_MODULE_TEST(LlmInferenceModuleTest_Batch);
