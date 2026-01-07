// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"

class EmbeddingMergerModuleTest : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(EmbeddingMergerModuleTest)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "qwen2_5_vl"
pipeline_modules:

  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "input_ids"
        type: "OVTensor"
      - name: "input_embedding"
        type: "OVTensor"
      - name: "image_embedding"
        type: "OVTensor"
      - name: "video_embedding"
        type: "OVTensor"

  embedding_merger:
    type: "EmbeddingMergerModule"
    device: "GPU"
    inputs:
      - name: "input_ids"
        type: "OVTensor"
        source: "pipeline_params.input_ids"
      - name: "input_embedding"
        type: "OVTensor"
        source: "pipeline_params.input_embedding"
      - name: "image_embedding"
        type: "OVTensor"
        source: "pipeline_params.image_embedding"
      - name: "video_embedding"
        type: "OVTensor"
        source: "pipeline_params.video_embedding"
    outputs:
      - name: "merged_embedding"
        type: "OVTensor"
    params:
      model_path: "./ut_pipelines/Qwen2.5-VL-3B-Instruct/INT4/"

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "merged_embedding"
        type: "OVTensor"
        source: "embedding_merger.merged_embedding"
)";
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

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("merged_embedding").as<ov::Tensor>();

        std::vector<float> expected_merged_embeds = { 
          0.37454, 0.796543, 0.950714, 0.183435, 0.731994, 0.779691, 0.598659, 0.59685, 0.156019, 0.445833
        };
        CHECK(compare_big_tensor<float>(output, expected_merged_embeds, 1e-2), "merged_embedding do not match expected values");
        CHECK(compare_shape(output.get_shape(), ov::Shape{1, 6, 2048}), "merged_embedding's shape not match expected shape");
    }
};

REGISTER_MODULE_TEST(EmbeddingMergerModuleTest);
