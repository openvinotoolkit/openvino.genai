// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/load_image.hpp"

using namespace ov::genai::module;

struct VisionEncoderTestData {
    ov::Tensor preprocessed_image;
    std::vector<int> source_size;
    std::vector<int> images_sequence;
    ov::Tensor input_ids;
};

namespace TEST_DATA {

VisionEncoderTestData vision_encoder_test_data() {
    VisionEncoderTestData data;
    data.preprocessed_image = ov::genai::module::ModuleTestBase::ut_randn_tensor(ov::Shape{64, 1280}, 42);
    data.source_size = {8, 8};
    data.images_sequence = {0};
    data.input_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 6});
    int64_t *input_ids_ptr = data.input_ids.data<int64_t>();
    std::vector<int64_t> input_id_values = {1986, 374, 264, 6077, 9934, 13};
    std::copy(input_id_values.begin(), input_id_values.end(), input_ids_ptr);
    return data;
}

}

using test_params = std::tuple<VisionEncoderTestData, std::string>;

class VisionEncoderModuleTest : public ov::genai::module::ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    VisionEncoderTestData m_test_data;
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
        config["global_context"]["model_type"] = "qwen2_5_vl";
        YAML::Node pipeline_modules = config["pipeline_modules"];
        YAML::Node vision_encoder;
        vision_encoder["type"] = "VisionEncoderModule";
        vision_encoder["device"] = m_device;
        YAML::Node inputs;
        YAML::Node preprocessed_image;
        preprocessed_image["name"] = "preprocessed_image";
        preprocessed_image["type"] = "OVTensor";
        inputs.push_back(preprocessed_image);
        YAML::Node source_size;
        source_size["name"] = "source_size";
        source_size["type"] = "VecInt";
        inputs.push_back(source_size);
        YAML::Node images_sequence;
        images_sequence["name"] = "images_sequence";
        images_sequence["type"] = "VecInt";
        inputs.push_back(images_sequence);
        YAML::Node input_ids;
        input_ids["name"] = "input_ids";
        input_ids["type"] = "OVTensor";
        inputs.push_back(input_ids);
        vision_encoder["inputs"] = inputs;
        YAML::Node outputs;
        YAML::Node image_embedding;
        image_embedding["name"] = "image_embedding";
        image_embedding["type"] = "OVTensor";
        outputs.push_back(image_embedding);
        YAML::Node position_ids;
        position_ids["name"] = "position_ids";
        position_ids["type"] = "OVTensor";
        outputs.push_back(position_ids);
        YAML::Node rope_delta;
        rope_delta["name"] = "rope_delta";
        rope_delta["type"] = "Int";
        outputs.push_back(rope_delta);
        vision_encoder["outputs"] = outputs;
        YAML::Node params;
        params["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        params["vision_start_token_id"] = 151652;
        vision_encoder["params"] = params;
        pipeline_modules["vision_encoder"] = vision_encoder;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["preprocessed_image"] = m_test_data.preprocessed_image;
        inputs["source_size"] = m_test_data.source_size;
        inputs["images_sequence"] = m_test_data.images_sequence;
        inputs["input_ids"] = m_test_data.input_ids;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto image_embedding = pipe.get_output("image_embedding").as<ov::Tensor>();
        auto position_ids = pipe.get_output("position_ids").as<ov::Tensor>();
        auto rope_delta = pipe.get_output("rope_delta").as<int>();

        const std::vector<float> expected_image_embedding = {
            -0.377955, -0.911449, 1.51156, 0.503151, -1.6234, 1.64198, 2.41668, 0.666558, -0.437834, 0.183663
        };

        const std::vector<float> expected_image_embedding_gpu = {
            -0.356445, -0.875488, 1.36426, 0.532715, -1.56934, 1.53516, 2.41016, 0.65625, -0.41626, 0.221802
        };

        const std::vector<int64_t> expected_position_id = {
            0, 1, 2, 3, 4, 5, 0, 1, 2, 3
        };
        const int expected_rope_delta = 0;

        if (m_device == "GPU") {
            EXPECT_TRUE(compare_big_tensor(image_embedding, expected_image_embedding_gpu, m_threshold))
                << "image_embedding does not match expected values";
        } else {
            EXPECT_TRUE(compare_big_tensor(image_embedding, expected_image_embedding, m_threshold))
                << "image_embedding does not match expected values";
        }
        EXPECT_TRUE(compare_shape(image_embedding.get_shape(), ov::Shape{16, 2048}))
            << "image_embedding's shape does not match expected shape";

        EXPECT_TRUE(compare_big_tensor(position_ids, expected_position_id))
            << "position_ids do not match expected values";
        EXPECT_TRUE(compare_shape(position_ids.get_shape(), ov::Shape{3, 1, 6}))
            << "position_ids's shape does not match expected shape";

        EXPECT_EQ(rope_delta, expected_rope_delta)
            << "rope_delta does not match expected value";
    }
};

TEST_P(VisionEncoderModuleTest, ModuleTest) {
    run();
}
namespace vision_encoder_test {

auto test_data = std::vector<VisionEncoderTestData> {TEST_DATA::vision_encoder_test_data()};
auto test_devices = std::vector<std::string> {TEST_MODEL::get_device()};

}

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite, 
                         VisionEncoderModuleTest,
                         ::testing::Combine(::testing::ValuesIn(vision_encoder_test::test_data),
                                            ::testing::ValuesIn(vision_encoder_test::test_devices)),
                         VisionEncoderModuleTest::get_test_case_name);
   