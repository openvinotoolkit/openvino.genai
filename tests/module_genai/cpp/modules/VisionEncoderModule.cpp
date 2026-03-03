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

struct Qwen3_5VisionEncoderTestData {
    ov::Tensor pixel_values;
    ov::Tensor grid_thw;
    ov::Tensor pos_embeds;
    ov::Tensor rotary_cos;
    ov::Tensor rotary_sin;
    ov::Tensor input_ids;
    ov::Tensor attention_mask;
};

namespace TEST_DATA {

Qwen3_5VisionEncoderTestData qwen3_5_vision_encoder_test_data() {
    Qwen3_5VisionEncoderTestData data;
    const size_t seed = 42;

    data.pixel_values = ov::genai::module::ModuleTestBase::ut_randn_tensor(ov::Shape{256, 3, 2, 16, 16}, seed);
    data.pos_embeds   = ov::genai::module::ModuleTestBase::ut_randn_tensor(ov::Shape{256, 1152}, seed);
    data.rotary_cos   = ov::genai::module::ModuleTestBase::ut_randn_tensor(ov::Shape{256, 72},   seed);
    data.rotary_sin   = ov::genai::module::ModuleTestBase::ut_randn_tensor(ov::Shape{256, 72},   seed);

    data.grid_thw = ov::Tensor(ov::element::i64, ov::Shape{1, 3});
    {
        int64_t* p = data.grid_thw.data<int64_t>();
        p[0] = 1; p[1] = 16; p[2] = 16;
    }

    data.input_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 78});
    {
        int64_t* p = data.input_ids.data<int64_t>();
        p[0] = 248045; p[1] = 846; p[2] = 198;
        p[3] = 248053;
        for (int i = 4; i < 68; ++i) p[i] = 248056;
        p[68] = 248054;
        int64_t text_tokens[] = {5606, 420, 6866, 198, 14524, 2534, 553, 445, 13};
        for (int i = 0; i < 9; ++i) p[69 + i] = text_tokens[i];
    }

    data.attention_mask = ov::Tensor(ov::element::i64, ov::Shape{1, 78});
    {
        int64_t* p = data.attention_mask.data<int64_t>();
        std::fill(p, p + 78, int64_t(1));
    }

    return data;
}

}

using qwen3_5_test_params = std::tuple<Qwen3_5VisionEncoderTestData, std::string>;

class Qwen3_5VisionEncoderModuleTest
    : public ov::genai::module::ModuleTestBase,
      public ::testing::TestWithParam<qwen3_5_test_params> {
private:
    std::string m_device;
    Qwen3_5VisionEncoderTestData m_test_data;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<qwen3_5_test_params>& obj) {
        return "Qwen3_5_" + std::get<1>(obj.param);
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_test_data, m_device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "qwen3_5";
        YAML::Node pipeline_modules = config["pipeline_modules"];

        std::string vision_encoder_name = "vision_encoder";
        {
            YAML::Node cur_node;
            cur_node["type"] = "VisionEncoderModule";
            cur_node["device"] = m_device;
            cur_node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            YAML::Node preprocessed_image;
            preprocessed_image["name"] = "preprocessed_image";
            preprocessed_image["type"] = "OVTensor";
            cur_node["inputs"].push_back(preprocessed_image);
            YAML::Node grid_thw;
            grid_thw["name"] = "grid_thw";
            grid_thw["type"] = "OVTensor";
            cur_node["inputs"].push_back(grid_thw);
            YAML::Node pos_embeds;
            pos_embeds["name"] = "pos_embeds";
            pos_embeds["type"] = "OVTensor";
            cur_node["inputs"].push_back(pos_embeds);
            YAML::Node rotary_cos;
            rotary_cos["name"] = "rotary_cos";
            rotary_cos["type"] = "OVTensor";
            cur_node["inputs"].push_back(rotary_cos);
            YAML::Node rotary_sin;
            rotary_sin["name"] = "rotary_sin";
            rotary_sin["type"] = "OVTensor";
            cur_node["inputs"].push_back(rotary_sin);
            YAML::Node input_ids;
            input_ids["name"] = "input_ids";
            input_ids["type"] = "OVTensor";
            cur_node["inputs"].push_back(input_ids);
            YAML::Node attention_mask;
            attention_mask["name"] = "attention_mask";
            attention_mask["type"] = "OVTensor";
            cur_node["inputs"].push_back(attention_mask);
            cur_node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["outputs"].push_back(output_node("image_embedding", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("visual_pos_mask", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("position_ids", to_string(DataType::OVTensor)));
            cur_node["outputs"].push_back(output_node("rope_delta", to_string(DataType::OVTensor)));
            cur_node["params"] = YAML::Node();
            cur_node["params"]["model_path"] = TEST_MODEL::Qwen3_5() + "qwen3_5_vision_q4a_b4a_g128.xml";
            cur_node["params"]["vision_start_token_id"] = 248053;
            pipeline_modules[vision_encoder_name] = cur_node;
        }
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["preprocessed_image"]   = m_test_data.pixel_values;
        inputs["grid_thw"]       = m_test_data.grid_thw;
        inputs["pos_embeds"]     = m_test_data.pos_embeds;
        inputs["rotary_cos"]     = m_test_data.rotary_cos;
        inputs["rotary_sin"]     = m_test_data.rotary_sin;
        inputs["input_ids"]      = m_test_data.input_ids;
        inputs["attention_mask"] = m_test_data.attention_mask;
        return inputs;
    }

    std::vector<float> expected_image_embedding = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    ov::Shape expected_image_embedding_shape = ov::Shape{1, 78, 2048};

    std::vector<bool> expected_visual_pos_mask = {
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };
    ov::Shape expected_visual_pos_mask_shape = ov::Shape{1, 78};

    std::vector<int64_t> expected_position_ids = {
        0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
    };
    ov::Shape expected_position_ids_shape = ov::Shape{3, 1, 78};

    std::vector<int64_t> expected_rope_delta = {
        -56
    };
    ov::Shape expected_rope_delta_shape = ov::Shape{1, 1};

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto image_embedding   = pipe.get_output("image_embedding").as<ov::Tensor>();
        auto visual_pos_mask = pipe.get_output("visual_pos_mask").as<ov::Tensor>();
        auto position_ids    = pipe.get_output("position_ids").as<ov::Tensor>();
        auto rope_delta     = pipe.get_output("rope_delta").as<ov::Tensor>();

        EXPECT_TRUE(compare_big_tensor(image_embedding, expected_image_embedding))
            << "image_embedding do not match expected values";
        EXPECT_TRUE(compare_shape(image_embedding.get_shape(), expected_image_embedding_shape))
            << "image_embedding shape mismatch: got " << image_embedding.get_shape();
        
        EXPECT_TRUE(compare_big_tensor(visual_pos_mask, expected_visual_pos_mask))
            << "visual_pos_mask do not match expected values";
        EXPECT_TRUE(compare_shape(visual_pos_mask.get_shape(), expected_visual_pos_mask_shape))
            << "visual_pos_mask shape mismatch: got " << visual_pos_mask.get_shape();
        
        EXPECT_TRUE(compare_big_tensor(position_ids, expected_position_ids))
            << "position_ids do not match expected values";
        EXPECT_TRUE(compare_shape(position_ids.get_shape(), expected_position_ids_shape))
            << "position_ids shape mismatch: got " << position_ids.get_shape();

        EXPECT_TRUE(compare_big_tensor(rope_delta, expected_rope_delta))
            << "rope_delta do not match expected values";
        EXPECT_TRUE(compare_shape(rope_delta.get_shape(), expected_rope_delta_shape))
            << "rope_delta shape mismatch: got " << rope_delta.get_shape();
    }
};

TEST_P(Qwen3_5VisionEncoderModuleTest, ModuleTest) {
    run();
}

namespace qwen3_5_vision_encoder_test {
    auto test_data    = std::vector<Qwen3_5VisionEncoderTestData>{TEST_DATA::qwen3_5_vision_encoder_test_data()};
    auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};
}

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         Qwen3_5VisionEncoderModuleTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(qwen3_5_vision_encoder_test::test_data),
                             ::testing::ValuesIn(qwen3_5_vision_encoder_test::test_devices)),
                         Qwen3_5VisionEncoderModuleTest::get_test_case_name);
