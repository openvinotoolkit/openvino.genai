// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/load_image.hpp"
namespace AudioPreprocessModuleTest {
struct ExpectedOutput {
    std::pair<std::vector<float>, ov::Shape> input_features;
    std::pair<std::vector<int32_t>, ov::Shape> feature_attention_mask;
};

// Tuple: (is_single_audio, device, pair<models_type, models_path, expected_output>)
using test_params = std::tuple<bool, std::string, std::tuple<std::string, std::string, ExpectedOutput>>;
using namespace ov::genai::module;

class AudioPreprocessModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;
    bool _is_single_audio;
    std::string _models_path;
    std::string _models_type;
    ExpectedOutput _expected_output;

    bool is_single_audio() const {
        return _is_single_audio;
    }
    float _threshold = 1e-2;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get image paths and device from parameters
        const auto& is_single_audio = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        const auto& models_tuple = std::get<2>(obj.param);

        std::string result;
        result += (is_single_audio) ? "SingleAudio_" : "BatchAudio_";
        result += device;
        result += "_ModelType_" + sanitize_for_gtest(std::get<0>(models_tuple)); // models_type
        result += "_ModelPath_" + sanitize_for_gtest(get_last_component(std::get<1>(models_tuple))); // models_path
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tuple<std::string, std::string, ExpectedOutput> model;
        std::tie(_is_single_audio, _device, model) = GetParam();
        _models_type = std::get<0>(model);
        _models_path = std::get<1>(model);
        _expected_output = std::get<2>(model);
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = _models_type;

        YAML::Node pipeline_modules = config["pipeline_modules"];

        YAML::Node audio_preprocessor;
        audio_preprocessor["type"] = "AudioPreprocessModule";
        audio_preprocessor["device"] = _device;

        // Define inputs
        YAML::Node inputs;
        if (is_single_audio()) {
            inputs.push_back(input_node("audio", "OVTensor"));
        } else {
            inputs.push_back(input_node("audios", "VecOVTensor"));
        }
        audio_preprocessor["inputs"] = inputs;

        // Define outputs
        YAML::Node outputs;
        if (is_single_audio()) {
            outputs.push_back(output_node("input_features", "OVTensor"));
            outputs.push_back(output_node("feature_attention_mask", "OVTensor"));
        } else {
            outputs.push_back(output_node("input_features_vec", "VecOVTensor"));
            outputs.push_back(output_node("feature_attention_mask_vec", "VecOVTensor"));
        }
        audio_preprocessor["outputs"] = outputs;

        // Define parameters
        YAML::Node model_path;
        model_path["model_path"] = _models_path;
        audio_preprocessor["params"] = model_path;

        pipeline_modules["audio_preprocessor"] = audio_preprocessor;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        auto audio = TEST_DATA::audio_dummy_data(2.9013125f);
        if (_is_single_audio) {
            inputs["audio"] = audio;
            return inputs;
        } else {
            std::vector<ov::Tensor> audios = {audio, audio};
            inputs["audios"] = audios;
            return inputs;
        }
    }

    void check_output_tensor(const ov::Tensor& output, const std::vector<float>& expected, const ov::Shape& expected_shape, const std::string& tensor_name) {
        EXPECT_TRUE(compare_shape(output.get_shape(), expected_shape))
            << tensor_name << " shape does not match expected shape: " << expected_shape
            << ", got: " << output.get_shape();
        EXPECT_TRUE(compare_big_tensor(output, expected, _threshold)) << tensor_name << " values do not match expected values.";
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto input_features = pipe.get_output("input_features").as<ov::Tensor>();
        check_output_tensor(input_features, _expected_output.input_features.first, _expected_output.input_features.second, "input_features");
        
        auto feature_attention_mask = pipe.get_output("feature_attention_mask").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(feature_attention_mask.get_shape(), _expected_output.feature_attention_mask.second)) << "feature_attention_mask shape does not match expected shape.";
    }
};

TEST_P(AudioPreprocessModuleTest, ModuleTest) {
    run();
}

auto test_audio_types = std::vector<bool>{true};  // true: single audio, false: batch audio
auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};

ExpectedOutput qwen3_omni_expected_output = {
    /*input_features={data, shape}*/ {std::vector<float>{-0.171524f,
                                                         0.324615f,
                                                         0.451607f,
                                                         0.534605f,
                                                         0.595251f,
                                                         0.642851f,
                                                         0.681975f,
                                                         0.715166f,
                                                         0.743978f,
                                                         0.769428f},
                                      ov::Shape{128, 290}},
    /*feature_attention_mask={data, shape}*/ {std::vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ov::Shape{290}}};

std::vector<std::tuple<std::string, std::string, ExpectedOutput>> test_models = {{"qwen3_omni", TEST_MODEL::Qwen3_Omni_4B_Instruct_Multilingual(), qwen3_omni_expected_output}};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         AudioPreprocessModuleTest,
                         ::testing::Combine(::testing::ValuesIn(test_audio_types),
                                            ::testing::ValuesIn(test_devices),
                                            ::testing::ValuesIn(test_models)),
                         AudioPreprocessModuleTest::get_test_case_name);

}  // namespace AudioPreprocessModuleTest