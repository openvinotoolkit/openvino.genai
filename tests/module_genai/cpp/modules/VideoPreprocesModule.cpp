// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/load_image.hpp"

struct ExpectedOutput {
    std::vector<float> pixel_values;
    ov::Shape pixel_values_shape;
    std::vector<int64_t> video_grid_thw;
    ov::Shape video_grid_thw_shape;
    std::vector<float> pos_embeds;
    ov::Shape pos_embeds_shape;
    std::vector<float> rotary_cos;
    ov::Shape rotary_cos_shape;
    std::vector<float> rotary_sin;
    ov::Shape rotary_sin_shape;
};

// Tuple: (is_single_video, device, pair<models_type, models_path, expected_output>)
using test_params = std::tuple<bool, std::string, std::tuple<std::string, std::string, ExpectedOutput>>;
using namespace ov::genai::module;

class VideoPreprocessModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;
    bool _is_single_video;
    std::string _models_path;
    std::string _models_type;
    ExpectedOutput _expected_output;

    bool is_single_video() const {
        return _is_single_video;
    }
    float _threshold = 1e-2;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get image paths and device from parameters
        const auto& is_single_video = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        const auto& models_tuple = std::get<2>(obj.param);

        std::string result;
        result += (is_single_video) ? "SingleVideo_" : "BatchVideo_";
        result += device;
        result += "_ModelType_" + sanitize_for_gtest(std::get<0>(models_tuple)); // models_type
        result += "_ModelPath_" + sanitize_for_gtest(get_last_component(std::get<1>(models_tuple))); // models_path
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tuple<std::string, std::string, ExpectedOutput> model;
        std::tie(_is_single_video, _device, model) = GetParam();
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

        YAML::Node video_preprocessor;
        video_preprocessor["type"] = "VideoPreprocessModule";
        video_preprocessor["device"] = _device;
        video_preprocessor["description"] = "Video preprocessing.";

        YAML::Node inputs;
        YAML::Node input_image;
        input_image["name"] = (is_single_video()) ? "video" : "videos";
        input_image["type"] = (is_single_video()) ? "OVTensor" : "VecOVTensor";
        inputs.push_back(input_image);
        video_preprocessor["inputs"] = inputs;

        YAML::Node outputs;
        YAML::Node pixel_values_videos;
        pixel_values_videos["name"] = "pixel_values_videos";
        pixel_values_videos["type"] = "OVTensor";
        outputs.push_back(pixel_values_videos);
        YAML::Node video_grid_thw;
        video_grid_thw["name"] = "video_grid_thw";
        video_grid_thw["type"] = "OVTensor";
        outputs.push_back(video_grid_thw);
        YAML::Node pos_embeds;
        pos_embeds["name"] = "pos_embeds";
        pos_embeds["type"] = "OVTensor";
        outputs.push_back(pos_embeds);
        YAML::Node rotary_cos;
        rotary_cos["name"] = "rotary_cos";
        rotary_cos["type"] = "OVTensor";
        outputs.push_back(rotary_cos);
        YAML::Node rotary_sin;
        rotary_sin["name"] = "rotary_sin";
        rotary_sin["type"] = "OVTensor";
        outputs.push_back(rotary_sin);
        video_preprocessor["outputs"] = outputs;
    
        YAML::Node model_path;
        model_path["model_path"] = _models_path;
        video_preprocessor["params"] = model_path;
        pipeline_modules["video_preprocessor"] = video_preprocessor;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        auto video = utils::create_countdown_frames();
        if (_is_single_video) {
            inputs["video"] = video;
            return inputs;
        } else {
            std::vector<ov::Tensor> videos = {video, video};
            inputs["videos"] = videos;
            return inputs;
        }
    }

    void check_output_tensor(const ov::Tensor& output, const std::vector<float>& expected, const ov::Shape& expected_shape, const std::string& tensor_name) {
        EXPECT_TRUE(compare_shape(output.get_shape(), expected_shape)) << tensor_name << " shape does not match expected shape.";
        EXPECT_TRUE(compare_big_tensor(output, expected, _threshold)) << tensor_name << " values do not match expected values.";
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto pixel_values_videos = pipe.get_output("pixel_values_videos").as<ov::Tensor>();
        check_output_tensor(pixel_values_videos, _expected_output.pixel_values, _expected_output.pixel_values_shape, "pixel_values_videos");
        
        auto video_grid_thw = pipe.get_output("video_grid_thw").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(video_grid_thw.get_shape(), _expected_output.video_grid_thw_shape)) << "video_grid_thw shape does not match expected shape.";
        EXPECT_TRUE(compare_big_tensor<int64_t>(video_grid_thw, _expected_output.video_grid_thw, _threshold)) << "video_grid_thw values do not match expected values.";

        auto pos_embeds = pipe.get_output("pos_embeds").as<ov::Tensor>();
        check_output_tensor(pos_embeds, _expected_output.pos_embeds, _expected_output.pos_embeds_shape, "pos_embeds");

        auto rotary_cos = pipe.get_output("rotary_cos").as<ov::Tensor>();
        check_output_tensor(rotary_cos, _expected_output.rotary_cos, _expected_output.rotary_cos_shape, "rotary_cos");

        auto rotary_sin = pipe.get_output("rotary_sin").as<ov::Tensor>();
        check_output_tensor(rotary_sin, _expected_output.rotary_sin, _expected_output.rotary_sin_shape, "rotary_sin");
    }
};

TEST_P(VideoPreprocessModuleTest, ModuleTest) {
    run();
}

namespace VideoPreprocessModuleTestParams {

auto test_video_types = std::vector<bool>{true};  // true: single video, false: batch video
auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};

ExpectedOutput qwen3_5_expected_output = {
    .pixel_values = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    .pixel_values_shape = {1056, 3, 2, 16, 16},
    .video_grid_thw = {3, 16, 22},
    .video_grid_thw_shape = {1, 3},
    .pos_embeds = {-0.0263672, -0.320312, 0.0454102, -0.0693359, -0.25, 0.0288086, 0.15332, 0.125977, 0.104004, 0.15625},
    .pos_embeds_shape = {1056, 768},
    .rotary_cos = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    .rotary_cos_shape = {1056, 64},
    .rotary_sin = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    .rotary_sin_shape = {1056, 64}};
// <models_type, models_path, expected_output>
std::vector<std::tuple<std::string, std::string, ExpectedOutput>> test_models = {{"qwen3_5", TEST_MODEL::Qwen3_5_0_8B(), qwen3_5_expected_output}};
}  // namespace VideoPreprocessModuleTestParams

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         VideoPreprocessModuleTest,
                         ::testing::Combine(::testing::ValuesIn(VideoPreprocessModuleTestParams::test_video_types),
                                            ::testing::ValuesIn(VideoPreprocessModuleTestParams::test_devices),
                                            ::testing::ValuesIn(VideoPreprocessModuleTestParams::test_models)),
                         VideoPreprocessModuleTest::get_test_case_name);
