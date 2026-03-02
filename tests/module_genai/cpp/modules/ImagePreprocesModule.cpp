// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/load_image.hpp"

// add device param to test_params
using test_params = std::tuple<std::vector<std::string>, std::string>;
using namespace ov::genai::module;

class ImagePreprocesModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;
    std::vector<std::string> _image_paths;
    bool is_single_image() const {
        return _image_paths.size() == 1u;
    }
    float _threshold = 1e-2;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get image paths and device from parameters
        const auto& paths = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        std::string result;
        for (size_t i = 0; i < paths.size(); ++i) {
            std::filesystem::path p(paths[i]);
            result += p.stem().string();
            if (i < paths.size() - 1)
                result += "_";
        }
        result += "_" + device;
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(_image_paths, _device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "qwen2_5_vl";

        YAML::Node pipeline_modules = config["pipeline_modules"];

        YAML::Node image_preprocessor;
        image_preprocessor["type"] = "ImagePreprocessModule";
        image_preprocessor["device"] = _device;
        image_preprocessor["description"] = "Image or Video preprocessing."; 
        YAML::Node inputs;
        YAML::Node input_image;
        input_image["name"] = (is_single_image()) ? "image" : "images";
        input_image["type"] = "OVTensor";
        inputs.push_back(input_image);
        image_preprocessor["inputs"] = inputs;
        YAML::Node outputs;
        YAML::Node output_raw_data;
        output_raw_data["name"] = (is_single_image()) ? "raw_data" : "raw_datas";
        output_raw_data["type"] = (is_single_image()) ? "OVTensor" : "VecOVTensor";
        outputs.push_back(output_raw_data);
        YAML::Node output_source_size;
        output_source_size["name"] = (is_single_image()) ? "source_size" : "source_sizes";
        output_source_size["type"] = (is_single_image()) ? "VecInt" : "VecVecInt";
        outputs.push_back(output_source_size);
        image_preprocessor["outputs"] = outputs;
        YAML::Node model_path;
        model_path["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        image_preprocessor["params"] = model_path;
        pipeline_modules["image_preprocessor"] = image_preprocessor;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        if (_image_paths.size() == 1u) {
            auto img1 = utils::load_image(_image_paths[0]);
            EXPECT_TRUE(img1) << "Failed to load test image: " + _image_paths[0];
            if (!img1) return inputs;
            inputs["image"] = img1;
        } else if (_image_paths.size() > 1u) {
            std::vector<ov::Tensor> batch_images;
            for (const auto& image_path : _image_paths) {
                auto img = utils::load_image(image_path);
                EXPECT_TRUE(img) << "Failed to load test image: " + image_path;
                if (!img) return inputs;
                batch_images.push_back(img);
            }
            inputs["images"] = batch_images;
        } else {
            EXPECT_TRUE(false) << "No image paths provided for testing.";
        }
        return inputs;
    }

    const std::vector<float> expected_input_ids_for_input_1 =
        {-0.0712891, 0.251953, 0.0825195, 0.078125, 0.122559, 0.0986328, 0.0844727, -0.0932617, 0.130859, -0.0274658};
    const std::vector<float> expected_input_ids_for_input_2 =
        {-0.341116, 0.223862, -0.00889927, 0.0725508, 0.39603, 0.0243148, 0.151016, -0.21681, 0.289946, -0.170177};

    void check_output_input_1(ov::genai::module::ModulePipeline& pipe) {
        auto raw_data = pipe.get_output("raw_data").as<ov::Tensor>();
        EXPECT_TRUE(compare_big_tensor(raw_data, expected_input_ids_for_input_1, _threshold))
            << "raw_data do not match expected values";
        EXPECT_TRUE(compare_shape(raw_data.get_shape(), ov::Shape{64, 1280}))
            << "raw_data's shape not match expected shape";

        auto source_size = pipe.get_output("source_size").as<std::vector<int>>();
        auto expected_source_size = std::vector<int>{8, 8};
        EXPECT_TRUE(source_size == expected_source_size) << "source_size not match expected values";
    }

    void check_output_input_2(ov::genai::module::ModulePipeline& pipe) {
        auto raw_datas = pipe.get_output("raw_datas").as<std::vector<ov::Tensor>>();

        std::vector<std::vector<float>> expected_input_ids = {expected_input_ids_for_input_1,
                                                              expected_input_ids_for_input_2};
        EXPECT_EQ(raw_datas.size(), expected_input_ids.size()) << "Number of raw_datas does not match expected";

        for (size_t i = 0; i < raw_datas.size(); ++i) {
            EXPECT_TRUE(compare_big_tensor(raw_datas[i], expected_input_ids[i], _threshold))
                << "raw_data do not match expected values";
            EXPECT_TRUE(compare_shape(raw_datas[i].get_shape(), ov::Shape{64, 1280}))
                << "raw_data's shape not match expected shape";
        }

        auto source_sizes = pipe.get_output("source_sizes").as<std::vector<std::vector<int>>>();
        std::vector<std::vector<int>> expected_source_sizes = {std::vector<int>{8, 8}, std::vector<int>{8, 8}};
        EXPECT_EQ(source_sizes.size(), expected_source_sizes.size()) << "Number of source_sizes does not match expected";
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        if (is_single_image()) {
            check_output_input_1(pipe);
        } else {
            check_output_input_2(pipe);
        }
    }
};

TEST_P(ImagePreprocesModuleTest, ModuleTest) {
    run();
}

auto test_image_1 = std::vector<std::string>{TEST_DATA::img_cat_120_100()};
auto test_image_2 = std::vector<std::string>{TEST_DATA::img_cat_120_100(), TEST_DATA::img_dog_120_120()};

auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         ImagePreprocesModuleTest,
                         ::testing::Combine(::testing::Values(test_image_1, test_image_2),
                                            ::testing::ValuesIn(test_devices)),
                         ImagePreprocesModuleTest::get_test_case_name);

class Qwen3_5ImagePreprocessModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;
    std::vector<std::string> _image_paths;
    bool is_single_image() const {
        return _image_paths.size() == 1u;
    }
    float _threshold = 1e-2;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get image paths and device from parameters
        const auto& paths = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        std::string result = "Qwen3_5_";
        for (size_t i = 0; i < paths.size(); ++i) {
            std::filesystem::path p(paths[i]);
            result += p.stem().string();
            if (i < paths.size() - 1)
                result += "_";
        }
        result += "_" + device;
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(_image_paths, _device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "qwen3_5";

        YAML::Node pipeline_modules = config["pipeline_modules"];

        YAML::Node image_preprocessor;
        image_preprocessor["type"] = "ImagePreprocessModule";
        image_preprocessor["device"] = _device;
        image_preprocessor["description"] = "Image or Video preprocessing.";
        YAML::Node inputs;
        YAML::Node input_image;
        input_image["name"] = (is_single_image()) ? "image" : "images";
        input_image["type"] = "OVTensor";
        inputs.push_back(input_image);
        image_preprocessor["inputs"] = inputs;
        YAML::Node outputs;

        YAML::Node pixel_values;
        pixel_values["name"] = "pixel_values";
        pixel_values["type"] = "OVTensor";
        outputs.push_back(pixel_values);
        YAML::Node grid_thw;
        grid_thw["name"] = "grid_thw";
        grid_thw["type"] = "OVTensor";
        outputs.push_back(grid_thw);
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

        image_preprocessor["outputs"] = outputs;
        YAML::Node model_path;
        model_path["model_path"] = TEST_MODEL::Qwen3_5();
        image_preprocessor["params"] = model_path;
        pipeline_modules["image_preprocessor"] = image_preprocessor;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        auto img1 = utils::load_image(_image_paths[0]);
        EXPECT_TRUE(img1) << "Failed to load test image: " + _image_paths[0];
        if (!img1) return inputs;
        inputs["image"] = img1;
        return inputs;
    }

    const std::vector<float> expected_pixel_values = {
        0.968627, 0.968627, 0.968627, 0.968627, 0.968627, 0.968015, 0.964338, 0.960539, 0.953186, 0.945833, 0.935172, 0.924142, 0.913113, 0.902083, 0.891054, 0.880025, 0.968627, 0.968627, 0.968627, 0.968403
    };
    const ov::Shape expected_pixel_values_shape = {256, 3, 2, 16, 16};

    const std::vector<int64_t> expected_grid_thw = {
        1, 16, 16
    };
    const ov::Shape expected_grid_thw_shape = {1, 3};

    const std::vector<float> expected_pos_embeds = {
        0.012573, 0.046143, -0.080566, 0.085938, -0.155273, 0.015869, -0.026367, 0.055908, 0.036377, 0.308594, -0.073242, -0.038818, -0.030273, -0.094727, 0.200195, 0.069336, -0.347656, 0.126953, 0.063965, 0.042480
    };
    const ov::Shape expected_pos_embeds_shape = {256, 1152};

    const std::vector<float> expected_rotary_cos = {
        1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000
    };
    const ov::Shape expected_rotary_cos_shape = {256, 72};

    const std::vector<float> expected_rotary_sin = {
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    };
    const ov::Shape expected_rotary_sin_shape = {256, 72};

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto pixel_values = pipe.get_output("pixel_values").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(pixel_values.get_shape(), expected_pixel_values_shape))
            << "pixel_values's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(pixel_values, expected_pixel_values, _threshold))
            << "pixel_values do not match expected values";

        auto grid_thw = pipe.get_output("grid_thw").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(grid_thw.get_shape(), expected_grid_thw_shape))
            << "grid_thw's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor<int64_t>(grid_thw, expected_grid_thw, _threshold))
            << "grid_thw do not match expected values";

        auto pos_embeds = pipe.get_output("pos_embeds").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(pos_embeds.get_shape(), expected_pos_embeds_shape))
            << "pos_embeds's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(pos_embeds, expected_pos_embeds, _threshold))
            << "pos_embeds do not match expected values";

        auto rotary_cos = pipe.get_output("rotary_cos").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(rotary_cos.get_shape(), expected_rotary_cos_shape))
            << "rotary_cos's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(rotary_cos, expected_rotary_cos, _threshold))
            << "rotary_cos do not match expected values";

        auto rotary_sin = pipe.get_output("rotary_sin").as<ov::Tensor>();
        EXPECT_TRUE(compare_shape(rotary_sin.get_shape(), expected_rotary_sin_shape))
            << "rotary_sin's shape not match expected shape";
        EXPECT_TRUE(compare_big_tensor(rotary_sin, expected_rotary_sin, _threshold))
            << "rotary_sin do not match expected values";
    }
};

TEST_P(Qwen3_5ImagePreprocessModuleTest, ModuleTest) {
    run();
}

auto test_image_3 = std::vector<std::string>{TEST_DATA::img_dog_120_120()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         Qwen3_5ImagePreprocessModuleTest,
                         ::testing::Combine(::testing::Values(test_image_3),
                                            ::testing::ValuesIn(test_devices)),
                         Qwen3_5ImagePreprocessModuleTest::get_test_case_name);
