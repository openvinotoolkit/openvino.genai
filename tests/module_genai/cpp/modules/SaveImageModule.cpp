// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include <filesystem>

using namespace ov::genai::module;

// ============================================================================
// Test Data Structure
// ============================================================================

struct SaveImageTestData {
    std::string name;                    // Test case name
    ov::element::Type element_type;      // Tensor element type (u8, f32)
    ov::Shape shape;                     // Tensor shape (NHWC or NCHW)
    std::string filename_prefix;         // Output filename prefix
    size_t expected_file_count;          // Expected number of output files
    
    // Helper to create test tensor
    ov::Tensor create_tensor() const {
        ov::Tensor tensor(element_type, shape);
        
        if (element_type == ov::element::u8) {
            fill_u8_tensor(tensor);
        } else if (element_type == ov::element::f32) {
            fill_f32_tensor(tensor);
        }
        return tensor;
    }
    
private:
    void fill_u8_tensor(ov::Tensor& tensor) const {
        uint8_t* data = tensor.data<uint8_t>();
        size_t total_size = tensor.get_size();
        
        // Fill with gradient pattern
        for (size_t i = 0; i < total_size; i++) {
            data[i] = static_cast<uint8_t>((i * 7) % 256);
        }
    }
    
    void fill_f32_tensor(ov::Tensor& tensor) const {
        float* data = tensor.data<float>();
        size_t total_size = tensor.get_size();
        
        // Fill with normalized gradient pattern (0-1 range)
        for (size_t i = 0; i < total_size; i++) {
            data[i] = static_cast<float>((i % 100)) / 100.0f;
        }
    }
};

// ============================================================================
// Test Data Definitions
// ============================================================================

namespace TEST_DATA {

SaveImageTestData basic_u8_nhwc() {
    return {
        "Basic_U8_NHWC",
        ov::element::u8,
        ov::Shape{1, 64, 64, 3},  // NHWC format
        "test_basic",
        1
    };
}

SaveImageTestData float32_nhwc() {
    return {
        "Float32_NHWC",
        ov::element::f32,
        ov::Shape{1, 32, 32, 3},  // NHWC format
        "test_float",
        1
    };
}

SaveImageTestData u8_nchw() {
    return {
        "U8_NCHW",
        ov::element::u8,
        ov::Shape{1, 3, 48, 48},  // NCHW format
        "test_nchw",
        1
    };
}

SaveImageTestData batch_u8_nhwc() {
    return {
        "Batch_U8_NHWC",
        ov::element::u8,
        ov::Shape{3, 24, 24, 3},  // Batch of 3 images in NHWC format
        "test_batch",
        3
    };
}

}  // namespace TEST_DATA

// ============================================================================
// Test Parameters
// ============================================================================

// Test parameters:
// <0> SaveImageTestData - test data structure containing tensor info and expected results
// <1> std::string - device name (e.g., "CPU", "GPU")
using test_params = std::tuple<SaveImageTestData, std::string>;

// ============================================================================
// Test Fixture
// ============================================================================

class SaveImageModuleTest : public ov::genai::module::ModuleTestBase, 
                            public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    SaveImageTestData m_test_data;
    std::string m_output_folder = "./test_output";

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& test_data = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        return test_data.name + "_" + device;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_test_data, m_device) = GetParam();
        
        // Clean up output folder before test
        if (std::filesystem::exists(m_output_folder)) {
            std::filesystem::remove_all(m_output_folder);
        }
    }

    void TearDown() override {
        // Clean up output folder after test
        if (std::filesystem::exists(m_output_folder)) {
            std::filesystem::remove_all(m_output_folder);
        }
    }

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "zimage";
        
        YAML::Node pipeline_modules = config["pipeline_modules"];
        
        // ParameterModule
        YAML::Node params_module;
        params_module["type"] = "ParameterModule";
        YAML::Node param_outputs;
        YAML::Node image_data_output;
        image_data_output["name"] = "image_data";
        image_data_output["type"] = "OVTensor";
        param_outputs.push_back(image_data_output);
        params_module["outputs"] = param_outputs;
        pipeline_modules["pipeline_params"] = params_module;
        
        // SaveImageModule
        YAML::Node save_module;
        save_module["type"] = "SaveImageModule";
        save_module["device"] = m_device;
        
        YAML::Node save_inputs;
        YAML::Node raw_data_input;
        raw_data_input["name"] = "raw_data";
        raw_data_input["type"] = "OVTensor";
        raw_data_input["source"] = "pipeline_params.image_data";
        save_inputs.push_back(raw_data_input);
        save_module["inputs"] = save_inputs;
        
        YAML::Node save_outputs;
        YAML::Node saved_image_output;
        saved_image_output["name"] = "saved_image";
        saved_image_output["type"] = "String";
        save_outputs.push_back(saved_image_output);
        YAML::Node saved_images_output;
        saved_images_output["name"] = "saved_images";
        saved_images_output["type"] = "VecString";
        save_outputs.push_back(saved_images_output);
        save_module["outputs"] = save_outputs;
        
        YAML::Node save_params;
        save_params["filename_prefix"] = m_test_data.filename_prefix;
        save_params["output_folder"] = m_output_folder;
        save_module["params"] = save_params;
        
        pipeline_modules["save_image"] = save_module;
        
        // ResultModule
        YAML::Node result_module;
        result_module["type"] = "ResultModule";
        result_module["device"] = "CPU";
        
        YAML::Node result_inputs;
        YAML::Node result_saved_image;
        result_saved_image["name"] = "saved_image";
        result_saved_image["type"] = "String";
        result_saved_image["source"] = "save_image.saved_image";
        result_inputs.push_back(result_saved_image);
        YAML::Node result_saved_images;
        result_saved_images["name"] = "saved_images";
        result_saved_images["type"] = "VecString";
        result_saved_images["source"] = "save_image.saved_images";
        result_inputs.push_back(result_saved_images);
        result_module["inputs"] = result_inputs;
        
        pipeline_modules["pipeline_results"] = result_module;
        
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["image_data"] = m_test_data.create_tensor();
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        // Verify saved_image output
        auto image_path = pipe.get_output("saved_image").as<std::string>();
        EXPECT_FALSE(image_path.empty()) << "saved_image output should not be empty";
        EXPECT_TRUE(std::filesystem::exists(image_path)) 
            << "Output image file should exist: " << image_path;
        EXPECT_TRUE(image_path.find(m_test_data.filename_prefix) != std::string::npos) 
            << "Image path should contain prefix '" << m_test_data.filename_prefix << "'";
        EXPECT_TRUE(image_path.find(".bmp") != std::string::npos) 
            << "Image path should have .bmp extension";
        
        // Verify saved_images output (vector of paths)
        auto images = pipe.get_output("saved_images").as<std::vector<std::string>>();
        EXPECT_EQ(images.size(), m_test_data.expected_file_count) 
            << "saved_images should contain " << m_test_data.expected_file_count << " file path(s)";
        
        // Verify all files exist
        for (const auto& filepath : images) {
            EXPECT_TRUE(std::filesystem::exists(filepath)) 
                << "Output image file should exist: " << filepath;
            EXPECT_TRUE(filepath.find(m_test_data.filename_prefix) != std::string::npos)
                << "Image path should contain prefix '" << m_test_data.filename_prefix << "'";
        }
        
        // For single image, verify saved_images[0] matches saved_image
        if (m_test_data.expected_file_count == 1) {
            EXPECT_EQ(images[0], image_path) 
                << "saved_images[0] should match saved_image output";
        }
    }
};

// ============================================================================
// Test Case
// ============================================================================

TEST_P(SaveImageModuleTest, ModuleTest) {
    run();
}

// ============================================================================
// Test Data and Device Configuration
// ============================================================================

namespace save_image_test {

auto test_data = std::vector<SaveImageTestData> {
    TEST_DATA::basic_u8_nhwc(),
    TEST_DATA::float32_nhwc(),
    TEST_DATA::u8_nchw(),
    TEST_DATA::batch_u8_nhwc()
};

auto test_devices = std::vector<std::string> {TEST_MODEL::get_device()};

}  // namespace save_image_test

// ============================================================================
// Test Suite Instantiation
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    ModuleTestSuite,
    SaveImageModuleTest,
    ::testing::Combine(
        ::testing::ValuesIn(save_image_test::test_data),
        ::testing::ValuesIn(save_image_test::test_devices)
    ),
    SaveImageModuleTest::get_test_case_name
);
