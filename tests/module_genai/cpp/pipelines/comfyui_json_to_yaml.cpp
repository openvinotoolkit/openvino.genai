// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ComfyUIJsonConverter.cpp
 * @brief GTest-based parameterized tests for ModulePipeline::comfyui_json_to_yaml
 *        and ModulePipeline::comfyui_json_string_to_yaml
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>

#include "openvino/genai/module_genai/pipeline.hpp"
#include "utils/utils.hpp"
#include "utils/model_yaml.hpp"

namespace fs = std::filesystem;
using namespace ov::genai::module;

// ============================================================================
// Parameterized Test Structure
// ============================================================================

struct ConverterTestCase {
    std::string name;               // Test case name
    std::string json_filepath;      // Full path to JSON file (use get_json_file_path())
    bool expected_success;          // Expected conversion result (true = success)
    std::string model_path;         // Model path to use
    std::string device;             // Device to use
    std::string yaml_keyword;       // Keyword expected in YAML (empty = don't check)
    bool check_extracted_inputs;    // Whether to check extracted inputs
};

class ComfyUIJsonConverterParamTest : public ::testing::TestWithParam<ConverterTestCase> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<ConverterTestCase>& info) {
        return info.param.name;
    }
};

// ============================================================================
// Test Cases
// ============================================================================

namespace {

// Test case definitions for ComfyUI JSON converter parameterized tests.
// Using anonymous namespace to ensure internal linkage and avoid ODR violations.
const ConverterTestCase kConverterTestCases[] = {
    // Successful conversions - API JSON format
    {
        "ApiJson_ZImageTurbo2kTiled",
        get_test_file_path("z_image_turbo_2k_tiled_api.json"),
        true,                   // expected_success
        TEST_MODEL::ZImage_Turbo_fp16_ov(),  // model_path
        "CPU",                  // device
        "pipeline_modules",     // yaml_keyword
        true                    // check_extracted_inputs
    },
    // Successful conversions - Workflow JSON format
    {
        "WorkflowJson_ZImageTurbo2kTiled",
        get_test_file_path("z_image_turbo_2k_tiled.json"),
        true,
        TEST_MODEL::ZImage_Turbo_fp16_ov(),
        "GPU",
        "pipeline_modules",
        true
    },
    // Custom model path and device
    {
        "CustomModelPathAndDevice",
        get_test_file_path("z_image_turbo_2k_tiled_api.json"),
        true,
        "/custom/model/path/",
        "GPU.0",
        "/custom/model/path/",
        true
    },
    // Unsupported/Failed conversions - Video API JSON
    {
        "UnsupportedVideoApiJson",
        get_test_file_path("video_humo_api.json"),
        false,                  // expected_success = false
        TEST_MODEL::ZImage_Turbo_fp16_ov(),
        "CPU",
        "",                     // yaml_keyword (don't check)
        false                   // check_extracted_inputs
    },
    // Unsupported/Failed conversions - Video Workflow JSON
    {
        "UnsupportedVideoWorkflowJson",
        get_test_file_path("video_humo_workflow.json"),
        false,
        TEST_MODEL::ZImage_Turbo_fp16_ov(),
        "CPU",
        "",
        false
    }
};

}  // anonymous namespace

// ============================================================================
// File Path API Tests (comfyui_json_to_yaml)
// ============================================================================

TEST_P(ComfyUIJsonConverterParamTest, ConvertJsonFileToYaml) {
    const auto& tc = GetParam();

    fs::path json_file = tc.json_filepath;

    // Skip if test file doesn't exist
    if (!fs::exists(json_file)) {
        GTEST_SKIP() << "Test file not found: " << json_file;
    }

    ov::AnyMap pipeline_inputs;
    pipeline_inputs["model_path"] = tc.model_path;
    pipeline_inputs["device"] = tc.device;

    std::string yaml_content = ModulePipeline::comfyui_json_to_yaml(
        json_file, pipeline_inputs);

    if (tc.expected_success) {
        // Should succeed
        ASSERT_FALSE(yaml_content.empty())
            << "Conversion should succeed for: " << tc.name;

        // Check YAML keyword if specified
        if (!tc.yaml_keyword.empty()) {
            EXPECT_NE(yaml_content.find(tc.yaml_keyword), std::string::npos)
                << "YAML should contain '" << tc.yaml_keyword << "' for: " << tc.name;
        }

        // Check device was applied
        EXPECT_NE(yaml_content.find(tc.device), std::string::npos)
            << "YAML should contain device '" << tc.device << "' for: " << tc.name;

        // Check extracted inputs if requested
        if (tc.check_extracted_inputs) {
            EXPECT_TRUE(pipeline_inputs.count("prompt") > 0 ||
                        pipeline_inputs.count("width") > 0 ||
                        pipeline_inputs.count("height") > 0)
                << "Should extract at least some inputs for: " << tc.name;
        }

        // Validate the generated YAML
        auto validation_result = ModulePipeline::validate_config_string(yaml_content);
        EXPECT_TRUE(validation_result.valid)
            << "Generated YAML should pass validation for: " << tc.name;
    } else {
        // Should fail
        EXPECT_TRUE(yaml_content.empty())
            << "Conversion should fail for: " << tc.name;
    }
}

// ============================================================================
// String API Tests (comfyui_json_string_to_yaml)
// ============================================================================

TEST_P(ComfyUIJsonConverterParamTest, ConvertJsonStringToYaml) {
    const auto& tc = GetParam();

    fs::path json_file = tc.json_filepath;

    // Skip if test file doesn't exist
    if (!fs::exists(json_file)) {
        GTEST_SKIP() << "Test file not found: " << json_file;
    }

    std::string json_content;
    if (!readFileToString(json_file.string(), json_content)) {
        FAIL() << "Failed to read JSON file: " << json_file;
    }

    ov::AnyMap pipeline_inputs;
    pipeline_inputs["model_path"] = tc.model_path;
    pipeline_inputs["device"] = tc.device;

    std::string yaml_content = ModulePipeline::comfyui_json_string_to_yaml(
        json_content, pipeline_inputs);

    if (tc.expected_success) {
        // Should succeed
        ASSERT_FALSE(yaml_content.empty())
            << "String conversion should succeed for: " << tc.name;

        // Check YAML keyword if specified
        if (!tc.yaml_keyword.empty()) {
            EXPECT_NE(yaml_content.find(tc.yaml_keyword), std::string::npos)
                << "YAML should contain '" << tc.yaml_keyword << "' for: " << tc.name;
        }

        // Check device was applied
        EXPECT_NE(yaml_content.find(tc.device), std::string::npos)
            << "YAML should contain device '" << tc.device << "' for: " << tc.name;

        // Check extracted inputs if requested
        if (tc.check_extracted_inputs) {
            EXPECT_TRUE(pipeline_inputs.count("prompt") > 0 ||
                        pipeline_inputs.count("width") > 0 ||
                        pipeline_inputs.count("height") > 0)
                << "Should extract at least some inputs for: " << tc.name;
        }
    } else {
        // Should fail
        EXPECT_TRUE(yaml_content.empty())
            << "String conversion should fail for: " << tc.name;
    }
}

INSTANTIATE_TEST_SUITE_P(
    PipelineTestSuite,
    ComfyUIJsonConverterParamTest,
    ::testing::ValuesIn(kConverterTestCases),
    ComfyUIJsonConverterParamTest::get_test_case_name
);

// ============================================================================
// Edge Case Tests (non-parameterized)
// ============================================================================
// Edge Case Parameterized Tests
// ============================================================================

enum class EdgeCaseTestType {
    FILE_API,       // Test comfyui_json_to_yaml (file path)
    STRING_API      // Test comfyui_json_string_to_yaml (string content)
};

struct EdgeCaseTestCase {
    std::string name;               // Test case name
    EdgeCaseTestType test_type;     // Which API to test
    std::string json_input;         // JSON file path or content
    bool expected_success;          // Expected result
    std::string expected_keyword;   // Keyword expected in YAML (if success)
};

class ComfyUIJsonConverterEdgeCaseParamTest : public ::testing::TestWithParam<EdgeCaseTestCase> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<EdgeCaseTestCase>& info) {
        return info.param.name;
    }
};

namespace {

// Edge case test definitions for boundary conditions and error handling.
// Using anonymous namespace to ensure internal linkage and avoid ODR violations.
const EdgeCaseTestCase kEdgeCaseTestCases[] = {
    // Non-existent file (file API)
    {
        "NonExistentFile",
        EdgeCaseTestType::FILE_API,
        "/non/existent/path/file.json",
        false,
        ""
    },
    // Invalid JSON string (string API)
    {
        "InvalidJsonString",
        EdgeCaseTestType::STRING_API,
        "{ this is not valid json }",
        false,
        ""
    },
    // Empty JSON string (string API)
    {
        "EmptyJsonString",
        EdgeCaseTestType::STRING_API,
        "",
        false,
        ""
    },
    // Default values when inputs empty (file API)
    {
        "DefaultValuesWhenInputsEmpty",
        EdgeCaseTestType::FILE_API,
        get_test_file_path("z_image_turbo_2k_tiled_api.json"),
        true,
        "./models/", // Check default model path (hardcoded default in pipeline.cpp)
    }
};

}  // anonymous namespace

TEST_P(ComfyUIJsonConverterEdgeCaseParamTest, EdgeCaseTest) {
    const auto& tc = GetParam();

    ov::AnyMap pipeline_inputs;
    std::string yaml_content;

    if (tc.test_type == EdgeCaseTestType::FILE_API) {
        fs::path json_file = tc.json_input;
        // For valid file tests, check existence first
        if (tc.expected_success && !fs::exists(json_file)) {
            FAIL() << "Test file not found: " << json_file;
        }
        yaml_content = ModulePipeline::comfyui_json_to_yaml(json_file, pipeline_inputs);
    } else {
        yaml_content = ModulePipeline::comfyui_json_string_to_yaml(tc.json_input, pipeline_inputs);
    }

    if (tc.expected_success) {
        ASSERT_FALSE(yaml_content.empty())
            << "Conversion should succeed for: " << tc.name;

        if (!tc.expected_keyword.empty()) {
            EXPECT_NE(yaml_content.find(tc.expected_keyword), std::string::npos)
                << "YAML should contain '" << tc.expected_keyword << "' for: " << tc.name;
        }
    } else {
        EXPECT_TRUE(yaml_content.empty())
            << "Conversion should fail for: " << tc.name;
    }
}

INSTANTIATE_TEST_SUITE_P(
    PipelineTestSuite,
    ComfyUIJsonConverterEdgeCaseParamTest,
    ::testing::ValuesIn(kEdgeCaseTestCases),
    ComfyUIJsonConverterEdgeCaseParamTest::get_test_case_name
);