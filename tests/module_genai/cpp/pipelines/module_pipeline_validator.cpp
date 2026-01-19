// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ModulePipelineValidatorTest.cpp
 * @brief GTest-based parameterized tests for ModulePipeline::validate_config
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <tuple>

#include "openvino/genai/module_genai/pipeline.hpp"

namespace fs = std::filesystem;
using namespace ov::genai::module;

// ============================================================================
// Test Case Data Structure
// ============================================================================

struct ValidatorTestCase {
    std::string name;           // Test case name
    std::string yaml_content;   // YAML content to validate
    bool expected_valid;        // Expected validation result
    size_t expected_error_count;// Expected number of errors
    std::string error_keyword;  // Keyword expected in error message (empty = don't check)
    bool check_warning;         // Whether to check warnings
    std::string warning_keyword;// Keyword expected in warning message
};

// ============================================================================
// Parameterized Test Fixture
// ============================================================================

class ModulePipelineValidatorTest : public ::testing::TestWithParam<ValidatorTestCase> {
public:
    // Test case name generator for better output (must be public for INSTANTIATE_TEST_SUITE_P)
    static std::string get_test_case_name(const ::testing::TestParamInfo<ValidatorTestCase>& info) {
        return info.param.name;
    }

protected:
    std::string test_dir_ = "./test_validator_temp";

    void SetUp() override {
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }

    std::string create_temp_yaml(const std::string& content) {
        std::string filepath = test_dir_ + "/test_config.yaml";
        std::ofstream file(filepath);
        file << content;
        file.close();
        return filepath;
    }
};

// ============================================================================
// Test Case Definitions
// ============================================================================

const ValidatorTestCase kValidatorTestCases[] = {
    // Valid configs
    {
        "ValidConfigWithParameterAndResult",
        R"(
global_context:
  model_type: zimage

pipeline_modules:
  pipeline_params:
    type: ParameterModule
    outputs:
      - name: prompt
        type: String
      - name: width
        type: Int
      - name: height
        type: Int

  pipeline_result:
    type: ResultModule
    inputs:
      - name: image
        source: vae.image
        type: OVTensor
)",
        true,   // expected_valid
        0,      // expected_error_count
        "",     // error_keyword
        false,  // check_warning
        ""      // warning_keyword
    },
    
    {
        "ValidConfigWithManyModules",
        R"(
global_context:
  model_type: zimage

pipeline_modules:
  pipeline_params:
    type: ParameterModule
    outputs:
      - name: prompt
        type: String

  clip_text_encoder:
    type: ClipTextEncoderModule
    device: GPU
    inputs:
      - name: prompt
        source: pipeline_params.prompt
        type: String
    outputs:
      - name: prompt_embeds
        type: VecOVTensor

  vae:
    type: VAEDecoderModule
    device: GPU

  pipeline_result:
    type: ResultModule
    inputs:
      - name: image
        source: vae.image
        type: OVTensor
)",
        true, 0, "", false, ""
    },

    // Missing module tests
    {
        "MissingParameterModule",
        R"(
global_context:
  model_root: .

pipeline_modules:
  vae:
    type: VAEDecoderModule

  pipeline_result:
    type: ResultModule
    inputs:
      - name: image
        source: vae.image
        type: OVTensor
)",
        false,  // expected_valid
        1,      // expected_error_count
        "ParameterModule",  // error_keyword
        false, ""
    },

    {
        "MissingResultModule",
        R"(
global_context:
  model_root: .

pipeline_modules:
  pipeline_params:
    type: ParameterModule
    outputs:
      - name: prompt
        type: String

  vae:
    type: VAEDecoderModule
)",
        false, 1, "ResultModule", false, ""
    },

    {
        "MissingBothModules",
        R"(
global_context:
  model_root: .

pipeline_modules:
  vae:
    type: VAEDecoderModule

  clip:
    type: ClipTextEncoderModule
)",
        false, 2, "", false, ""  // 2 errors (ParameterModule and ResultModule), don't check specific keyword
    },

    {
        "MissingPipelineModulesSection",
        R"(
global_context:
  model_type: zimage

some_other_section:
  key: value
)",
        false, 1, "pipeline_modules", false, ""
    },

    // Invalid YAML tests
    {
        "InvalidYamlSyntax",
        R"(
global_context:
  model_root: .

pipeline_modules:
  pipeline_params:
    type: ParameterModule
      invalid_indent: this is wrong
    outputs:
      - name: prompt
)",
        false, 1, "YAML", false, ""
    },

    {
        "EmptyConfig",
        "",
        false, 1, "Empty", false, ""  // 1 error: empty YAML content
    },

    // global_context is optional (warning only)
    {
        "MissingGlobalContextWarning",
        R"(
pipeline_modules:
  pipeline_params:
    type: ParameterModule
    outputs:
      - name: prompt
        type: String

  pipeline_result:
    type: ResultModule
    inputs:
      - name: image
        source: vae.image
        type: OVTensor
)",
        true,   // valid - global_context is optional
        0,      // 0 errors
        "",     // no error_keyword
        true,   // check_warning
        "global_context"  // warning_keyword
    },

    {
        "ValidConfigWithGlobalContext",
        R"(
global_context:
  model_type: zimage

pipeline_modules:
  pipeline_params:
    type: ParameterModule
    outputs:
      - name: prompt
        type: String

  pipeline_result:
    type: ResultModule
    inputs:
      - name: image
        source: vae.image
        type: OVTensor
)",
        true, 0, "", false, ""  // No warnings expected
    },
};

// ============================================================================
// Parameterized Test Implementation
// ============================================================================

TEST_P(ModulePipelineValidatorTest, ValidateConfigString) {
    const auto& tc = GetParam();
    
    auto result = ModulePipeline::validate_config_string(tc.yaml_content);
    
    // Check validity
    EXPECT_EQ(result.valid, tc.expected_valid) 
        << "Test case: " << tc.name;
    
    // Check error count
    if (tc.expected_error_count > 0) {
        EXPECT_EQ(result.errors.size(), tc.expected_error_count)
            << "Test case: " << tc.name;
    }
    
    // Check error keyword if specified
    if (!tc.error_keyword.empty() && !result.errors.empty()) {
        bool found_keyword = false;
        for (const auto& err : result.errors) {
            if (err.find(tc.error_keyword) != std::string::npos) {
                found_keyword = true;
                break;
            }
        }
        EXPECT_TRUE(found_keyword) 
            << "Expected error containing '" << tc.error_keyword << "' in test case: " << tc.name;
    }
    
    // Check warnings if requested
    if (tc.check_warning) {
        if (!tc.warning_keyword.empty()) {
            EXPECT_FALSE(result.warnings.empty()) 
                << "Expected warnings in test case: " << tc.name;
            if (!result.warnings.empty()) {
                bool found_warning = false;
                for (const auto& warn : result.warnings) {
                    if (warn.find(tc.warning_keyword) != std::string::npos) {
                        found_warning = true;
                        break;
                    }
                }
                EXPECT_TRUE(found_warning)
                    << "Expected warning containing '" << tc.warning_keyword << "' in test case: " << tc.name;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    PipelineTestSuite,
    ModulePipelineValidatorTest,
    ::testing::ValuesIn(kValidatorTestCases),
    ModulePipelineValidatorTest::get_test_case_name
);

// ============================================================================
// File-based Tests (non-parameterized, as they need file I/O)
// ============================================================================

class PipelineTest : public ::testing::Test {
protected:
    std::string test_dir_ = "./test_validator_file_temp";

    void SetUp() override {
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }

    std::string create_temp_yaml(const std::string& content) {
        std::string filepath = test_dir_ + "/test_config.yaml";
        std::ofstream file(filepath);
        file << content;
        file.close();
        return filepath;
    }
};

TEST_F(PipelineTest, ValidConfigFromFile) {
    const std::string valid_yaml = R"(
global_context:
  model_root: .

pipeline_modules:
  pipeline_params:
    type: ParameterModule
    outputs:
      - name: prompt
        type: String

  pipeline_result:
    type: ResultModule
    inputs:
      - name: image
        source: vae.image
        type: OVTensor
)";

    std::string filepath = create_temp_yaml(valid_yaml);
    auto result = ModulePipeline::validate_config(filepath);
    
    EXPECT_TRUE(result.valid) << "Valid config file should pass";
    EXPECT_TRUE(result.errors.empty());
}

TEST_F(PipelineTest, InvalidConfigFromFile) {
    const std::string invalid_yaml = R"(
pipeline_modules:
  vae:
    type: VAEDecoderModule
)";

    std::string filepath = create_temp_yaml(invalid_yaml);
    auto result = ModulePipeline::validate_config(filepath);
    
    EXPECT_FALSE(result.valid) << "Invalid config file should fail";
    EXPECT_EQ(result.errors.size(), 2) << "Should have 2 errors (missing ParameterModule and ResultModule)";
}

TEST_F(PipelineTest, NonExistentFile) {
    auto result = ModulePipeline::validate_config("non_existent_file_12345.yaml");
    
    EXPECT_FALSE(result.valid) << "Non-existent file should fail";
    EXPECT_FALSE(result.errors.empty());
    EXPECT_TRUE(result.errors[0].find("open") != std::string::npos || 
                result.errors[0].find("file") != std::string::npos)
        << "Error should mention file issue: " << result.errors[0];
}
