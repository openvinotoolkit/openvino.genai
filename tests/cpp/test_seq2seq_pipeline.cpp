// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <iostream>

#include "openvino/genai/seq2seq_pipeline.hpp"

namespace fs = std::filesystem;

class Seq2SeqPipelineTest : public ::testing::Test {
protected:
    // Path to test models should be set up by the test environment
    // or models can be created on-the-fly from smaller test models
    std::string test_models_path;

    void SetUp() override {
        // This would typically be populated from environment or fixtures
        test_models_path = std::getenv("SEQ2SEQ_TEST_MODELS_PATH") ? 
                          std::getenv("SEQ2SEQ_TEST_MODELS_PATH") : "";
    }
};

/**
 * @brief Test that Seq2SeqPipeline can be constructed
 * 
 * Requires test models to be present in SEQ2SEQ_TEST_MODELS_PATH environment variable
 */
TEST_F(Seq2SeqPipelineTest, DISABLED_BasicConstruction) {
    if (test_models_path.empty()) {
        GTEST_SKIP() << "Test models path not set, skipping test";
    }
    
    ASSERT_NO_THROW({
        auto pipeline = ov::genai::Seq2SeqPipeline(test_models_path, "CPU");
    });
}

/**
 * @brief Test that generate() works with single input
 */
TEST_F(Seq2SeqPipelineTest, DISABLED_GenerateSingleInput) {
    if (test_models_path.empty()) {
        GTEST_SKIP() << "Test models path not set, skipping test";
    }

    auto pipeline = ov::genai::Seq2SeqPipeline(test_models_path, "CPU");
    
    ASSERT_NO_THROW({
        auto results = pipeline.generate("This is a test input");
        EXPECT_GT(results.texts.size(), 0);
        EXPECT_EQ(results.texts.size(), results.scores.size());
        EXPECT_EQ(results.texts.size(), results.finish_reasons.size());
    });
}

/**
 * @brief Test that generate() works with batch input
 */
TEST_F(Seq2SeqPipelineTest, DISABLED_GenerateBatchInput) {
    if (test_models_path.empty()) {
        GTEST_SKIP() << "Test models path not set, skipping test";
    }

    auto pipeline = ov::genai::Seq2SeqPipeline(test_models_path, "CPU");
    
    std::vector<std::string> inputs = {
        "This is the first input",
        "This is the second input",
        "This is the third input"
    };

    ASSERT_NO_THROW({
        auto results = pipeline.generate(inputs);
        EXPECT_EQ(results.texts.size(), 3);
        EXPECT_EQ(results.texts.size(), results.scores.size());
        EXPECT_EQ(results.texts.size(), results.finish_reasons.size());
    });
}

/**
 * @brief Test generation config get/set
 */
TEST_F(Seq2SeqPipelineTest, DISABLED_GenerationConfig) {
    if (test_models_path.empty()) {
        GTEST_SKIP() << "Test models path not set, skipping test";
    }

    auto pipeline = ov::genai::Seq2SeqPipeline(test_models_path, "CPU");
    
    auto config = pipeline.get_generation_config();
    EXPECT_GT(config.max_new_tokens, 0);
    
    ov::genai::GenerationConfig new_config;
    new_config.max_new_tokens = 256;
    ASSERT_NO_THROW({
        pipeline.set_generation_config(new_config);
    });
}
