// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

using namespace ov::genai;

class TextRerankPipelineConfigTest : public ::testing::Test {
protected:
    TextRerankPipeline::Config config;
};

// Test 1: Validation passes with default config
TEST_F(TextRerankPipelineConfigTest, ValidatePassesWithDefaultConfig) {
    // Default config: top_n = 3, max_length = nullopt
    EXPECT_NO_THROW(config.validate());
}

// Test 2: Validation passes with valid max_length and top_n
TEST_F(TextRerankPipelineConfigTest, ValidatePassesWithValidValues) {
    config.max_length = 512;
    config.top_n = 5;
    EXPECT_NO_THROW(config.validate());
}

// Test 3: Validation passes when max_length is not set
TEST_F(TextRerankPipelineConfigTest, ValidatePassesWithoutMaxLength) {
    config.max_length = std::nullopt;
    config.top_n = 10;
    EXPECT_NO_THROW(config.validate());
}

// Test 4: Validation throws when max_length is 0
TEST_F(TextRerankPipelineConfigTest, ValidateThrowsWhenMaxLengthIsZero) {
    config.max_length = 0;
    config.top_n = 3;
    EXPECT_THROW(config.validate(), ov::Exception);
}

// Test 5: Validation throws when top_n is 0
TEST_F(TextRerankPipelineConfigTest, ValidateThrowsWhenTopNIsZero) {
    config.max_length = 512;
    config.top_n = 0;
    EXPECT_THROW(config.validate(), ov::Exception);
}

// Test 6: Validation throws when both max_length and top_n are invalid
TEST_F(TextRerankPipelineConfigTest, ValidateThrowsWhenBothInvalid) {
    config.max_length = 0;
    config.top_n = 0;
    // Should throw on first validation failure (either max_length or top_n)
    EXPECT_THROW(config.validate(), ov::Exception);
}

// Test 7: Validation passes with max_length = 1 (boundary case)
TEST_F(TextRerankPipelineConfigTest, ValidatePassesWithMinValidMaxLength) {
    config.max_length = 1;
    config.top_n = 1;
    EXPECT_NO_THROW(config.validate());
}

// Test 8: Config constructed from AnyMap with valid values passes validation
TEST_F(TextRerankPipelineConfigTest, ValidatePassesWithValidAnyMapConfig) {
    ov::AnyMap properties = {
        {"top_n", size_t(5)},
        {"max_length", size_t(256)}
    };
    TextRerankPipeline::Config config_from_map(properties);
    EXPECT_NO_THROW(config_from_map.validate());
}

// Test 9: Config constructed from AnyMap with invalid top_n throws on validation
TEST_F(TextRerankPipelineConfigTest, ValidateThrowsWithInvalidAnyMapTopN) {
    ov::AnyMap properties = {
        {"top_n", size_t(0)}
    };
    TextRerankPipeline::Config config_from_map(properties);
    EXPECT_THROW(config_from_map.validate(), ov::Exception);
}