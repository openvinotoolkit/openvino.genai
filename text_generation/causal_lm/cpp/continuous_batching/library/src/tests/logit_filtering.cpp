// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "sampler.hpp"



struct TemperatureTransformTestStruct {
    float temperature;
    std::vector<LogitWithIdx> input;
    std::vector<ProbabilityWithIdx> expected_output;
};

class TemperatureTransformTest : public testing::TestWithParam<TemperatureTransformTestStruct> {
};

TEST_P(TemperatureTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto transform = TemperatureLogitTransform(test_struct.temperature);
    auto test_result = transform.apply(test_struct.input);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].first, test_struct.expected_output[i].first, 1e-6);
        EXPECT_EQ(test_result[i].second, test_struct.expected_output[i].second);
    }
}


const std::vector<TemperatureTransformTestStruct> TEMPERATURE_TRANSFORM_TEST_CASES = {
    {1.0f, { {1.0f, 0}, {2.0f, 1}, {3.0f, 2} }, { {0.090031, 0}, {0.244728, 1}, {0.665241, 2} } },
    {2.0f, { {1.0f, 2}, {2.0f, 1}, {3.0f, 0} }, { {0.186323, 2}, {0.307195, 1}, {0.506480, 0} } }
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         TemperatureTransformTest,
                         testing::ValuesIn(TEMPERATURE_TRANSFORM_TEST_CASES));



struct TopPTestStruct {
    float top_p;
    std::vector<ProbabilityWithIdx> input;
    std::vector<ProbabilityWithIdx> expected_output;
};

class TopPFilteringTest : public testing::TestWithParam<TopPTestStruct> {
};

TEST_P(TopPFilteringTest, FilterResultEqualToReference) {
    auto test_struct = GetParam();
    auto transform = TopPFilter(test_struct.top_p);
    auto test_result = transform.filter(test_struct.input);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].first, test_struct.expected_output[i].first, 1e-6);
        EXPECT_EQ(test_result[i].second, test_struct.expected_output[i].second);
    }
}


const std::vector<TopPTestStruct> TOP_P_TRANSFORM_TEST_CASES = {
    {0.2f, { {0.090031, 0}, {0.244728, 1}, {0.665241, 2} }, { {0.665241, 2} } },
    {0.9f, { {0.090031, 0}, {0.244728, 1}, {0.665241, 2} }, { {0.665241, 2}, {0.244728, 1} } },
    {1.0f, { {0.090031, 0}, {0.244728, 1}, {0.665241, 2} }, { {0.665241, 2}, {0.244728, 1}, {0.090031, 0} } },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         TopPFilteringTest,
                         testing::ValuesIn(TOP_P_TRANSFORM_TEST_CASES));



struct TopKTestStruct {
    size_t top_k;
    std::vector<ProbabilityWithIdx> input;
    std::vector<ProbabilityWithIdx> expected_output;
};

class TopKFilteringTest : public testing::TestWithParam<TopKTestStruct> {
};

TEST_P(TopKFilteringTest, FilterResultEqualToReference) {
    auto test_struct = GetParam();
    auto transform = TopKFilter(test_struct.top_k);
    auto test_result = transform.filter(test_struct.input);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].first, test_struct.expected_output[i].first, 1e-6);
        EXPECT_EQ(test_result[i].second, test_struct.expected_output[i].second);
    }
}


const std::vector<TopKTestStruct> TOP_K_TRANSFORM_TEST_CASES = {
    {1, { {0.090031, 0}, {0.244728, 1}, {0.665241, 2} }, { {0.665241, 2} } },
    {2, { {0.090031, 0}, {0.244728, 1}, {0.665241, 2} }, { {0.665241, 2}, {0.244728, 1} } },
    {5, { {0.090031, 0}, {0.244728, 1}, {0.665241, 2} }, { {0.665241, 2}, {0.244728, 1}, {0.090031, 0} } },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         TopKFilteringTest,
                         testing::ValuesIn(TOP_K_TRANSFORM_TEST_CASES));


struct ProbabilityNormalizeTransformTestStruct {
    std::vector<ProbabilityWithIdx> input;
    std::vector<ProbabilityWithIdx> expected_output;
};

class ProbabilityNormalizeTransformTest : public testing::TestWithParam<ProbabilityNormalizeTransformTestStruct> {
};

TEST_P(ProbabilityNormalizeTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto transform = ProbabilityNormalizeTransform();
    auto test_result = transform.apply(test_struct.input);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].first, test_struct.expected_output[i].first, 1e-6);
        EXPECT_EQ(test_result[i].second, test_struct.expected_output[i].second);
    }
}


const std::vector<ProbabilityNormalizeTransformTestStruct> NORMALIZE_TRANSFORM_TEST_CASES = {
    { { {0.090031, 2}, {0.244728, 0}, {0.665241, 1} }, { {0.090031, 2}, {0.244728, 0}, {0.665241, 1}  } },
    { { {0.05, 0}, {0.03, 1}, {0.02, 2} }, { {0.5, 0}, {0.3, 1}, {0.2, 2}  } },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         ProbabilityNormalizeTransformTest,
                         testing::ValuesIn(NORMALIZE_TRANSFORM_TEST_CASES));
