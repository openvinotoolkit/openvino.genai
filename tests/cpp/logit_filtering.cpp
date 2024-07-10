// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <openvino/core/except.hpp>

#include "logit_processor.hpp"

using namespace LogitTransformers;

struct TemperatureTransformTestStruct {
    float temperature;
    std::vector<Token> input;
    std::vector<Token> expected_output;
};

using TemperatureTransformTest = testing::TestWithParam<TemperatureTransformTestStruct>;

TEST_P(TemperatureTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = test_struct.input;
    auto transform = TemperatureLogitTransform(test_struct.temperature);
    transform.apply(logits);
    ASSERT_EQ(logits.size(), test_struct.expected_output.size());
    std::sort(logits.begin(), logits.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
    for (size_t i = 0; i < logits.size(); i++) {
        EXPECT_NEAR(logits[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(logits[i].m_index, test_struct.expected_output[i].m_index);
    }
}


const std::vector<TemperatureTransformTestStruct> TEMPERATURE_TRANSFORM_TEST_CASES = {
    {1.0f, { {1.0f, 0}, {2.0f, 1}, {3.0f, 2} }, { {0.665241, 2}, {0.244728, 1}, {0.090031, 0} } },
    {2.0f, { {1.0f, 2}, {2.0f, 1}, {3.0f, 0} }, { {0.506480, 0}, {0.307195, 1}, {0.186323, 2} } },
    {1.0f, { {3.0f, 0}, {1.0f, 1}, {2.0f, 2} }, { {0.665241, 0}, {0.244728, 2}, {0.090031, 1} } },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         TemperatureTransformTest,
                         testing::ValuesIn(TEMPERATURE_TRANSFORM_TEST_CASES));



struct TopPTestStruct {
    float top_p;
    std::vector<Token> input;
    std::vector<Token> expected_output;
};

using TopPFilteringTest = testing::TestWithParam<TopPTestStruct>;

TEST_P(TopPFilteringTest, FilterResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = test_struct.input;
    auto transform = TopPFilter(test_struct.top_p);
    transform.apply(logits);
    ASSERT_EQ(logits.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < logits.size(); i++) {
        EXPECT_NEAR(logits[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(logits[i].m_index, test_struct.expected_output[i].m_index);
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
    std::vector<Token> input;
    std::vector<Token> expected_output;
};

using TopKFilteringTest = testing::TestWithParam<TopKTestStruct>;

TEST_P(TopKFilteringTest, FilterResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = test_struct.input;
    auto transform = TopKFilter(test_struct.top_k);
    transform.apply(logits);
    ASSERT_EQ(logits.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < logits.size(); i++) {
        EXPECT_NEAR(logits[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(logits[i].m_index, test_struct.expected_output[i].m_index);
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

struct RepetitionPenaltyTransformTestStruct {
    float penalty;
    std::vector<Token> input;
    TokenIds input_ids;
    std::vector<Token> expected_output;
};

using RepetitionPenaltyTransformTest = testing::TestWithParam<RepetitionPenaltyTransformTestStruct>;

TEST_P(RepetitionPenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = test_struct.input;
    auto transform = RepetitionPenaltyTransform(test_struct.penalty);
    transform.apply(logits, test_struct.input_ids);
    ASSERT_EQ(logits.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < logits.size(); i++) {
        EXPECT_NEAR(logits[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(logits[i].m_index, test_struct.expected_output[i].m_index);
    }
}


const std::vector<RepetitionPenaltyTransformTestStruct> REPETITION_PENALTY_TRANSFORM_TEST_CASES = {
    { // basic case, indices are applied, order is left as-is
        1.2f,
        { {1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { 2, 0 },
        { {0.8333333f, 0}, {2.0f, 1}, {2.5f, 2} }
    },
    { // negative scores case
        2.0f,
        { {-1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { 0, 1 },
        { {-2.0f, 0}, {1.0f, 1}, {3.0f, 2} }
    },
    { // repeated tokens in prompt, check that the penalty is only applied once
        0.5f,
        { {-1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { 1, 1 },
        { {-1.0f, 0}, {4.0f, 1}, {3.0f, 2} }
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         RepetitionPenaltyTransformTest,
                         testing::ValuesIn(REPETITION_PENALTY_TRANSFORM_TEST_CASES));

TEST(RepetitionPenaltyTransformInitializationTest, ThrowsForInvalidInputIds) {
    auto transform = RepetitionPenaltyTransform(1.5);
    std::vector<Token> input {{43.0f, 0}};
    EXPECT_THROW(transform.apply(input, {1337}), ov::Exception);
    input = {{18.0f, 0}};
    EXPECT_THROW(transform.apply(input, {0, -1}), ov::Exception);
}

struct FrequencyPenaltyTransformTestStruct {
    float penalty;
    std::vector<Token> input;
    TokenIds input_ids;
    std::vector<Token> expected_output;
};

using FrequencyPenaltyTransformTest = testing::TestWithParam<FrequencyPenaltyTransformTestStruct>;

TEST_P(FrequencyPenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = test_struct.input;
    auto transform = FrequencyPenaltyTransform(test_struct.penalty);
    transform.apply(logits, test_struct.input_ids);
    ASSERT_EQ(logits.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < logits.size(); i++) {
        EXPECT_NEAR(logits[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(logits[i].m_index, test_struct.expected_output[i].m_index);
    }
};


const std::vector<FrequencyPenaltyTransformTestStruct> FREQUENCY_PENALTY_TRANSFORM_TEST_CASES = {
    { // basic case, indices are applied, order is left as-is
        0.5f,
        { {-1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { 1, 0 },
        { {-0.5f, 0}, {1.5f, 1}, {3.0f, 2} }
    },
    { // negative scores case
        -0.6f,
        { {-1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { 0, 1, 1 },
        { {-1.6f, 0}, {3.2f, 1}, {3.0f, 2} }
    },
    { // repeated tokens in prompt, check that the penalty is only applied once
        0.2f,
        { {1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { 2, 0, 2 },
        { {0.8f, 0}, {2.0f, 1}, {2.6f, 2} }
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         FrequencyPenaltyTransformTest,
                         testing::ValuesIn(FREQUENCY_PENALTY_TRANSFORM_TEST_CASES));

TEST(FrequencyPenaltyTransformInitializationTest, ThrowsForInvalidInputIds) {
    auto transform = FrequencyPenaltyTransform(1.5);
    std::vector<Token> input {{43.0f, 0}};
    EXPECT_THROW(transform.apply(input, {1337}), ov::Exception);
    input = {{18.0f, 0}};
    EXPECT_THROW(transform.apply(input, {0, -1}), ov::Exception);
}


struct PresencePenaltyTransformTestStruct {
    float penalty;
    std::vector<Token> input;
    TokenIds input_ids;
    std::vector<Token> expected_output;
};

using PresencePenaltyTransformTest = testing::TestWithParam<PresencePenaltyTransformTestStruct>;

TEST_P(PresencePenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = test_struct.input;
    auto transform = PresencePenaltyTransform(test_struct.penalty);
    transform.apply(logits, test_struct.input_ids);
    ASSERT_EQ(logits.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < logits.size(); i++) {
        EXPECT_NEAR(logits[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(logits[i].m_index, test_struct.expected_output[i].m_index);
    }
};


const std::vector<PresencePenaltyTransformTestStruct> PRESENCE_PENALTY_TRANSFORM_TEST_CASES = {
    { // basic case, indices are applied, order is left as-is
        0.5f,
        { {-1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { 1, 0 },
        { {-0.5f, 0}, {1.5f, 1}, {3.0f, 2} }
    },
    { // negative scores case
        -0.6f,
        { {-1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { 0, 1, 1 },
        { {-1.6f, 0}, {2.6f, 1}, {3.0f, 2} }
    },
    { // repeated tokens in prompt, check that the penalty is only applied once
        0.2f,
        { {1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { 2, 0, 2 },
        { {0.8f, 0}, {2.0f, 1}, {2.8f, 2} }
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         PresencePenaltyTransformTest,
                         testing::ValuesIn(PRESENCE_PENALTY_TRANSFORM_TEST_CASES));

TEST(PresencePenaltyTransformInitializationTest, ThrowsForInvalidInputIds) {
    auto transform = PresencePenaltyTransform(1.5);
    std::vector<Token> input {{43.0f, 0}};
    EXPECT_THROW(transform.apply(input, {1337}), ov::Exception);
    input = {{18.0f, 0}};
    EXPECT_THROW(transform.apply(input, {0, -1}), ov::Exception);
}

struct EOSPenaltyTransformTestStruct {
    size_t eos_token_id;
    std::vector<Token> input;
    std::vector<Token> expected_output;
};

using EOSPenaltyTransformTest = testing::TestWithParam<EOSPenaltyTransformTestStruct>;

TEST_P(EOSPenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = test_struct.input;
    auto transform = EOSPenaltyTransform(test_struct.eos_token_id, std::numeric_limits<size_t>::max());
    transform.apply(logits);
    ASSERT_EQ(logits.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < logits.size(); i++) {
        EXPECT_NEAR(logits[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(logits[i].m_index, test_struct.expected_output[i].m_index);
    }
}


const std::vector<EOSPenaltyTransformTestStruct> EOS_PENALTY_TRANSFORM_TEST_CASES = {
    { // basic case, indices are applied, order is left as-is
        1,
        { {1.0f, 0}, {2.0f, 1}, {3.0f, 2} },
        { {1.0f, 0}, {0.0f, 1}, {3.0f, 2} },
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         EOSPenaltyTransformTest,
                         testing::ValuesIn(EOS_PENALTY_TRANSFORM_TEST_CASES));