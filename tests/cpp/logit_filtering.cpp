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
    auto transform = TemperatureLogitTransform(test_struct.temperature);
    auto test_result = transform.apply(test_struct.input);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    std::sort(test_result.begin(), test_result.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(test_result[i].m_index, test_struct.expected_output[i].m_index);
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
    auto transform = TopPFilter(test_struct.top_p);
    auto test_result = transform.apply(test_struct.input);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(test_result[i].m_index, test_struct.expected_output[i].m_index);
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
    auto transform = TopKFilter(test_struct.top_k);
    auto test_result = transform.apply(test_struct.input);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(test_result[i].m_index, test_struct.expected_output[i].m_index);
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
    std::vector<Token> input;
    std::vector<Token> expected_output;
};

using ProbabilityNormalizeTransformTest = testing::TestWithParam<ProbabilityNormalizeTransformTestStruct>;

TEST_P(ProbabilityNormalizeTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto transform = ProbabilityNormalizeTransform();
    auto test_result = transform.apply(test_struct.input);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(test_result[i].m_index, test_struct.expected_output[i].m_index);
    }
}


const std::vector<ProbabilityNormalizeTransformTestStruct> NORMALIZE_TRANSFORM_TEST_CASES = {
    { { {0.090031, 2}, {0.244728, 0}, {0.665241, 1} }, { {0.090031, 2}, {0.244728, 0}, {0.665241, 1}  } },
    { { {0.05, 0}, {0.03, 1}, {0.02, 2} }, { {0.5, 0}, {0.3, 1}, {0.2, 2}  } },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         ProbabilityNormalizeTransformTest,
                         testing::ValuesIn(NORMALIZE_TRANSFORM_TEST_CASES));

struct RepetitionPenaltyTransformTestStruct {
    float penalty;
    std::vector<Token> input_logits;
    TokenIds input_ids;
    std::vector<Token> expected_output;
};

using RepetitionPenaltyTransformTest = testing::TestWithParam<RepetitionPenaltyTransformTestStruct>;

TEST_P(RepetitionPenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto transform = RepetitionPenaltyTransform(test_struct.penalty);
    auto test_result = transform.apply(test_struct.input_logits, test_struct.input_ids);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(test_result[i].m_index, test_struct.expected_output[i].m_index);
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
    EXPECT_THROW(transform.apply({{43.0f, 0}}, {1337}), ov::Exception);
    EXPECT_THROW(transform.apply({{18.0f, 0}}, {0, -1}), ov::Exception);
}

struct FrequencyPenaltyTransformTestStruct {
    float penalty;
    std::vector<Token> input_logits;
    TokenIds input_ids;
    std::vector<Token> expected_output;
};

using FrequencyPenaltyTransformTest = testing::TestWithParam<FrequencyPenaltyTransformTestStruct>;

TEST_P(FrequencyPenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto transform = FrequencyPenaltyTransform(test_struct.penalty);
    auto test_result = transform.apply(test_struct.input_logits, test_struct.input_ids);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(test_result[i].m_index, test_struct.expected_output[i].m_index);
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
    EXPECT_THROW(transform.apply({{43.0f, 0}}, {1337}), ov::Exception);
    EXPECT_THROW(transform.apply({{18.0f, 0}}, {0, -1}), ov::Exception);
}


struct PresencePenaltyTransformTestStruct {
    float penalty;
    std::vector<Token> input_logits;
    TokenIds input_ids;
    std::vector<Token> expected_output;
};

using PresencePenaltyTransformTest = testing::TestWithParam<PresencePenaltyTransformTestStruct>;

TEST_P(PresencePenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto transform = PresencePenaltyTransform(test_struct.penalty);
    auto test_result = transform.apply(test_struct.input_logits, test_struct.input_ids);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(test_result[i].m_index, test_struct.expected_output[i].m_index);
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
    EXPECT_THROW(transform.apply({{43.0f, 0}}, {1337}), ov::Exception);
    EXPECT_THROW(transform.apply({{18.0f, 0}}, {0, -1} ), ov::Exception);
}

struct EOSPenaltyTransformTestStruct {
    size_t eos_token_id;
    std::vector<Token> input_logits;
    std::vector<Token> expected_output;
};

using EOSPenaltyTransformTest = testing::TestWithParam<EOSPenaltyTransformTestStruct>;

TEST_P(EOSPenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto transform = EOSPenaltyTransform(test_struct.eos_token_id, std::numeric_limits<size_t>::max());
    auto test_result = transform.apply(test_struct.input_logits);
    ASSERT_EQ(test_result.size(), test_struct.expected_output.size());
    for (size_t i = 0; i < test_result.size(); i++) {
        EXPECT_NEAR(test_result[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(test_result[i].m_index, test_struct.expected_output[i].m_index);
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