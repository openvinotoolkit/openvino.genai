// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <openvino/core/except.hpp>

#include "sampling/logit_processor.hpp"

using namespace ov::genai;
using namespace ov::genai::LogitTransformers;

struct TemperatureTransformTestStruct {
    static inline const size_t size = 3;

    float temperature;
    float input[size];
    float expected_output[size];
};

using TemperatureTransformTest = testing::TestWithParam<TemperatureTransformTestStruct>;

TEST_P(TemperatureTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = Logits(test_struct.input, TemperatureTransformTestStruct::size);
    auto transform = TemperatureLogitTransform(test_struct.temperature);
    transform.apply(logits);
    ASSERT_FALSE(logits.is_vector_initialized());
    ASSERT_EQ(logits.m_size, TemperatureTransformTestStruct::size); // temperature transform should not change buffer size
    for (size_t i = 0; i < logits.m_size; i++) {
        EXPECT_NEAR(logits.m_data[i], test_struct.expected_output[i], 1e-6);
    }
}


const std::vector<TemperatureTransformTestStruct> TEMPERATURE_TRANSFORM_TEST_CASES = {
    {1.0f, { 1.0f, 2.0f, 3.0f }, { 0.090031, 0.244728, 0.665241 } },
    {2.0f, { 3.0f, 2.0f, 1.0f }, { 0.506480, 0.307195, 0.186323 } },
    {1.0f, { 3.0f, 1.0f, 2.0f }, { 0.665241, 0.090031, 0.244728 } },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         TemperatureTransformTest,
                         testing::ValuesIn(TEMPERATURE_TRANSFORM_TEST_CASES));



struct TopPTestStruct {
    static inline const size_t size = 3;

    float top_p;
    float input[size];
    std::vector<Token> expected_output;
};

using TopPFilteringTest = testing::TestWithParam<TopPTestStruct>;

TEST_P(TopPFilteringTest, FilterResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = Logits(test_struct.input, TopPTestStruct::size);
    auto transform = TopPFilter(test_struct.top_p);
    transform.apply(logits);
    ASSERT_TRUE(logits.is_vector_initialized());
    ASSERT_EQ(logits.m_size, logits.m_vector.size());
    ASSERT_EQ(logits.m_size, test_struct.expected_output.size());
    for (size_t i = 0; i < logits.m_vector.size(); i++) {
        EXPECT_NEAR(logits.m_vector[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(logits.m_vector[i].m_index, test_struct.expected_output[i].m_index);
    }
}


const std::vector<TopPTestStruct> TOP_P_TRANSFORM_TEST_CASES = {
    {0.2f, { 0.090031, 0.244728, 0.665241 }, { {0.665241, 2} } },
    {0.9f, { 0.090031, 0.244728, 0.665241 }, { {0.665241, 2}, {0.244728, 1} } },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         TopPFilteringTest,
                         testing::ValuesIn(TOP_P_TRANSFORM_TEST_CASES));



struct TopKTestStruct {
    static inline const size_t size = 3;

    size_t top_k;
    float input[size];
    std::vector<Token> expected_output;
};

using TopKFilteringTest = testing::TestWithParam<TopKTestStruct>;

TEST_P(TopKFilteringTest, FilterResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = Logits(test_struct.input, TopKTestStruct::size);
    auto transform = TopKFilter(test_struct.top_k);
    transform.apply(logits);
    ASSERT_TRUE(logits.is_vector_initialized());
    ASSERT_EQ(logits.m_size, logits.m_vector.size());
    ASSERT_EQ(logits.m_size, test_struct.expected_output.size());
    for (size_t i = 0; i < logits.m_vector.size(); i++) {
        EXPECT_NEAR(logits.m_vector[i].m_log_prob, test_struct.expected_output[i].m_log_prob, 1e-6);
        EXPECT_EQ(logits.m_vector[i].m_index, test_struct.expected_output[i].m_index);
    }
}


const std::vector<TopKTestStruct> TOP_K_TRANSFORM_TEST_CASES = {
    {1, { 0.090031, 0.244728, 0.665241 }, { {0.665241, 2} } },
    {2, { 0.090031, 0.244728, 0.665241 }, { {0.665241, 2}, {0.244728, 1} } },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         TopKFilteringTest,
                         testing::ValuesIn(TOP_K_TRANSFORM_TEST_CASES));

TEST(TopKFilteringTest, FilterNotAppliedTopKGreaterThanInputSize) {
    float input[]{0.090031, 0.244728, 0.665241};
    float expected_output[]{0.090031, 0.244728, 0.665241}; // no change expected
    size_t top_k = 5;
    auto logits = Logits(input, 3);
    auto transform = TopKFilter(top_k);
    transform.apply(logits);
    ASSERT_FALSE(logits.is_vector_initialized());
    ASSERT_EQ(logits.m_size, 3);
    for (size_t i = 0; i < logits.m_size; i++) {
        EXPECT_EQ(logits.m_data[i], expected_output[i]);
    }
}

struct RepetitionPenaltyTransformTestStruct {
    static inline const size_t size = 3;

    float penalty;
    float input[size];
    TokenIds input_ids;
    float expected_output[size];
};

using RepetitionPenaltyTransformTest = testing::TestWithParam<RepetitionPenaltyTransformTestStruct>;

TEST_P(RepetitionPenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = Logits(test_struct.input, RepetitionPenaltyTransformTestStruct::size);
    auto transform = RepetitionPenaltyTransform(test_struct.penalty);
    transform.apply(logits, test_struct.input_ids);
    ASSERT_FALSE(logits.is_vector_initialized());
    ASSERT_EQ(logits.m_size, RepetitionPenaltyTransformTestStruct::size); // penalty transform should not change buffer size
    for (size_t i = 0; i < logits.m_size; i++) {
        EXPECT_NEAR(logits.m_data[i], test_struct.expected_output[i], 1e-6);
    }
}


const std::vector<RepetitionPenaltyTransformTestStruct> REPETITION_PENALTY_TRANSFORM_TEST_CASES = {
    RepetitionPenaltyTransformTestStruct{ // basic case, indices are applied, order is left as-is
        1.2f,
        { 1.0f, 2.0f, 3.0f },
        TokenIds{ 2, 0 },
        { 0.8333333f, 2.0f, 2.5f }
    },
    RepetitionPenaltyTransformTestStruct{ // negative scores case
        2.0f,
        { -1.0f, 2.0f, 3.0f },
        TokenIds{ 0, 1 },
        { -2.0f, 1.0f, 3.0f }
    },
    RepetitionPenaltyTransformTestStruct{ // repeated tokens in prompt, check that the penalty is only applied once
        0.5f,
        { -1.0f, 2.0f, 3.0f },
        TokenIds{ 1, 1 },
        { -1.0f, 4.0f, 3.0f }
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         RepetitionPenaltyTransformTest,
                         testing::ValuesIn(REPETITION_PENALTY_TRANSFORM_TEST_CASES));

TEST(RepetitionPenaltyTransformInitializationTest, ThrowsForInvalidInputIds) {
    auto transform = RepetitionPenaltyTransform(1.5);
    float input[]{43.0f};
    Logits logits(input, 1);
    EXPECT_THROW(transform.apply(logits, {1337}), ov::Exception);
    input[0] = {18.0f};
    EXPECT_THROW(transform.apply(logits, {0, -1}), ov::Exception);
}


struct FrequencyPenaltyTransformTestStruct {
    static inline const size_t size = 3;

    float penalty;
    float input[size];
    TokenIds input_ids;
    float expected_output[size];
};

using FrequencyPenaltyTransformTest = testing::TestWithParam<FrequencyPenaltyTransformTestStruct>;

TEST_P(FrequencyPenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = Logits(test_struct.input, FrequencyPenaltyTransformTestStruct::size);
    auto transform = FrequencyPenaltyTransform(test_struct.penalty);
    transform.apply(logits, test_struct.input_ids);
    ASSERT_FALSE(logits.is_vector_initialized());
    ASSERT_EQ(logits.m_size, FrequencyPenaltyTransformTestStruct::size); // penalty transform should not change buffer size
    for (size_t i = 0; i < logits.m_size; i++) {
        EXPECT_NEAR(logits.m_data[i], test_struct.expected_output[i], 1e-6);
    }
};


const std::vector<FrequencyPenaltyTransformTestStruct> FREQUENCY_PENALTY_TRANSFORM_TEST_CASES = {
    FrequencyPenaltyTransformTestStruct{ // basic case, indices are applied, order is left as-is
        0.5f,
        { -1.0f, 2.0f, 3.0f },
        TokenIds{ 1, 0 },
        { -0.5f, 1.5f, 3.0f }
    },
    FrequencyPenaltyTransformTestStruct{ // negative scores case
        -0.6f,
        { -1.0f, 2.0f, 3.0f },
        TokenIds{ 0, 1, 1 },
        { -1.6f, 3.2f, 3.0f }
    },
    FrequencyPenaltyTransformTestStruct{ // repeated tokens in prompt, check that the penalty is only applied once
        0.2f,
        { 1.0f, 2.0f, 3.0f },
        TokenIds{ 2, 0, 2 },
        { 0.8f, 2.0f, 2.6f }
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         FrequencyPenaltyTransformTest,
                         testing::ValuesIn(FREQUENCY_PENALTY_TRANSFORM_TEST_CASES));

TEST(FrequencyPenaltyTransformInitializationTest, ThrowsForInvalidInputIds) {
    auto transform = FrequencyPenaltyTransform(1.5);
    float input[]{43.0f};
    Logits logits(input, 1);
    EXPECT_THROW(transform.apply(logits, {1337}), ov::Exception);
    input[0] = {18.0f};
    EXPECT_THROW(transform.apply(logits, {0, -1}), ov::Exception);
}


struct PresencePenaltyTransformTestStruct {
    static inline const size_t size = 3;

    float penalty;
    float input[size];
    TokenIds input_ids;
    float expected_output[size];
};

using PresencePenaltyTransformTest = testing::TestWithParam<PresencePenaltyTransformTestStruct>;

TEST_P(PresencePenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = Logits(test_struct.input, PresencePenaltyTransformTestStruct::size);
    auto transform = PresencePenaltyTransform(test_struct.penalty);
    transform.apply(logits, test_struct.input_ids);
    ASSERT_FALSE(logits.is_vector_initialized());
    ASSERT_EQ(logits.m_size, PresencePenaltyTransformTestStruct::size); // penalty transform should not change buffer size
    for (size_t i = 0; i < logits.m_size; i++) {
        EXPECT_NEAR(logits.m_data[i], test_struct.expected_output[i], 1e-6);
    }
};


const std::vector<PresencePenaltyTransformTestStruct> PRESENCE_PENALTY_TRANSFORM_TEST_CASES = {
    PresencePenaltyTransformTestStruct{ // basic case, indices are applied, order is left as-is
        0.5f,
        { -1.0f, 2.0f, 3.0f },
        TokenIds{ 1, 0 },
        { -0.5f, 1.5f, 3.0f }
    },
    PresencePenaltyTransformTestStruct{ // negative scores case
        -0.6f,
        { -1.0f, 2.0f, 3.0f },
        TokenIds{ 0, 1, 1 },
        { -1.6f, 2.6f, 3.0f }
    },
    PresencePenaltyTransformTestStruct{ // repeated tokens in prompt, check that the penalty is only applied once
        0.2f,
        { 1.0f, 2.0f, 3.0f },
        TokenIds{ 2, 0, 2 },
        { 0.8f, 2.0f, 2.8f }
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         PresencePenaltyTransformTest,
                         testing::ValuesIn(PRESENCE_PENALTY_TRANSFORM_TEST_CASES));

TEST(PresencePenaltyTransformInitializationTest, ThrowsForInvalidInputIds) {
    auto transform = PresencePenaltyTransform(1.5);
    float input[]{43.0f};
    Logits logits(input, 1);
    EXPECT_THROW(transform.apply(logits, {1337}), ov::Exception);
    input[0] = {18.0f};
    EXPECT_THROW(transform.apply(logits, {0, -1}), ov::Exception);
}

struct EOSPenaltyTransformTestStruct {
    static inline const size_t size = 3;

    std::set<int64_t> stop_token_ids;
    float input[size];
    float expected_output[size];
};

using EOSPenaltyTransformTest = testing::TestWithParam<EOSPenaltyTransformTestStruct>;

TEST_P(EOSPenaltyTransformTest, TransformResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = Logits(test_struct.input, EOSPenaltyTransformTestStruct::size);
    auto transform = EOSPenaltyTransform(test_struct.stop_token_ids, std::numeric_limits<size_t>::max());
    transform.apply(logits);
    ASSERT_FALSE(logits.is_vector_initialized());
    ASSERT_EQ(logits.m_size, EOSPenaltyTransformTestStruct::size); // penalty transform should not change buffer size
    for (size_t i = 0; i < logits.m_size; i++) {
        EXPECT_NEAR(logits.m_data[i], test_struct.expected_output[i], 1e-6);
    }
}


const std::vector<EOSPenaltyTransformTestStruct> EOS_PENALTY_TRANSFORM_TEST_CASES = {
    EOSPenaltyTransformTestStruct{ // basic case, indices are applied, order is left as-is
        { 1 },
        { 1.0f, 2.0f, 3.0f },
        { 1.0f, 0.0f, 3.0f },
    },
    EOSPenaltyTransformTestStruct{
        { 1, 0 },
        { 1.0f, 2.0f, 3.0f },
        { 0.0f, 0.0f, 3.0f },
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         EOSPenaltyTransformTest,
                         testing::ValuesIn(EOS_PENALTY_TRANSFORM_TEST_CASES));

