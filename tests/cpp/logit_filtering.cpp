// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

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

namespace {
// Softmax(logits / 0.7) of a row with a N(0, spread) tail and `head` tokens `lead` above it.
std::vector<float> random_probabilities(size_t size, size_t head, float lead, float spread, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> tail(0.0f, spread);
    std::vector<float> row(size);
    for (auto& value : row)
        value = tail(rng);
    std::uniform_int_distribution<size_t> pick(0, size - 1);
    for (size_t i = 0; i < head; ++i)
        row[pick(rng)] = lead - 4.0f * static_cast<float>(i) / static_cast<float>(head);
    Logits logits(row.data(), size);
    TemperatureLogitTransform(0.7f).apply(logits);
    return row;
}

// Sort every token, then walk to the first prefix whose mass exceeds top_p.
std::vector<Token> reference_nucleus(const std::vector<float>& probabilities, float top_p) {
    std::vector<Token> tokens;
    for (size_t i = 0; i < probabilities.size(); ++i)
        tokens.emplace_back(probabilities[i], static_cast<int64_t>(i));
    std::sort(tokens.begin(), tokens.end(), [](const Token& lhs, const Token& rhs) {
        return lhs.m_log_prob > rhs.m_log_prob || (lhs.m_log_prob == rhs.m_log_prob && lhs.m_index < rhs.m_index);
    });
    float sum = 0.0f;
    for (size_t i = 0; i < tokens.size(); ++i) {
        sum += tokens[i].m_log_prob;
        if (sum > top_p) {
            tokens.resize(i + 1);
            break;
        }
    }
    return tokens;
}

void expect_same_tokens(const Logits& logits, const std::vector<Token>& expected) {
    ASSERT_EQ(logits.m_size, logits.m_vector.size());
    ASSERT_EQ(logits.m_size, expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(logits.m_vector[i].m_index, expected[i].m_index) << "position " << i;
        EXPECT_EQ(logits.m_vector[i].m_log_prob, expected[i].m_log_prob) << "position " << i;
    }
}
}  // namespace

// The threshold rounds must give the nucleus of a full sort, from m_data and from an initialized m_vector. The
// largest top_p values need tokens far below the first threshold, so they exercise the widening rounds.
TEST(TopPFilteringTest, MatchesFullSortOnLargeRows) {
    struct Row { size_t head; float lead; float spread; };
    const Row rows[] = {{1, 16.0f, 2.0f}, {64, 10.0f, 2.0f}, {4096, 4.0f, 2.0f}, {1, 4.0f, 0.5f}, {1, 40.0f, 6.0f}};
    for (const float top_p : {0.2f, 0.9f, 0.95f, 0.999f, 0.99999f}) {
        for (size_t r = 0; r < std::size(rows); ++r) {
            SCOPED_TRACE("top_p " + std::to_string(top_p) + ", row " + std::to_string(r));
            const auto probabilities = random_probabilities(50000, rows[r].head, rows[r].lead, rows[r].spread, 11 + r);
            const auto expected = reference_nucleus(probabilities, top_p);

            auto data = probabilities;
            Logits from_data(data.data(), data.size());
            TopPFilter(top_p).apply(from_data);
            expect_same_tokens(from_data, expected);

            auto copy = probabilities;
            Logits from_vector(copy.data(), copy.size());
            from_vector.initialize_vector();
            TopPFilter(top_p).apply(from_vector);
            expect_same_tokens(from_vector, expected);
        }
    }
}

// The softmax records the top probability exactly, and TopPFilter only starts its threshold from it.
TEST(TopPFilteringTest, MaxProbabilityHintDoesNotChangeTheNucleus) {
    const auto probabilities = random_probabilities(50000, 64, 10.0f, 2.0f, 3);
    const float top = *std::max_element(probabilities.begin(), probabilities.end());
    const auto expected = reference_nucleus(probabilities, 0.95f);
    for (const float hint : {top, 1.0f, 1e-30f, 0.0f, NAN}) {
        SCOPED_TRACE("hint " + std::to_string(hint));
        auto data = probabilities;
        Logits logits(data.data(), data.size());
        logits.m_max_probability = hint;
        TopPFilter(0.95f).apply(logits);
        expect_same_tokens(logits, expected);
    }

    std::vector<float> row(1000);
    for (size_t i = 0; i < row.size(); ++i)
        row[i] = static_cast<float>(i % 97) * 0.1f;
    for (const bool vector_path : {false, true}) {
        auto data = row;
        Logits logits(data.data(), data.size());
        if (vector_path)
            logits.initialize_vector();
        TemperatureLogitTransform(0.7f).apply(logits);
        float max_probability = 0.0f;
        for (size_t i = 0; i < logits.m_size; ++i)
            max_probability = std::max(max_probability, vector_path ? logits.m_vector[i].m_log_prob : logits.m_data[i]);
        EXPECT_EQ(logits.m_max_probability, max_probability);
    }
}

TEST(TopPFilteringTest, KeepsEveryTokenWhenNoPrefixExceedsTopP) {
    float input[] = {0.1f, 0.3f, 0.2f};
    Logits logits(input, 3);
    TopPFilter(0.9f).apply(logits);
    expect_same_tokens(logits, {{0.3f, 1}, {0.2f, 2}, {0.1f, 0}});
}

TEST(TopPFilteringTest, OrdersTiesByIndex) {
    float input[] = {0.125f, 0.25f, 0.125f, 0.25f, 0.125f, 0.125f};
    Logits logits(input, 6);
    TopPFilter(0.6f).apply(logits);
    expect_same_tokens(logits, {{0.25f, 1}, {0.25f, 3}, {0.125f, 0}});
}

TEST(TopPFilteringTest, KeepsEveryTokenOfANaNRow) {
    float input[] = {NAN, NAN, NAN, NAN};
    Logits logits(input, 4);
    TopPFilter(0.9f).apply(logits);
    ASSERT_EQ(logits.m_size, 4u);
    ASSERT_EQ(logits.m_vector.size(), 4u);
    std::set<int64_t> indices;
    for (const auto& token : logits.m_vector)
        indices.insert(token.m_index);
    EXPECT_EQ(indices, (std::set<int64_t>{0, 1, 2, 3}));
}

TEST(TemperatureTransformTest, FlushesTheFarTailToZero) {
    // After the 1/T scaling the tail sits 100+ below the top, where expf would return denormals.
    std::vector<float> input(1000, -80.0f);
    input[3] = 0.0f;
    input[7] = -5.0f;
    input[11] = -44.0f;
    Logits logits(input.data(), input.size());
    TemperatureLogitTransform(0.7f).apply(logits);
    const float scale = 1.0f / 0.7f;
    const float norm_sum = 1.0f + expf(-5.0f * scale) + expf(-44.0f * scale);
    EXPECT_FLOAT_EQ(input[3], 1.0f / norm_sum);
    EXPECT_FLOAT_EQ(input[7], expf(-5.0f * scale) / norm_sum);
    EXPECT_FLOAT_EQ(input[11], expf(-44.0f * scale) / norm_sum);
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NE(std::fpclassify(input[i]), FP_SUBNORMAL) << "token " << i;
        if (i != 3 && i != 7 && i != 11) {
            EXPECT_EQ(input[i], 0.0f) << "token " << i;
        }
    }
}



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
    // Sort by index for order-independent comparison: TopKFilter uses a min-heap
    // so output order is not deterministic across stdlib implementations.
    auto actual = logits.m_vector;
    auto expected = test_struct.expected_output;
    auto by_index = [](const Token& a, const Token& b) { return a.m_index < b.m_index; };
    std::sort(actual.begin(), actual.end(), by_index);
    std::sort(expected.begin(), expected.end(), by_index);
    for (size_t i = 0; i < actual.size(); i++) {
        EXPECT_NEAR(actual[i].m_log_prob, expected[i].m_log_prob, 1e-6);
        EXPECT_EQ(actual[i].m_index, expected[i].m_index);
    }
}


const std::vector<TopKTestStruct> TOP_K_TRANSFORM_TEST_CASES = {
    {1, { 0.090031, 0.244728, 0.665241 }, { {0.665241, 2} } },
    {2, { 0.090031, 0.244728, 0.665241 }, { {0.244728, 1}, {0.665241, 2} } },
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
        if (std::isinf(test_struct.expected_output[i])) {
            EXPECT_EQ(logits.m_data[i], test_struct.expected_output[i]);
        } else {
            EXPECT_NEAR(logits.m_data[i], test_struct.expected_output[i], 1e-6);
        }
    }
}


const std::vector<EOSPenaltyTransformTestStruct> EOS_PENALTY_TRANSFORM_TEST_CASES = {
    EOSPenaltyTransformTestStruct{ // basic case, indices are applied, order is left as-is
        { 1 },
        { 1.0f, 2.0f, 3.0f },
        { 1.0f, -std::numeric_limits<float>::infinity(), 3.0f },
    },
    EOSPenaltyTransformTestStruct{
        { 1, 0 },
        { 1.0f, 2.0f, 3.0f },
        { -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 3.0f },
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         EOSPenaltyTransformTest,
                         testing::ValuesIn(EOS_PENALTY_TRANSFORM_TEST_CASES));

// ---------------------------------------------------------------------------
// Tests for the deferred-expf code path (TopKFilter + TemperatureLogitTransform
// with defer_expf=true).  The deferred path keeps raw logits in m_vector and
// moves the expf call to _multinomial_sample; Temperature must only scale.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MinPFilter tests
// ---------------------------------------------------------------------------
// MinPFilter works on normalised probabilities in m_data / m_vector.
// Following the TopPFilter test pattern, input values are pre-computed
// softmax probabilities placed directly in m_data (no Temperature step),
// so every parameterised test exercises the fast path that collects the
// surviving tokens from m_data.
//
// Probabilities used across test cases (softmax of logits {1, 2, 4, 3}):
//   idx=0: 0.032058   idx=1: 0.087144   idx=2: 0.643952   idx=3: 0.236846
//   p_max = p[2] ≈ 0.6440
//
// min_p=0.0  → threshold=0      → all 4 tokens survive
// min_p=0.1  → threshold≈0.0644 → p[0]=0.032 removed; tokens 1, 2, 3 survive
// min_p=0.4  → threshold≈0.2576 → only token 2 survives

struct MinPTestStruct {
    static inline const size_t size = 4;

    float min_p;
    float input[size];               // normalised probabilities, fed directly as m_data
    std::vector<Token> expected_output;  // surviving tokens {prob, original_index}
};

using MinPFilteringTest = testing::TestWithParam<MinPTestStruct>;

TEST_P(MinPFilteringTest, FilterResultEqualToReference) {
    auto test_struct = GetParam();
    auto logits = Logits(test_struct.input, MinPTestStruct::size);
    ASSERT_FALSE(logits.is_vector_initialized());

    MinPFilter filter(test_struct.min_p);
    filter.apply(logits);

    ASSERT_TRUE(logits.is_vector_initialized());  // MinPFilter must initialize m_vector from m_data
    ASSERT_EQ(logits.m_size, logits.m_vector.size());
    ASSERT_EQ(logits.m_size, test_struct.expected_output.size());

    // Sort by index for a deterministic comparison (partition order is input order,
    // but sorting makes the test independent of implementation details).
    auto actual = logits.m_vector;
    auto expected = test_struct.expected_output;
    auto by_index = [](const Token& a, const Token& b) { return a.m_index < b.m_index; };
    std::sort(actual.begin(), actual.end(), by_index);
    std::sort(expected.begin(), expected.end(), by_index);

    for (size_t i = 0; i < actual.size(); i++) {
        EXPECT_NEAR(actual[i].m_log_prob, expected[i].m_log_prob, 1e-5f);
        EXPECT_EQ(actual[i].m_index, expected[i].m_index);
    }
}

const std::vector<MinPTestStruct> MIN_P_TRANSFORM_TEST_CASES = {
    MinPTestStruct{
        // min_p = 0.0 is a no-op: all 4 tokens survive with their probabilities unchanged.
        0.0f,
        { 0.032058f, 0.087144f, 0.643952f, 0.236846f },
        { {0.032058f, 0}, {0.087144f, 1}, {0.643952f, 2}, {0.236846f, 3} }
    },
    MinPTestStruct{
        // min_p = 0.1 → threshold ≈ 0.0644 → p[0]=0.032 removed; tokens 1, 2, 3 survive.
        0.1f,
        { 0.032058f, 0.087144f, 0.643952f, 0.236846f },
        { {0.087144f, 1}, {0.643952f, 2}, {0.236846f, 3} }
    },
    MinPTestStruct{
        // min_p = 0.4 → threshold ≈ 0.2576 → only the top token (idx=2) survives.
        0.4f,
        { 0.032058f, 0.087144f, 0.643952f, 0.236846f },
        { {0.643952f, 2} }
    },
};

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         MinPFilteringTest,
                         testing::ValuesIn(MIN_P_TRANSFORM_TEST_CASES));

TEST(MinPFilteringTest, TopTokenAlwaysSurvives) {
    // Uniform probabilities: threshold = min_p * 0.25, and every p_i = 0.25 >= threshold.
    float input[] = {0.25f, 0.25f, 0.25f, 0.25f};
    auto logits = Logits(input, 4);
    MinPFilter(0.99f).apply(logits);
    EXPECT_EQ(logits.m_size, 4u);
}

// Collecting from m_data must keep the same tokens, in the same order, as compacting an initialized m_vector.
TEST(MinPFilteringTest, FastPathMatchesVectorPath) {
    for (const float min_p : {0.0f, 0.01f, 0.05f, 0.5f}) {
        SCOPED_TRACE("min_p " + std::to_string(min_p));
        const auto probabilities = random_probabilities(50000, 64, 10.0f, 2.0f, 5);
        auto copy = probabilities;
        Logits from_vector(copy.data(), copy.size());
        from_vector.initialize_vector();
        MinPFilter(min_p).apply(from_vector);

        auto data = probabilities;
        Logits from_data(data.data(), data.size());
        MinPFilter(min_p).apply(from_data);
        expect_same_tokens(from_data, from_vector.m_vector);
    }
}

TEST(MinPFilteringTest, KeepsOneTokenOfANaNRow) {
    float input[] = {NAN, NAN, NAN};
    auto logits = Logits(input, 3);
    MinPFilter(0.1f).apply(logits);
    ASSERT_EQ(logits.m_size, 1u);
    ASSERT_EQ(logits.m_vector.size(), 1u);
    EXPECT_EQ(logits.m_vector[0].m_index, 0);
}

TEST(MinPFilteringTest, AlwaysKeepsAtLeastOneToken) {
    // Highly skewed: dominant token at index 0 with p ≈ 1.0.
    // min_p=0.99 → threshold ≈ 0.99 * 0.9988 ≈ 0.9888; the three tail tokens are removed.
    float input[] = {0.9988f, 0.0004f, 0.0004f, 0.0004f};
    auto logits = Logits(input, 4);
    MinPFilter(0.99f).apply(logits);
    ASSERT_EQ(logits.m_size, 1u);
    EXPECT_EQ(logits.m_vector[0].m_index, 0);
}

// Verify the subset relation for this specific distribution used by the test.
TEST(MinPFilteringTest, MinPThenTopPIsSubsetOfTopPAloneForSpecificDistribution) {
    float probs[] = {0.032058f, 0.087144f, 0.643952f, 0.236846f};

    // Reference for this distribution: TopP alone (top_p=0.95).
    auto logits_ref = Logits(probs, 4);
    TopPFilter(0.95f).apply(logits_ref);
    std::set<int64_t> ref_indices;
    for (size_t i = 0; i < logits_ref.m_size; ++i)
        ref_indices.insert(logits_ref.m_vector[i].m_index);

    // For this distribution, min_p=0.1 removes token 0 before TopP is applied.
    auto logits_mp = Logits(probs, 4);
    MinPFilter(0.1f).apply(logits_mp);
    TopPFilter(0.95f).apply(logits_mp);
    for (size_t i = 0; i < logits_mp.m_size; ++i)
        EXPECT_TRUE(ref_indices.count(logits_mp.m_vector[i].m_index))
            << "token " << logits_mp.m_vector[i].m_index << " not in TopP-only set";
}

TEST(TopKThenTemperatureTest, DeferredExpfPathScalesLogits) {
    // Input: {1.0f, 2.0f, 3.0f}.
    // TopKFilter(2) retains the 2 highest values: idx=2 (3.0f) and idx=1 (2.0f).
    // TemperatureLogitTransform(2.0, defer_expf=true) divides by T=2:
    //   idx=2 -> 1.5f,  idx=1 -> 1.0f.
    float input[] = {1.0f, 2.0f, 3.0f};
    auto logits = Logits(input, 3);

    TopKFilter topk(2);
    topk.apply(logits);
    ASSERT_TRUE(logits.is_vector_initialized());
    ASSERT_EQ(logits.m_size, 2u);

    TemperatureLogitTransform temp(2.0, /*defer_expf=*/true);
    temp.apply(logits);

    ASSERT_TRUE(logits.m_defer_expf);
    ASSERT_EQ(logits.m_size, 2u);

    // Sort by index for order-independent comparison (heap order varies).
    auto result = logits.m_vector;
    std::sort(result.begin(), result.end(), [](const Token& a, const Token& b) {
        return a.m_index < b.m_index;
    });
    EXPECT_EQ(result[0].m_index, 1);
    EXPECT_NEAR(result[0].m_log_prob, 1.0f, 1e-6f);
    EXPECT_EQ(result[1].m_index, 2);
    EXPECT_NEAR(result[1].m_log_prob, 1.5f, 1e-6f);

    // m_data must not be modified by the transforms.
    EXPECT_NEAR(input[0], 1.0f, 1e-6f);
    EXPECT_NEAR(input[1], 2.0f, 1e-6f);
    EXPECT_NEAR(input[2], 3.0f, 1e-6f);
}

TEST(TopKThenTemperatureTest, DeferredExpfPathTemperatureOneIsNoOp) {
    // T=1.0 must be a true no-op: values in m_vector must be identical after apply().
    float input[] = {1.0f, 2.0f, 3.0f};
    auto logits = Logits(input, 3);

    TopKFilter topk(2);
    topk.apply(logits);
    const auto before = logits.m_vector;

    TemperatureLogitTransform temp(1.0, /*defer_expf=*/true);
    temp.apply(logits);

    ASSERT_TRUE(logits.m_defer_expf);
    ASSERT_EQ(logits.m_vector.size(), before.size());
    for (size_t i = 0; i < logits.m_vector.size(); ++i) {
        EXPECT_EQ(logits.m_vector[i].m_index, before[i].m_index);
        EXPECT_EQ(logits.m_vector[i].m_log_prob, before[i].m_log_prob);
    }
}
