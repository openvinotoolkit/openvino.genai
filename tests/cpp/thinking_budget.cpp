// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>
#include <openvino/core/except.hpp>
#include <vector>

#include "sampling/logit_processor.hpp"

using namespace ov::genai;
using namespace ov::genai::LogitTransformers;

namespace {
const float minus_inf = -std::numeric_limits<float>::infinity();
}  // namespace

TEST(ThinkingBudgetTransformTest, ConstructorRejectsNegativeEndTokenId) {
    EXPECT_THROW(ThinkingBudgetTransform(4, -1), ov::Exception);
}

TEST(ThinkingBudgetTransformTest, WithinBudgetLeavesLogitsUntouched) {
    float input[]{1.0f, 3.0f, 2.0f};
    const std::vector<float> expected(input, input + 3);
    Logits logits(input, 3);
    ThinkingBudgetTransform transform(3, 2);
    for (size_t generated = 0; generated < 3; ++generated) {
        transform.is_applicable(generated);
        transform.apply(logits);
        for (size_t i = 0; i < 3; ++i)
            EXPECT_EQ(logits.m_data[i], expected[i]) << "token " << i << " at step " << generated;
    }
}

TEST(ThinkingBudgetTransformTest, BudgetExhaustedForcesEndTokenOnDataPath) {
    float input[]{1.0f, 3.0f, 2.0f};
    Logits logits(input, 3);
    ThinkingBudgetTransform transform(3, 2);
    transform.is_applicable(3);
    transform.apply(logits);
    for (size_t i = 0; i < 3; ++i) {
        if (i == 2)
            EXPECT_EQ(logits.m_data[i], 1.0f);
        else
            EXPECT_EQ(logits.m_data[i], minus_inf);
    }
}

// The forced token must replace the candidate set even when upstream filters
// (top_k/top_p) had already dropped it from the truncated m_vector.
TEST(ThinkingBudgetTransformTest, BudgetExhaustedForcesEndTokenDroppedByFilters) {
    float input[]{1.0f, 3.0f, 2.0f};
    Logits logits(input, 3);
    // top_k=1 keeps only the most probable token (index 1), the end token (index 2) is dropped.
    TopKFilter(1).apply(logits);
    ASSERT_EQ(logits.m_vector.size(), 1u);
    EXPECT_EQ(logits.m_vector[0].m_index, 1);

    ThinkingBudgetTransform transform(3, 2);
    transform.is_applicable(3);
    transform.apply(logits);
    ASSERT_EQ(logits.m_size, 1u);
    ASSERT_EQ(logits.m_vector.size(), 1u);
    EXPECT_EQ(logits.m_vector[0].m_index, 2);
    EXPECT_FLOAT_EQ(logits.m_vector[0].m_log_prob, 1.0f);
}

TEST(ThinkingBudgetTransformTest, BudgetExhaustedForcesEndTokenOnInitializedVector) {
    float input[]{1.0f, 3.0f, 2.0f};
    Logits logits(input, 3);
    logits.initialize_vector();

    ThinkingBudgetTransform transform(3, 2);
    transform.is_applicable(3);
    transform.apply(logits);
    ASSERT_EQ(logits.m_size, 1u);
    ASSERT_EQ(logits.m_vector.size(), 1u);
    EXPECT_EQ(logits.m_vector[0].m_index, 2);
    EXPECT_FLOAT_EQ(logits.m_vector[0].m_log_prob, 1.0f);
}

// Once the end-of-reasoning token was generated naturally, the transform must
// stay inapplicable and never touch the logits again.
TEST(ThinkingBudgetTransformTest, NaturalClosureDisablesIntervention) {
    float input[]{1.0f, 3.0f, 2.0f};
    const std::vector<float> expected(input, input + 3);
    Logits logits(input, 3);
    ThinkingBudgetTransform transform(3, 2);

    transform.is_applicable(0);
    transform.accept_tokens({1, 2, 0});  // reasoning closed naturally at token 2
    EXPECT_FALSE(transform.is_applicable(1));
    // apply() runs only while the transform is applicable, as LogitProcessor gates it.
    if (transform.is_applicable(10)) {
        transform.apply(logits);
    }
    for (size_t i = 0; i < 3; ++i)
        EXPECT_EQ(logits.m_data[i], expected[i]) << "token " << i;
}

TEST(ThinkingBudgetTransformTest, IsApplicableStaysTrueWhileOpen) {
    ThinkingBudgetTransform transform(3, 2);
    for (size_t generated = 0; generated < 100; ++generated)
        EXPECT_TRUE(transform.is_applicable(generated));
    transform.accept_tokens({2});
    EXPECT_FALSE(transform.is_applicable(0));
}

TEST(LogitProcessorThinkingBudgetTest, GreedyForceCloseAfterBudget) {
    GenerationConfig config;
    config.reasoning_config = ReasoningConfig{2, 2};
    LogitProcessor processor(config, {});
    float input[]{1.0f, 3.0f, 2.0f};

    // Step 0: no tokens generated yet, logits untouched.
    processor.update_generated_len(0);
    {
        Logits logits(input, 3);
        processor.apply(logits);
    }
    EXPECT_EQ(input[1], 3.0f);
    processor.register_new_generated_token(1);

    // Step 1: still within the budget of 2.
    processor.update_generated_len(1);
    {
        Logits logits(input, 3);
        processor.apply(logits);
    }
    EXPECT_EQ(input[1], 3.0f);
    processor.register_new_generated_token(1);

    // Step 2: budget exhausted, only the end-of-reasoning token can be drawn.
    processor.update_generated_len(2);
    {
        Logits logits(input, 3);
        processor.apply(logits);
    }
    EXPECT_EQ(input[0], minus_inf);
    EXPECT_EQ(input[1], minus_inf);
    EXPECT_EQ(input[2], 1.0f);
}

TEST(LogitProcessorThinkingBudgetTest, NaturalClosureStopsIntervention) {
    GenerationConfig config;
    config.reasoning_config = ReasoningConfig{2, 2};
    LogitProcessor processor(config, {});
    float input[]{1.0f, 3.0f, 2.0f};

    // Step 0: the end-of-reasoning token is drawn naturally within the budget.
    processor.update_generated_len(0);
    {
        Logits logits(input, 3);
        processor.apply(logits);
    }
    EXPECT_EQ(input[1], 3.0f);
    processor.register_new_generated_token(2);

    // Past the budget the logits must stay untouched: reasoning closed on its own.
    processor.update_generated_len(5);
    {
        Logits logits(input, 3);
        processor.apply(logits);
    }
    EXPECT_EQ(input[0], 1.0f);
    EXPECT_EQ(input[1], 3.0f);
    EXPECT_EQ(input[2], 2.0f);
}

// Multinomial path: top_k truncates m_vector first, the budget transform then
// replaces the truncated candidate set with the end-of-reasoning token.
TEST(LogitProcessorThinkingBudgetTest, MultinomialForceCloseAfterBudget) {
    GenerationConfig config;
    config.do_sample = true;
    config.top_k = 1;
    config.temperature = 0.7f;
    config.reasoning_config = ReasoningConfig{2, 2};
    LogitProcessor processor(config, {});
    float input[]{1.0f, 3.0f, 2.0f};

    // Step 0: within budget, top_k keeps the best token (index 1).
    processor.update_generated_len(0);
    {
        Logits logits(input, 3);
        processor.apply(logits);
        ASSERT_TRUE(logits.is_vector_initialized());
        ASSERT_EQ(logits.m_size, 1u);
        EXPECT_EQ(logits.m_vector[0].m_index, 1);
    }
    processor.register_new_generated_token(1);

    // Step 1: still within budget.
    processor.update_generated_len(1);
    {
        Logits logits(input, 3);
        processor.apply(logits);
        ASSERT_EQ(logits.m_size, 1u);
        EXPECT_EQ(logits.m_vector[0].m_index, 1);
    }
    processor.register_new_generated_token(1);

    // Step 2: budget exhausted, the candidate set is replaced with the end token
    // even though top_k would have kept token 1.
    processor.update_generated_len(2);
    {
        Logits logits(input, 3);
        processor.apply(logits);
        ASSERT_EQ(logits.m_size, 1u);
        EXPECT_EQ(logits.m_vector[0].m_index, 2);
    }
}
