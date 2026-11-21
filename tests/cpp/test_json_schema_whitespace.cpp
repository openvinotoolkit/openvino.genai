// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "sampling/structured_output/structured_output_controller.hpp"

using namespace ov::genai;

TEST(StructuredOutputConfigTest, MasksStructuralWhitespaceInGenericArrays) {
    if (!StructuredOutputController::get_backend_registry().count("xgrammar")) {
        GTEST_SKIP() << "XGrammar is disabled";
    }

    enum TokenId { OPEN_ARRAY, SPACE, TAB_NEWLINE, LONG_WHITESPACE, ITEM_AND_END_ARRAY, EOS };
    Tokenizer::TokenizerImpl tokenizer;
    tokenizer.m_vocab = {"[", " ", "\t\n", std::string(128, ' '), R"({"x":"a  b"}])", "<eos>"};
    tokenizer.m_eos_token_id = EOS;
    StructuredOutputController controller(tokenizer);
    StructuredOutputConfig config({json_schema(R"({"type":"array"})")});
    const std::optional<int> limits[] = {std::nullopt, 0, 1, 2};
    for (const auto limit : limits) {
        SCOPED_TRACE(limit.value_or(-1));
        config.max_whitespace_cnt = limit;
        GenerationConfig generation;
        generation.structured_output_config = config;
        generation.stop_token_ids = {tokenizer.m_eos_token_id};
        auto matcher = std::dynamic_pointer_cast<LogitTransformers::IStatefulLogitTransformer>(
            controller.get_logits_transformer(generation));
        ASSERT_TRUE(matcher);
        matcher->accept_tokens({OPEN_ARRAY});
        std::vector<float> scores(tokenizer.m_vocab.size(), 0.f);
        Logits logits(scores.data(), scores.size());
        matcher->apply(logits);
        EXPECT_EQ(std::isfinite(scores[SPACE]), limit != 0);
        EXPECT_EQ(std::isfinite(scores[TAB_NEWLINE]), !limit || *limit >= 2);
        EXPECT_EQ(std::isfinite(scores[LONG_WHITESPACE]), !limit.has_value());
        EXPECT_TRUE(std::isfinite(scores[ITEM_AND_END_ARRAY]));

        if (limit == 1) {
            matcher->accept_tokens({SPACE});
            std::fill(scores.begin(), scores.end(), 0.f);
            matcher->apply(logits);
            EXPECT_FALSE(std::isfinite(scores[SPACE]));
            EXPECT_TRUE(std::isfinite(scores[ITEM_AND_END_ARRAY]));
        }
        matcher->accept_tokens({ITEM_AND_END_ARRAY});
        std::fill(scores.begin(), scores.end(), 0.f);
        matcher->apply(logits);
        EXPECT_TRUE(std::isfinite(scores[EOS]));
    }
}
