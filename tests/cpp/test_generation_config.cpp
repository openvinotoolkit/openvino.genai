// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/genai/generation_config.hpp"

using ov::genai::GenerationConfig;

TEST(GenerationConfig, SetEosTokenIdIgnoresNegativeDefault) {
    GenerationConfig config;
    config.max_new_tokens = 10;

    // Mirrors the pipelines: when neither the tokenizer nor the config defines EOS, the default is -1.
    const int64_t default_eos_token_id = -1;
    config.set_eos_token_id(default_eos_token_id);

    EXPECT_EQ(config.eos_token_id, -1);
    EXPECT_TRUE(config.stop_token_ids.empty());
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfig, SetEosTokenIdAddsStopToken) {
    GenerationConfig config;
    config.max_new_tokens = 10;

    config.set_eos_token_id(2);

    EXPECT_EQ(config.eos_token_id, 2);
    EXPECT_EQ(config.stop_token_ids.count(2), 1u);
    EXPECT_NO_THROW(config.validate());
}
