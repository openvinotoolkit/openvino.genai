// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <openvino/core/except.hpp>
#include "openvino/genai/generation_config.hpp"


using namespace ov::genai;

TEST(GenerationConfigTest, invalid_temperature) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.temperature = -0.1;
    config.do_sample = true;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_temperature) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.temperature = 0.1;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_top_p) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.top_p = -0.5;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.top_p = 1.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_top_p) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.top_p = 0.1;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_repeatition_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.repetition_penalty = -3.0;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.repetition_penalty = -0.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_repeatition_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.repetition_penalty = 1.8;
    EXPECT_NO_THROW(config.validate());
    config.repetition_penalty = 0.1;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_presence_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.presence_penalty = 3.0;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.presence_penalty = -3.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_presence_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.presence_penalty = 1.8;
    EXPECT_NO_THROW(config.validate());
    config.presence_penalty = -2.0;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_frequency_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.frequency_penalty = 3.0;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.frequency_penalty = -3.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_frequency_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.frequency_penalty = 1.8;
    EXPECT_NO_THROW(config.validate());
    config.frequency_penalty = -2.0;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_static_spec_decoding) {
    GenerationConfig config = speculative_decoding_greedy();
    config.num_assistant_tokens_schedule = NumAssistatantTokensScheduleType::CONSTANT;
    config.num_assistant_tokens = 0;
    config.assistant_confidence_threshold = 0.5;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.num_assistant_tokens_schedule = NumAssistatantTokensScheduleType::HEURISTIC;
    config.num_assistant_tokens = 5;
    config.assistant_confidence_threshold = 0;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_static_spec_decoding) {
    GenerationConfig config = speculative_decoding_greedy();
    config.num_assistant_tokens_schedule = NumAssistatantTokensScheduleType::CONSTANT;
    config.num_assistant_tokens = 5;
    config.assistant_confidence_threshold = 0;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_dynamic_spec_decoding) {
    GenerationConfig config = speculative_decoding_greedy();
    config.num_assistant_tokens_schedule = NumAssistatantTokensScheduleType::HEURISTIC;
    config.num_assistant_tokens = 5;
    config.assistant_confidence_threshold = 0.5;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.num_assistant_tokens_schedule = NumAssistatantTokensScheduleType::CONSTANT;
    config.num_assistant_tokens = 0;
    config.assistant_confidence_threshold = 0.5;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_dynamic_spec_decoding) {
    GenerationConfig config = speculative_decoding_greedy();
    config.num_assistant_tokens_schedule = NumAssistatantTokensScheduleType::HEURISTIC;
    config.assistant_confidence_threshold = 0.5;
    config.num_assistant_tokens = 0;
    EXPECT_NO_THROW(config.validate());
}
