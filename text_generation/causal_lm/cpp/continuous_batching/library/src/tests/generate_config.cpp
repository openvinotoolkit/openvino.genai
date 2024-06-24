// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <openvino/core/except.hpp>
#include "generation_config.hpp"

TEST(GenerationConfigTest, invalid_temperature) {
    GenerationConfig config;
    config.temperature = -0.1;
    config.do_sample = true;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_temperature) {
    GenerationConfig config;
    config.do_sample = true;
    config.temperature = 0.1;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_top_p) {
    GenerationConfig config;
    config.do_sample = true;
    config.top_p = -0.5;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.top_p = 1.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_top_p) {
    GenerationConfig config;
    config.do_sample = true;
    config.top_p = 0.1;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_repeatition_penalty) {
    GenerationConfig config;
    config.do_sample = true;
    config.repetition_penalty = -3.0;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.repetition_penalty = -0.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_repeatition_penalty) {
    GenerationConfig config;
    config.do_sample = true;
    config.repetition_penalty = 1.8;
    EXPECT_NO_THROW(config.validate());
    config.repetition_penalty = 0.0;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_presence_penalty) {
    GenerationConfig config;
    config.do_sample = true;
    config.presence_penalty = 3.0;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.presence_penalty = -3.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_presence_penalty) {
    GenerationConfig config;
    config.do_sample = true;
    config.presence_penalty = 1.8;
    EXPECT_NO_THROW(config.validate());
    config.presence_penalty = -2.0;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_frequence_penalty) {
    GenerationConfig config;
    config.do_sample = true;
    config.frequence_penalty = 3.0;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.frequence_penalty = -3.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_frequence_penalty) {
    GenerationConfig config;
    config.do_sample = true;
    config.frequence_penalty = 1.8;
    EXPECT_NO_THROW(config.validate());
    config.frequence_penalty = -2.0;
    EXPECT_NO_THROW(config.validate());
}
