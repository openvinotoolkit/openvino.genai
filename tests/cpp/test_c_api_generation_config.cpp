// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/genai/c/generation_config.h"

class CApiGenerationConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(ov_genai_generation_config_create(&config), ov_status_e::OK);
        ASSERT_EQ(ov_genai_generation_config_set_max_new_tokens(config, 10), ov_status_e::OK);
    }

    void TearDown() override {
        ov_genai_generation_config_free(config);
    }

    ov_genai_generation_config* config = nullptr;
};

TEST_F(CApiGenerationConfigTest, SetEosTokenIdPassesValidation) {
    ASSERT_EQ(ov_genai_generation_config_set_eos_token_id(config, 2), ov_status_e::OK);
    EXPECT_EQ(ov_genai_generation_config_validate(config), ov_status_e::OK);
}

TEST_F(CApiGenerationConfigTest, ResetEosTokenIdPassesValidation) {
    ASSERT_EQ(ov_genai_generation_config_set_eos_token_id(config, 2), ov_status_e::OK);
    ASSERT_EQ(ov_genai_generation_config_set_eos_token_id(config, -1), ov_status_e::OK);
    EXPECT_EQ(ov_genai_generation_config_validate(config), ov_status_e::OK);
    ASSERT_EQ(ov_genai_generation_config_set_stop_token_ids(config, nullptr, 0), ov_status_e::OK);
    EXPECT_EQ(ov_genai_generation_config_validate(config), ov_status_e::OK);
}

TEST_F(CApiGenerationConfigTest, SetStopTokenIdsRejectsNullArray) {
    EXPECT_EQ(ov_genai_generation_config_set_stop_token_ids(config, nullptr, 2), ov_status_e::INVALID_C_PARAM);
    EXPECT_EQ(ov_genai_generation_config_set_stop_token_ids(config, nullptr, 0), ov_status_e::OK);
}
