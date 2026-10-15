// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "continuous_batching/generate_properties.hpp"

using ov::genai::CBGenerateProperties;

TEST(CBGenerateProperties, NoMediaIsNotMedia) {
    EXPECT_FALSE(CBGenerateProperties{}.has_media_properties());
}

TEST(CBGenerateProperties, AudioOnlyCountsAsMedia) {
    CBGenerateProperties properties;
    properties.audios_batches = std::vector<std::vector<ov::Tensor>>{{}};
    EXPECT_TRUE(properties.has_media_properties());
}
