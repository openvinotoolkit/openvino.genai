// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "visual_language/vision_token_pruning_processor.hpp"

using ov::genai::VisionTokenPruningProcessor;

TEST(VisionTokenPruningProcessor, ZeroPruningRatioDoesNotCreatePruner) {
    VisionTokenPruningProcessor processor("CPU");
    auto config = processor.get_config();
    config.pruning_ratio = 0;
    processor.set_config(config);
    EXPECT_FALSE(processor.is_available());
    EXPECT_EQ(processor.get_config().pruning_ratio, 0u);
}

TEST(VisionTokenPruningProcessor, NonZeroPruningRatioCreatesPruner) {
    VisionTokenPruningProcessor processor("CPU");
    auto config = processor.get_config();
    config.pruning_ratio = 50;
    processor.set_config(config);
    EXPECT_TRUE(processor.is_available());
    EXPECT_EQ(processor.get_config().pruning_ratio, 50u);
}

TEST(VisionTokenPruningProcessor, PrunerSurvivesReturnToZeroRatio) {
    VisionTokenPruningProcessor processor("CPU");
    auto config = processor.get_config();
    config.pruning_ratio = 50;
    processor.set_config(config);
    config.pruning_ratio = 0;
    processor.set_config(config);
    EXPECT_EQ(processor.get_config().pruning_ratio, 0u);
}
