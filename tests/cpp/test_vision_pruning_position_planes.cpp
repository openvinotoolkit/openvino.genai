// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "visual_language/vision_token_pruning_processor.hpp"

namespace {
using ov::genai::PositionPlaneLayout;
using ov::genai::VisionTokenPruningProcessor;

TEST(VisionPruningPositionPlanes, Qwen35CompactsLeadingTextPlane) {
    VisionTokenPruningProcessor processor("CPU");
    ov::Tensor input_ids(ov::element::i64, {1, 8});
    std::copy_n(std::array<int64_t, 8>{10, 11, 12, 12, 12, 12, 13, 14}.begin(), 8, input_ids.data<int64_t>());
    ov::Tensor original(ov::element::i64, {4, 1, 8});
    const std::array<std::array<int64_t, 8>, 4> planes{{
        {{0, 1, 2, 3, 4, 5, 6, 7}},
        {{0, 1, 2, 2, 2, 2, 4, 5}},
        {{0, 1, 2, 2, 3, 3, 4, 5}},
        {{0, 1, 2, 3, 2, 3, 4, 5}},
    }};
    for (size_t plane = 0; plane < 4; ++plane) {
        std::copy(planes[plane].begin(), planes[plane].end(), original.data<int64_t>() + plane * 8);
    }
    std::vector<std::vector<bool>> flags;
    ov::Tensor pruned = processor.update_position_ids_3d(original, input_ids, 11, 12, {{1, 2, 2}}, {{0, 2}}, 1,
                                                          flags, PositionPlaneLayout::TEXT_THW);
    const std::array<std::array<int64_t, 6>, 4> expected{{
        {{0, 1, 2, 3, 4, 5}},
        {{0, 1, 2, 2, 4, 5}},
        {{0, 1, 2, 3, 4, 5}},
        {{0, 1, 2, 2, 4, 5}},
    }};
    ASSERT_EQ(pruned.get_shape(), (ov::Shape{4, 1, 6}));
    for (size_t plane = 0; plane < expected.size(); ++plane) {
        for (size_t s = 0; s < expected[plane].size(); ++s) {
            EXPECT_EQ(pruned.data<const int64_t>()[plane * 6 + s], expected[plane][s])
                << "plane=" << plane << ", token=" << s;
        }
    }
    EXPECT_EQ(VisionTokenPruningProcessor::calculate_rope_delta(pruned, PositionPlaneLayout::TEXT_THW), 0);
    EXPECT_THROW(VisionTokenPruningProcessor::calculate_rope_delta(pruned, PositionPlaneLayout::THW), ov::Exception);
}

TEST(VisionPruningPositionPlanes, Qwen35KeepsOriginalOffsetsAcrossImageAndVideoFrames) {
    constexpr int64_t vision_start = 11;
    constexpr int64_t image_pad = 12;
    constexpr int64_t video_pad = 15;
    // Each video frame has its own vision marker and a timestamp token before it.
    const std::array<int64_t, 23> tokens{{
        10, vision_start, image_pad, image_pad, image_pad, image_pad, 13, 14,
        20, vision_start, video_pad, video_pad, video_pad, video_pad, 13,
        21, vision_start, video_pad, video_pad, video_pad, video_pad, 13, 14,
    }};
    ov::Tensor input_ids(ov::element::i64, {1, tokens.size()});
    std::copy(tokens.begin(), tokens.end(), input_ids.data<int64_t>());

    const std::array<std::array<int64_t, 23>, 4> original_planes{{
        {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}},
        {{0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 8, 8, 8, 8, 10, 11, 12, 13, 13, 13, 13, 15, 16}},
        {{0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 11, 12, 13, 13, 14, 14, 15, 16}},
        {{0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13, 14, 13, 14, 15, 16}},
    }};
    ov::Tensor original(ov::element::i64, {4, 1, tokens.size()});
    for (size_t plane = 0; plane < original_planes.size(); ++plane) {
        std::copy(original_planes[plane].begin(), original_planes[plane].end(),
                  original.data<int64_t>() + plane * tokens.size());
    }

    VisionTokenPruningProcessor processor("CPU");
    std::vector<std::vector<bool>> flags;
    // Pre-merge 4x4 grids yield 2x2 LLM-visible patches. Drop the final row or column.
    const ov::Tensor pruned = processor.update_position_ids_3d(
        original, input_ids, vision_start, image_pad,
        {{1, 4, 4}, {1, 4, 4}, {1, 4, 4}}, {{0}, {1}, {2}}, 2,
        flags, PositionPlaneLayout::TEXT_THW, video_pad);

    const std::array<std::array<int64_t, 14>, 4> expected{{
        {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}},
        {{0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16}},
        {{0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16}},
        {{0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16}},
    }};
    ASSERT_EQ(flags, (std::vector<std::vector<bool>>{{true, false, false, false},
                                                       {false, true, false, false},
                                                       {false, false, true, false}}));
    ASSERT_EQ(pruned.get_shape(), (ov::Shape{4, 1, 14}));
    for (size_t plane = 0; plane < expected.size(); ++plane) {
        for (size_t s = 0; s < expected[plane].size(); ++s) {
            EXPECT_EQ(pruned.data<const int64_t>()[plane * 14 + s], expected[plane][s])
                << "plane=" << plane << ", token=" << s;
        }
    }

    const int64_t rope_delta = VisionTokenPruningProcessor::calculate_rope_delta(pruned, PositionPlaneLayout::TEXT_THW);
    EXPECT_EQ(rope_delta, 3);
    // Qwen3.5's next generated token uses history_size for text and history_size + rope_delta for THW.
    // This verifies the pruning output consumed by generation; it does not construct a model embedder.
    const int64_t history_size = static_cast<int64_t>(pruned.get_shape().back());
    EXPECT_EQ(history_size, 14);
    EXPECT_EQ(history_size + rope_delta, 17);
}

TEST(VisionPruningPositionPlanes, Qwen35RopeDeltaIgnoresLongerTextPlane) {
    ov::Tensor position_ids(ov::element::i64, {4, 1, 5});
    const std::array<std::array<int64_t, 5>, 4> planes{{
        {{0, 1, 2, 3, 4}},
        {{0, 1, 2, 2, 3}},
        {{0, 1, 2, 2, 3}},
        {{0, 1, 2, 2, 3}},
    }};
    for (size_t plane = 0; plane < planes.size(); ++plane) {
        std::copy(planes[plane].begin(), planes[plane].end(), position_ids.data<int64_t>() + plane * 5);
    }
    EXPECT_EQ(VisionTokenPruningProcessor::calculate_rope_delta(position_ids, PositionPlaneLayout::TEXT_THW), -1);
}
}  // namespace
