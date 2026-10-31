// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "visual_language/qwen3_vl/classes.hpp"

namespace {

std::vector<size_t> sample(size_t total_frames, size_t num_frames) {
    ov::genai::VideoProcessorConfig config;
    config.do_sample_frames = true;
    config.num_frames = num_frames;
    ov::genai::EncodedVideo video;
    ov::genai::fill_video_metadata(video, total_frames, config);
    return video.metadata.frames_indices;
}

}  // namespace

// np.linspace(0, total - 1, n).round(): i * step with step = (total - 1) / (n - 1), halves rounded to even.
TEST(VideoFrameSampling, MatchesNumpyLinspaceRound) {
    const auto indices = sample(26, 23);
    ASSERT_EQ(indices.size(), 23u);
    EXPECT_EQ(indices[11], 13u);
    EXPECT_EQ(indices.front(), 0u);
    EXPECT_EQ(indices.back(), 25u);
}

TEST(VideoFrameSampling, CapsShortVideosAtTheirFrameCount) {
    ov::genai::VideoProcessorConfig config;
    config.do_sample_frames = true;
    config.fps = 2.0f;
    config.min_frames = 4;
    config.max_frames = 384;
    ov::genai::EncodedVideo video;
    video.metadata.fps = 2.0f;
    ov::genai::fill_video_metadata(video, 3, config);
    EXPECT_EQ(video.metadata.frames_indices, (std::vector<size_t>{0, 1, 2}));
}
