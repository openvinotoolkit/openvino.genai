// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "whisper/feature_extractor.hpp"

namespace {

constexpr float LOG_FLOOR = -12.0f;
constexpr double PI = 3.14159265358979323846;

ov::genai::WhisperFeatureExtractor make_lfm3_extractor() {
    // LFM3-Omni processor_config.json feature_extractor values.
    return ov::genai::WhisperFeatureExtractor(128, 16000, 400, 160);
}

std::vector<float> make_sine_wave(size_t samples) {
    std::vector<float> audio(samples);
    for (size_t sample = 0; sample < samples; ++sample) {
        audio[sample] = static_cast<float>(0.5 * std::sin(2.0 * PI * 440.0 * static_cast<double>(sample) / 16000.0));
    }
    return audio;
}

}  // namespace

TEST(LFM3AudioFeatureExtractor, MatchesTransformersReference) {
    const auto features = make_lfm3_extractor().extract_uncentered(make_sine_wave(16000), LOG_FLOOR);
    ASSERT_EQ(features.feature_size, 128u);
    ASSERT_EQ(features.n_frames, 99u);
    const auto value = [&features](size_t mel_bin, size_t frame) {
        return features.data[mel_bin * features.n_frames + frame];
    };
    // Lfm3OmniFeatureExtractor output for the same input, before the (x + 4) / 4 normalization.
    EXPECT_NEAR(value(0, 0), -12.0f, 4e-4f);
    EXPECT_NEAR(value(20, 5), 1.1495723724f, 4e-4f);
    EXPECT_NEAR(value(64, 98), -3.8719534872f, 4e-4f);
    EXPECT_NEAR(value(127, 98), -5.8539175988f, 4e-4f);
}

class LFM3AudioFeatureFrames : public ::testing::TestWithParam<std::pair<size_t, size_t>> {};

TEST_P(LFM3AudioFeatureFrames, ZeroPadsTheLastWindow) {
    const auto [samples, expected_frames] = GetParam();
    const auto features = make_lfm3_extractor().extract_uncentered(std::vector<float>(samples, 0.0f), LOG_FLOOR);
    EXPECT_EQ(features.n_frames, expected_frames);
    for (float value : features.data) {
        EXPECT_FLOAT_EQ(value, LOG_FLOOR);
    }
}

INSTANTIATE_TEST_SUITE_P(WindowEdges,
                         LFM3AudioFeatureFrames,
                         ::testing::Values(std::make_pair(1, 1),
                                           std::make_pair(400, 1),
                                           std::make_pair(401, 2),
                                           std::make_pair(560, 2),
                                           std::make_pair(561, 3)));

TEST(LFM3AudioFeatureExtractor, LastWindowMatchesExplicitZeroPadding) {
    // 401 samples need two frames; the second window reads 241 real samples and 159 zeros.
    const auto extractor = make_lfm3_extractor();
    const auto audio = make_sine_wave(401);
    auto padded = audio;
    padded.resize(560, 0.0f);
    const auto features = extractor.extract_uncentered(audio, LOG_FLOOR);
    const auto reference = extractor.extract_uncentered(padded, LOG_FLOOR);
    ASSERT_EQ(features.n_frames, 2u);
    ASSERT_EQ(reference.n_frames, 2u);
    EXPECT_EQ(features.data, reference.data);
}

TEST(LFM3AudioFeatureExtractor, RejectsEmptyAudio) {
    EXPECT_THROW(make_lfm3_extractor().extract_uncentered({}, LOG_FLOOR), ov::Exception);
}
