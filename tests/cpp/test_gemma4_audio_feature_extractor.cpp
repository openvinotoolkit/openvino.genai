// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>

#include "visual_language/gemma4/audio_feature_extractor.hpp"

namespace {

constexpr float PI = 3.14159265358979323846f;

ov::Tensor make_sine_wave(size_t samples, float frequency) {
    ov::Tensor audio(ov::element::f32, {samples});
    float* data = audio.data<float>();
    for (size_t sample = 0; sample < samples; ++sample) {
        data[sample] = std::sin(2.0f * PI * frequency * static_cast<float>(sample) / 16'000.0f);
    }
    return audio;
}

}  // namespace

TEST(Gemma4AudioFeatureExtractor, MatchesTransformersReference) {
    const ov::Tensor audio = make_sine_wave(1'000, 440.0f);
    const ov::genai::Gemma4AudioFeatureExtractor extractor;
    const ov::genai::Gemma4AudioFeatures features = extractor.extract(audio);

    EXPECT_EQ(features.input_features.get_shape(), (ov::Shape{1, 6, 128}));
    EXPECT_EQ(features.input_features_mask.get_shape(), (ov::Shape{1, 6}));
    const bool* mask = features.input_features_mask.data<const bool>();
    for (size_t frame = 0; frame < 6; ++frame) {
        EXPECT_TRUE(mask[frame]);
    }

    const float* data = features.input_features.data<const float>();
    const auto value = [data](size_t frame, size_t mel_bin) {
        return data[frame * 128 + mel_bin];
    };
    EXPECT_NEAR(value(0, 0), -6.9077553749f, 1e-4f);
    EXPECT_NEAR(value(0, 10), 0.7423009276f, 1e-4f);
    EXPECT_NEAR(value(1, 0), -6.9077553749f, 1e-4f);
    EXPECT_NEAR(value(1, 10), -3.2464833260f, 1e-4f);
    EXPECT_NEAR(value(3, 20), 0.3574853539f, 1e-4f);
    EXPECT_NEAR(value(5, 35), -2.0829672813f, 1e-4f);
}

TEST(Gemma4AudioFeatureExtractor, RejectsInvalidInputs) {
    const ov::genai::Gemma4AudioFeatureExtractor extractor;
    EXPECT_THROW(extractor.extract(ov::Tensor(ov::element::i16, {1'000})), ov::Exception);
    EXPECT_THROW(extractor.extract(ov::Tensor(ov::element::f32, {1, 1'000})), ov::Exception);
    EXPECT_THROW(extractor.extract(ov::Tensor(ov::element::f32, {160})), ov::Exception);
}

TEST(Gemma4AudioFeatureExtractor, PreservesPartialFrameMask) {
    const ov::Tensor audio = make_sine_wave(321, 440.0f);
    const ov::genai::Gemma4AudioFeatureExtractor extractor;
    const ov::genai::Gemma4AudioFeatures features = extractor.extract(audio);

    EXPECT_EQ(features.input_features.get_shape(), (ov::Shape{1, 2, 128}));
    const bool* mask = features.input_features_mask.data<const bool>();
    EXPECT_TRUE(mask[0]);
    EXPECT_TRUE(mask[1]);
}
