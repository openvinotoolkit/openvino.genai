// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "automatic_speech_recognition/models/fun-asr/feature_extractor.hpp"

using ov::genai::FunASRFeatureExtractor;

TEST(FunASRFeatureExtractor, ProducesLfrFeatures) {
    const std::vector<float> audio(FunASRFeatureExtractor::sampling_rate, 0.0f);
    const ov::Tensor features = FunASRFeatureExtractor{}.extract(audio);

    EXPECT_EQ(features.get_shape(), (ov::Shape{1, 17, 560}));
    EXPECT_EQ(features.get_element_type(), ov::element::f32);
}

TEST(FunASRFeatureExtractor, MatchesKaldiFbankReference) {
    std::vector<float> audio(FunASRFeatureExtractor::sampling_rate);
    uint32_t state = 0x12345678;
    for (size_t index = 0; index < audio.size(); ++index) {
        state = state * 1664525 + 1013904223;
        audio[index] = static_cast<float>(static_cast<int>((state >> 8) & 0xffff) - 32768) / 32768.0f;
    }

    const ov::Tensor features = FunASRFeatureExtractor{}.extract(audio);
    const float* data = features.data<const float>();
    const size_t row_size = features.get_shape().at(2);

    EXPECT_NEAR(data[0], 17.6947365f, 0.01f);
    EXPECT_NEAR(data[1], 16.8778381f, 0.01f);
    EXPECT_NEAR(data[40], 24.3819084f, 0.01f);
    EXPECT_NEAR(data[79], 28.3560619f, 0.01f);
    EXPECT_NEAR(data[row_size], 17.3301449f, 0.01f);
    EXPECT_NEAR(data[row_size + 80], 17.326931f, 0.01f);
    EXPECT_NEAR(data[16 * row_size + 559], 27.6938591f, 0.01f);
}

TEST(FunASRFeatureExtractor, RejectsEmptyAudio) {
    EXPECT_THROW(FunASRFeatureExtractor{}.extract({}), ov::Exception);
}

TEST(FunASRFeatureExtractor, RejectsOneSampleAudio) {
    EXPECT_THROW(FunASRFeatureExtractor{}.extract({0.5f}), ov::Exception);
}
