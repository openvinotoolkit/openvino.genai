// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "automatic_speech_recognition/models/fun-asr/feature_extractor.hpp"
#include "automatic_speech_recognition/models/sensevoice/feature_extractor.hpp"

using ov::genai::FunASRFeatureExtractor;
using ov::genai::SenseVoiceSmallFeatureExtractor;

namespace {

constexpr size_t kFeatureSize = SenseVoiceSmallFeatureExtractor::feature_size;

std::vector<float> pseudo_random_audio() {
    std::vector<float> audio(FunASRFeatureExtractor::sampling_rate);
    uint32_t state = 0x12345678;
    for (size_t index = 0; index < audio.size(); ++index) {
        state = state * 1664525 + 1013904223;
        audio[index] = static_cast<float>(static_cast<int>((state >> 8) & 0xffff) - 32768) / 32768.0f;
    }
    return audio;
}

std::filesystem::path write_mvn(const std::string& name,
                                const std::vector<float>& shifts,
                                const std::vector<float>& scales) {
    const std::filesystem::path path = std::filesystem::path(testing::TempDir()) / name;
    std::ofstream out(path);
    const auto write_section = [&out](const std::string& tag, const std::vector<float>& values) {
        out << tag << " " << values.size() << " " << values.size() << "\n";
        out << "<LearnRateCoef> 0 [";
        for (const float value : values) {
            out << " " << value;
        }
        out << " ]\n";
    };
    write_section("<AddShift>", shifts);
    write_section("<Rescale>", scales);
    return path;
}

}  // namespace

TEST(FunASRFeatureExtractor, ProducesLfrFeatures) {
    const std::vector<float> audio(FunASRFeatureExtractor::sampling_rate, 0.0f);
    const ov::Tensor features = FunASRFeatureExtractor{}.extract(audio);

    EXPECT_EQ(features.get_shape(), (ov::Shape{1, 17, 560}));
    EXPECT_EQ(features.get_element_type(), ov::element::f32);
}

TEST(FunASRFeatureExtractor, MatchesKaldiFbankReference) {
    const std::vector<float> audio = pseudo_random_audio();

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

TEST(SenseVoiceSmallFeatureExtractor, IdentityCmvnPreservesFunAsrFeatures) {
    const auto mvn = write_mvn("identity.mvn",
                               std::vector<float>(kFeatureSize, 0.0f),
                               std::vector<float>(kFeatureSize, 1.0f));
    const std::vector<float> audio = pseudo_random_audio();

    const ov::Tensor fun = FunASRFeatureExtractor{}.extract(audio);
    const ov::Tensor sv = SenseVoiceSmallFeatureExtractor{mvn}.extract(audio);

    EXPECT_EQ(sv.get_shape(), (ov::Shape{1, 17, kFeatureSize}));
    EXPECT_EQ(sv.get_element_type(), ov::element::f32);

    ASSERT_EQ(fun.get_shape(), sv.get_shape());
    const float* fun_data = fun.data<const float>();
    const float* sv_data = sv.data<const float>();
    for (size_t index = 0; index < fun.get_size(); ++index) {
        EXPECT_FLOAT_EQ(sv_data[index], fun_data[index]);
    }
}

TEST(SenseVoiceSmallFeatureExtractor, AppliesCmvnMath) {
    std::vector<float> shifts(kFeatureSize);
    std::vector<float> scales(kFeatureSize);
    for (size_t dim = 0; dim < kFeatureSize; ++dim) {
        shifts[dim] = -2.0f + 0.01f * static_cast<float>(dim);
        scales[dim] = 0.5f + 1.0f / static_cast<float>(dim + 1);
    }
    const auto mvn = write_mvn("math.mvn", shifts, scales);
    const std::vector<float> audio = pseudo_random_audio();

    const ov::Tensor fun = FunASRFeatureExtractor{}.extract(audio);
    const ov::Tensor sv = SenseVoiceSmallFeatureExtractor{mvn}.extract(audio);

    const float* fun_data = fun.data<const float>();
    const float* sv_data = sv.data<const float>();
    const size_t frames = sv.get_shape()[1];
    for (const size_t frame : {size_t{0}, frames / 2, frames - 1}) {
        for (const size_t dim : {size_t{0}, size_t{1}, kFeatureSize / 2, kFeatureSize - 1}) {
            const size_t index = frame * kFeatureSize + dim;
            const float expected = (fun_data[index] + shifts[dim]) * scales[dim];
            EXPECT_NEAR(sv_data[index], expected, 1e-3f) << "frame=" << frame << " dim=" << dim;
        }
    }
}

TEST(SenseVoiceSmallFeatureExtractor, RejectsIncompatibleMvnDimension) {
    const auto mvn = write_mvn("wrongdim.mvn", std::vector<float>(100, 0.0f), std::vector<float>(100, 1.0f));
    EXPECT_THROW((SenseVoiceSmallFeatureExtractor{mvn}), ov::Exception);
}
