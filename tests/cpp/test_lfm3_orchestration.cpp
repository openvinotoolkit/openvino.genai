// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>

#include "openvino/opsets/opset15.hpp"
#include "utils.hpp"
#include "visual_language/audio_encoder.hpp"
#include "visual_language/lfm3/classes.hpp"
#include "visual_language/lfm3/preprocessing.hpp"

namespace {
TEST(LFM3VisionValidation, RejectsMalformedImagesAndLaterVideoFramesBeforeInference) {
    struct Encoder : ov::genai::VisionEncoderLFM3 {
        Encoder() {
            m_video_processor_config.max_frames = 8;
        }
    } encoder;
    const auto rejects = [](auto&& encode) {
        try {
            encode();
            FAIL() << "Expected RGB validation before preprocessing or inference";
        } catch (const ov::Exception& error) {
            EXPECT_NE(std::string(error.what()).find("uint8 RGB tensor"), std::string::npos);
        }
    };
    for (const auto& shape : std::vector<ov::Shape>{{1, 128, 128, 1},
                                                    {1, 128, 128, 4},
                                                    {2, 128, 128, 3},
                                                    {128, 128, 3},
                                                    {1, 0, 128, 3},
                                                    {1, 128, 0, 3}}) {
        ov::Tensor invalid(ov::element::u8, shape);
        rejects([&] {
            encoder.encode(invalid);
        });
    }
    ov::Tensor invalid_type(ov::element::f32, {1, 128, 128, 3});
    rejects([&] {
        encoder.encode(invalid_type);
    });
    ov::Tensor valid(ov::element::u8, {1, 128, 128, 3});
    ov::Tensor invalid_frame(ov::element::u8, {1, 128, 128, 1});
    rejects([&] {
        encoder.encode_frames({valid, invalid_frame});
    });
    rejects([&] {
        encoder.encode_frames({valid, valid, valid, invalid_frame});
    });
}

ov::Tensor read(const std::string& name, const ov::Shape& shape) {
    ov::Tensor value(ov::element::f32, shape);
    auto directory = std::filesystem::current_path() / "data" / "lfm3";
#ifdef LFM3_TEST_DATA_DIR
    if (std::filesystem::exists(std::filesystem::path(LFM3_TEST_DATA_DIR) / name))
        directory = LFM3_TEST_DATA_DIR;
#endif
    if (!std::filesystem::exists(directory / name))
        directory = std::filesystem::current_path() / "tests" / "data" / "lfm3";
    std::ifstream stream(directory / name, std::ios::binary);
    if (!stream.read(reinterpret_cast<char*>(value.data()), value.get_byte_size()))
        throw std::runtime_error("Missing LFM3 publisher reference: " + name);
    return value;
}
void compare(const ov::Tensor& actual, const ov::Tensor& expected, float tolerance) {
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    for (size_t i = 0; i < actual.get_size(); ++i)
        ASSERT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], tolerance) << "element " << i;
}
void compare_features(const ov::Tensor& actual, const ov::Tensor& expected) {
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    double squared_error = 0., reference_energy = 0., max_abs = 0.;
    for (size_t i = 0; i < actual.get_size(); ++i) {
        const auto value = actual.data<const float>()[i], reference = expected.data<const float>()[i];
        ASSERT_TRUE(std::isfinite(value) && std::isfinite(reference)) << "element " << i;
        const double difference = double(value) - reference;
        squared_error += difference * difference;
        reference_energy += double(reference) * reference;
        max_abs = std::max(max_abs, std::abs(difference));
    }
    ASSERT_GT(reference_energy, 0.);
    const auto nmse = squared_error / reference_energy;
    std::ostringstream metrics;
    metrics << "nmse=" << nmse << ",max_abs=" << max_abs;
    testing::Test::RecordProperty("numerical_error", metrics.str());
    // Log scaling amplifies FFT rounding in the two-tone fixture's near-zero bins.
    ASSERT_LE(nmse, 1e-5) << metrics.str();
}
ov::InferRequest inverse_fft() {
    using namespace ov::opset15;
    auto input = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1, 641, 2});
    auto output = std::make_shared<IRDFT>(input,
                                          Constant::create(ov::element::i64, {1}, {1}),
                                          Constant::create(ov::element::i64, {1}, {1280}));
    return ov::genai::utils::singleton_core()
        .compile_model(std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input}),
                       "CPU",
                       {{ov::hint::inference_precision.name(), ov::element::f32}})
        .create_infer_request();
}
class LFM3Features : public testing::TestWithParam<size_t> {};
TEST_P(LFM3Features, MatchesPublisherFeatureExtractorAcrossStreamingChunks) {
    using namespace ov::opset15;
    const auto samples = GetParam();
    const auto padded = samples <= 400 ? 400 : samples + (160 - (samples - 400) % 160) % 160;
    const auto frames = 1 + (padded - 400) / 160;
    auto input = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 1, 128, -1});
    input->output(0).set_names({"features"});
    auto transposed = std::make_shared<Transpose>(input, Constant::create(ov::element::i64, {4}, {0, 1, 3, 2}));
    auto output = std::make_shared<Reshape>(transposed, Constant::create(ov::element::i64, {3}, {1, -1, 128}), false);
    output->output(0).set_names({"audio_features"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
    auto encoder = ov::genai::create_audio_encoder(model, {ov::genai::AudioEncoderType::LFM3, 128}, "CPU", {});
    auto waveform = read("waveform-" + std::to_string(samples) + ".bin", {samples});
    const auto chunks = encoder(waveform);
    ASSERT_EQ(chunks.size(), (frames + 255) / 256);
    ov::Tensor merged(ov::element::f32, {1, frames, 128});
    auto* destination = merged.data<float>();
    for (const auto& chunk : chunks) {
        std::copy_n(chunk.data<const float>(), chunk.get_size(), destination);
        destination += chunk.get_size();
    }
    const auto reference = read("features-" + std::to_string(samples) + ".bin", {1, frames, 128});
    compare_features(merged, reference);
    if (samples == 1)
        compare(merged, reference, 2e-5f);
    auto repeated = encoder(waveform);
    ASSERT_EQ(repeated.size(), chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i)
        compare(repeated[i], chunks[i], 0.f);
}
INSTANTIATE_TEST_SUITE_P(PublisherReference, LFM3Features, testing::Values(1, 400, 1281, 16000, 48001));

class LFM3Waveform : public testing::TestWithParam<size_t> {};
TEST_P(LFM3Waveform, WholeAndChunkedISTFTMatchPublisherAndFlushExactlyOnce) {
    const auto frames = GetParam();
    auto window = read("istft-window.bin", {1280});
    std::vector<float> values(window.data<const float>(), window.data<const float>() + 1280);
    const auto spectral = read("spectral-" + std::to_string(frames) + ".bin", {1, frames, 1282});
    const auto expected = read("istft-" + std::to_string(frames) + ".bin", {1, frames * 320});
    auto inverse = inverse_fft();
    ov::genai::LFM3ISTFT stream(values);
    compare(stream.append(spectral, inverse, true), expected, 1e-6f);
    EXPECT_THROW(stream.flush(), ov::Exception);
    EXPECT_THROW(stream.append(spectral, inverse), ov::Exception);
    stream.reset();
    std::vector<ov::Tensor> chunks;
    for (size_t start = 0; start < frames; start += 5) {
        const auto length = std::min(size_t(5), frames - start);
        ov::Tensor view(ov::element::f32,
                        {1, length, 1282},
                        const_cast<float*>(spectral.data<const float>() + start * 1282));
        chunks.push_back(stream.append(view, inverse));
    }
    chunks.push_back(stream.flush());
    size_t count = 0;
    for (const auto& chunk : chunks)
        count += chunk.get_size();
    ASSERT_EQ(count, frames * 320);
    ov::Tensor merged(ov::element::f32, {1, count});
    auto* destination = merged.data<float>();
    for (const auto& chunk : chunks) {
        std::copy_n(chunk.data<const float>(), chunk.get_size(), destination);
        destination += chunk.get_size();
    }
    compare(merged, expected, 1e-6f);
    stream.reset();
    compare(stream.append(spectral, inverse, true), expected, 1e-6f);
}
INSTANTIATE_TEST_SUITE_P(PublisherReference, LFM3Waveform, testing::Values(1, 6, 18, 37));

TEST(LFM3WaveformValidation, RejectsMalformedSpectrumAndSupportsEmptyFlush) {
    auto window = read("istft-window.bin", {1280});
    ov::genai::LFM3ISTFT stream({window.data<const float>(), window.data<const float>() + 1280});
    auto inverse = inverse_fft();
    EXPECT_EQ(stream.flush().get_size(), 0);
    stream.reset();
    ov::Tensor invalid(ov::element::f32, {1, 3, 1281});
    EXPECT_THROW(stream.append(invalid, inverse), ov::Exception);
    EXPECT_EQ(stream.flush().get_size(), 0);
}
}  // namespace
