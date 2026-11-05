// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "automatic_speech_recognition/models/qwen3-asr/audio_chunk.hpp"
#include "automatic_speech_recognition/models/qwen3-asr/encoder_decoder/encoder.hpp"
#include "automatic_speech_recognition/models/qwen3-asr/encoder_decoder/npu/encoder.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

namespace {

class Qwen3ASRStaticEncoderTest : public ::testing::TestWithParam<size_t> {
protected:
    std::filesystem::path m_model_dir;
    bool m_owns_directory = false;

    void SetUp() override {
        // NPUW:CPU needs the NPU plugin, but not physical NPU hardware.
        try {
            if (ov::Core().get_versions("NPU").empty()) {
                GTEST_SKIP() << "The NPU plugin is not registered.";
            }
        } catch (const ov::Exception& error) {
            if (std::string(error.what()).find("not registered") != std::string::npos) {
                GTEST_SKIP() << error.what();
            }
            throw;
        }
        m_model_dir =
            std::filesystem::temp_directory_path() /
            ("qwen3_asr_encoder_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        m_owns_directory = std::filesystem::create_directory(m_model_dir);
        ASSERT_TRUE(m_owns_directory);
        std::ofstream(m_model_dir / "config.json") << R"({"thinker_config":{"audio_config":{"n_window":2}}})";
        std::ofstream(m_model_dir / "preprocessor_config.json")
            << R"({"feature_size":8,"sampling_rate":16000,"hop_length":160,"n_fft":400})";

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});
        input->output(0).set_names({"input_features"});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(
            input,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2}),
            false);
        auto output = std::make_shared<ov::op::v0::Unsqueeze>(
            mean,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2}));
        output->output(0).set_names({"last_hidden_state"});
        auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
        ov::save_model(model, m_model_dir / "openvino_encoder_model.xml", false);
    }

    void TearDown() override {
        if (m_owns_directory) {
            std::filesystem::remove_all(m_model_dir);
        }
    }

    static ov::genai::WhisperFeatures make_features(size_t frames) {
        ov::genai::WhisperFeatures features{};
        features.feature_size = 8;
        features.n_frames = frames;
        features.n_active_frames = frames;
        features.data.resize(features.feature_size * frames);
        for (size_t feature = 0; feature < features.feature_size; ++feature) {
            for (size_t frame = 0; frame < frames; ++frame) {
                features.data[feature * frames + frame] = static_cast<float>(frame + 1);
            }
        }
        return features;
    }

    static ov::AnyMap cpu_fallback() {
        return {{"NPU_USE_NPUW", "YES"}, {"NPUW_DEVICES", "CPU"}, {"NPUW_ONLINE_PIPELINE", "NONE"}};
    }
};

TEST_P(Qwen3ASRStaticEncoderTest, FullPartialAndRepeatedChunksMatchCPU) {
    const auto frames = GetParam();
    const auto features = make_features(frames);
    ov::genai::Qwen3ASREncoder cpu(m_model_dir, "CPU", {});
    std::unique_ptr<ov::genai::Qwen3ASREncoder> npu =
        std::make_unique<ov::genai::Qwen3ASREncoderNPU>(m_model_dir, cpu_fallback());
    const auto expected = cpu.encode(features);
    const auto first = npu->encode(features);
    const auto repeated = npu->encode(features);

    const ov::Shape expected_shape{1, (frames + 3) / 4, 1};
    ASSERT_EQ(expected.get_shape(), expected_shape);
    ASSERT_EQ(first.get_shape(), expected_shape);
    ASSERT_EQ(repeated.get_shape(), expected_shape);
    for (size_t index = 0; index < expected.get_size(); ++index) {
        EXPECT_FLOAT_EQ(first.data<const float>()[index], expected.data<const float>()[index]);
        EXPECT_FLOAT_EQ(repeated.data<const float>()[index], expected.data<const float>()[index]);
    }
}

TEST_F(Qwen3ASRStaticEncoderTest, EncoderCapacityIncludesBoundarySearchMargin) {
    ov::genai::Qwen3ASREncoderNPU encoder(m_model_dir, cpu_fallback());
    constexpr size_t sampling_rate = 16000;
    constexpr size_t hop_length = 160;
    std::vector<float> audio(66 * sampling_rate, 0.01f);
    std::fill(audio.begin() + 62 * sampling_rate, audio.begin() + 63 * sampling_rate, 0.0f);
    const auto chunks = ov::genai::split_audio_into_chunks({audio}, sampling_rate, 60);
    ASSERT_EQ(chunks.size(), 2u);
    ASSERT_GT(chunks.front().wav.size(), 60 * sampling_rate);
    const auto capacity =
        encoder.get_max_encoder_len(60 + ov::genai::CHUNK_BOUNDARY_SEARCH_SECONDS, sampling_rate, hop_length);
    const auto nominal_capacity = encoder.get_max_encoder_len(60, sampling_rate, hop_length);
    EXPECT_EQ(capacity, 1625u);
    for (const auto& chunk : chunks) {
        ASSERT_LE(chunk.wav.size(), 65 * sampling_rate);
        const auto frames = chunk.wav.size() / hop_length;
        const auto output = encoder.encode(make_features(frames));
        EXPECT_EQ(output.get_shape(), (ov::Shape{1, (frames + 3) / 4, 1}));
        EXPECT_LE(output.get_shape()[1], capacity);
        if (chunk.chunk_index == 0) {
            EXPECT_GT(output.get_shape()[1], nominal_capacity);
        }
    }
}

TEST_F(Qwen3ASRStaticEncoderTest, SharedDecoderOptionsDoNotEnableLLMModeForEncoder) {
    auto properties = cpu_fallback();
    properties["NPUW_LLM"] = "YES";
    properties["MAX_PROMPT_LEN"] = 4096;
    properties["MIN_RESPONSE_LEN"] = 128;
    properties["NPUW_LLM_PREFILL_HINT"] = "DYNAMIC";
    properties["NPUW_LLM_PREFILL_CHUNK_SIZE"] = 1024;
    properties["++NPUW_LLM_GENERATE_CONFIG"] = ov::AnyMap{};
    properties["PREFILL_CONFIG"] = ov::AnyMap{};
    properties["BLOB_PATH"] = "decoder-only.blob";
    properties["EXPORT_BLOB"] = false;
    ov::genai::Qwen3ASREncoderNPU encoder(m_model_dir, properties);
    const auto output = encoder.encode(make_features(6));
    ASSERT_EQ(output.get_shape(), (ov::Shape{1, 2, 1}));
    EXPECT_FLOAT_EQ(output.data<const float>()[0], 2.5f);
    EXPECT_FLOAT_EQ(output.data<const float>()[1], 2.75f);
    EXPECT_EQ(properties.at("NPUW_LLM").as<std::string>(), "YES");
    EXPECT_EQ(properties.at("MAX_PROMPT_LEN").as<int>(), 4096);
}

INSTANTIATE_TEST_SUITE_P(ChunkLengths, Qwen3ASRStaticEncoderTest, ::testing::Values(1u, 4u, 6u, 8u));

}  // namespace
