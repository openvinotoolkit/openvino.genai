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
#include "automatic_speech_recognition/models/qwen3-asr/split/audio_encoder.hpp"
#include "automatic_speech_recognition/models/qwen3-asr/split/npu/generation.hpp"
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
    }

    void TearDown() override {
        if (m_owns_directory) {
            std::filesystem::remove_all(m_model_dir);
        }
    }

    static ov::AnyMap cpu_fallback() {
        return {{"NPU_USE_NPUW", "YES"}, {"NPUW_DEVICES", "CPU"}, {"NPUW_ONLINE_PIPELINE", "NONE"}};
    }

    static ov::genai::VLMConfig split_config() {
        ov::genai::VLMConfig config;
        config.model_type = ov::genai::VLMModelType::QWEN3_ASR;
        config.hidden_size = 1;
        config.audio_config_num_mel_bins = 8;
        config.audio_config_n_window = 2;
        config.audio_config_n_window_infer = 8;
        return config;
    }

    static std::shared_ptr<ov::Model> split_model(bool use_length = false) {
        auto features = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8, -1});
        features->output(0).set_names({"padded_feature"});
        auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{-1, -1});
        mask->output(0).set_names({"padded_mask_after_cnn"});
        auto lengths = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
        lengths->output(0).set_names({"aftercnn_lens"});
        auto offsets = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        offsets->output(0).set_names({"cu_seqlens"});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(
            features, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
        auto first = std::make_shared<ov::op::v8::Gather>(
            mean,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        auto masked = std::make_shared<ov::op::v1::Multiply>(first, std::make_shared<ov::op::v0::Convert>(
                                                                       mask, ov::element::f32));
        ov::Output<ov::Node> values = masked;
        if (use_length) {
            values = std::make_shared<ov::op::v1::Add>(values, std::make_shared<ov::op::v0::Convert>(
                                                                  lengths, ov::element::f32));
        }
        auto output = std::make_shared<ov::op::v0::Unsqueeze>(
            values, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
        output->output(0).set_names({"audio_features"});
        return std::make_shared<ov::Model>(ov::OutputVector{output},
                                           ov::ParameterVector{features, mask, lengths, offsets});
    }
};

TEST_P(Qwen3ASRStaticEncoderTest, FullPartialAndRepeatedChunksMatchCPU) {
    const auto frames = GetParam();
    auto model = split_model();
    ov::genai::AudioEncoderQwen3ASR split_cpu(split_config(), model, m_model_dir, "CPU", {});
    auto split_npu =
        ov::genai::create_qwen3_asr_audio_encoder(split_config(), model, m_model_dir, "NPU", cpu_fallback());
    const size_t sample_count = std::max<size_t>(frames, 3) * 160;
    ov::Tensor audio(ov::element::f32, {sample_count});
    for (size_t i = 0; i < sample_count; ++i) {
        audio.data<float>()[i] = static_cast<float>(i % 31) / 31.0f;
    }
    const auto split_expected = split_cpu.encode({audio}).at(0);
    const auto split_first = split_npu->encode({audio}).at(0);
    const auto split_repeated = split_npu->encode({audio}).at(0);
    ASSERT_EQ(split_first.num_audio_tokens, split_expected.num_audio_tokens);
    ASSERT_EQ(split_first.audio_features.get_shape(), split_expected.audio_features.get_shape());
    for (size_t index = 0; index < split_expected.audio_features.get_size(); ++index) {
        EXPECT_FLOAT_EQ(split_first.audio_features.data<const float>()[index],
                        split_expected.audio_features.data<const float>()[index]);
        EXPECT_FLOAT_EQ(split_repeated.audio_features.data<const float>()[index],
                        split_expected.audio_features.data<const float>()[index]);
    }
}

TEST_F(Qwen3ASRStaticEncoderTest, EncoderCapacityIncludesBoundarySearchMargin) {
    auto encoder =
        ov::genai::create_qwen3_asr_audio_encoder(split_config(), split_model(), m_model_dir, "NPU", cpu_fallback());
    constexpr size_t sampling_rate = 16000;
    constexpr size_t hop_length = 160;
    std::vector<float> audio(66 * sampling_rate, 0.01f);
    std::fill(audio.begin() + 62 * sampling_rate, audio.begin() + 63 * sampling_rate, 0.0f);
    const auto chunks = ov::genai::split_audio_into_chunks({audio}, sampling_rate, 60);
    ASSERT_EQ(chunks.size(), 2u);
    ASSERT_GT(chunks.front().wav.size(), 60 * sampling_rate);
    const auto capacity = (65 * sampling_rate / hop_length + 3) / 4;
    const auto nominal_capacity = (60 * sampling_rate / hop_length + 3) / 4;
    EXPECT_EQ(capacity, 1625u);
    for (const auto& chunk : chunks) {
        ASSERT_LE(chunk.wav.size(), 65 * sampling_rate);
        const auto frames = chunk.wav.size() / hop_length;
        auto chunk_audio = ov::Tensor(ov::element::f32, {chunk.wav.size()}, chunk.wav.data());
        const auto output = encoder->encode({chunk_audio}).at(0);
        EXPECT_EQ(output.num_audio_tokens, (frames + 3) / 4);
        EXPECT_EQ(output.audio_features.get_shape(), (ov::Shape{output.num_audio_tokens, 1}));
        EXPECT_LE(output.num_audio_tokens, capacity);
        if (chunk.chunk_index == 0) {
            EXPECT_GT(output.num_audio_tokens, nominal_capacity);
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
    properties["WEIGHTS_PATH"] = "decoder-only.bin";
    properties["EXPORT_BLOB"] = false;
    EXPECT_EQ(properties.at("NPUW_LLM").as<std::string>(), "YES");
    EXPECT_EQ(properties.at("MAX_PROMPT_LEN").as<int>(), 4096);

    auto split_encoder =
        ov::genai::create_qwen3_asr_audio_encoder(split_config(), split_model(), m_model_dir, "NPU", properties);
    EXPECT_THROW(
        ov::genai::create_qwen3_asr_audio_encoder(split_config(), split_model(true), m_model_dir, "NPU", properties),
        ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(ChunkLengths, Qwen3ASRStaticEncoderTest, ::testing::Values(1u, 4u, 6u, 8u));

TEST(Qwen3ASRNPUGenerationTest, EnforcesCapacityAndCapsUnboundedRequests) {
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 65;
    EXPECT_FALSE(ov::genai::limit_qwen3_asr_npu_generation(128, config, 128, 191));
    EXPECT_EQ(config.max_new_tokens, 65u);
    config.max_new_tokens = 66;
    EXPECT_THROW(ov::genai::limit_qwen3_asr_npu_generation(128, config, 128, 191), ov::Exception);
    EXPECT_EQ(config.max_new_tokens, 66u);
    config.max_new_tokens = SIZE_MAX;
    EXPECT_TRUE(ov::genai::limit_qwen3_asr_npu_generation(128, config, 128, 191));
    EXPECT_EQ(config.max_new_tokens, 65u);
    config.min_new_tokens = 66;
    EXPECT_THROW(ov::genai::limit_qwen3_asr_npu_generation(128, config, 128, 191), ov::Exception);
}

TEST(Qwen3ASRNPUGenerationTest, RejectsInvalidPromptsAndBeamSearch) {
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 1;
    EXPECT_THROW(ov::genai::limit_qwen3_asr_npu_generation(0, config, 128, 191), ov::Exception);
    EXPECT_THROW(ov::genai::limit_qwen3_asr_npu_generation(129, config, 128, 191), ov::Exception);
    config.num_beams = 2;
    EXPECT_THROW(ov::genai::limit_qwen3_asr_npu_generation(2, config, 128, 191), ov::Exception);
}

}  // namespace
