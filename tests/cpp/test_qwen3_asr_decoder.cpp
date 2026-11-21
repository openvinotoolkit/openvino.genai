// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

#include "automatic_speech_recognition/models/qwen3-asr/encoder_decoder/decoder.hpp"
#include "automatic_speech_recognition/models/qwen3-asr/encoder_decoder/npu/decoder.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/openvino.hpp"

namespace {

class Qwen3ASRDecoderFormatTest : public ::testing::TestWithParam<std::string> {
protected:
    std::filesystem::path m_model_dir;
    bool m_owns_directory = false;

    void SetUp() override {
        m_model_dir =
            std::filesystem::temp_directory_path() /
            ("qwen3_asr_decoder_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        m_owns_directory = std::filesystem::create_directory(m_model_dir);
        ASSERT_TRUE(m_owns_directory);
    }

    void TearDown() override {
        if (m_owns_directory) {
            std::filesystem::remove_all(m_model_dir);
        }
    }

    void save_decoder(bool stateful, bool has_beam_idx) {
        auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
        input_ids->output(0).set_names({"input_ids"});
        auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1});
        hidden->output(0).set_names({"encoder_hidden_states"});
        ov::ParameterVector parameters{input_ids, hidden};
        if (has_beam_idx) {
            auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
            beam_idx->set_friendly_name("reorder_indices");
            beam_idx->output(0).set_names({"beam_idx"});
            parameters.push_back(beam_idx);
        }
        auto converted = std::make_shared<ov::op::v0::Convert>(input_ids, ov::element::f32);
        auto logits = std::make_shared<ov::op::v0::Unsqueeze>(
            converted,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
        logits->output(0).set_names({"logits"});
        ov::OutputVector outputs{logits};
        ov::SinkVector sinks;
        if (stateful) {
            auto variable = std::make_shared<ov::op::util::Variable>(
                ov::op::util::VariableInfo{ov::Shape{1, 1, 1}, ov::element::f32, "past_key_values.0.key"});
            auto initial = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {0.0f});
            auto read = std::make_shared<ov::op::v6::ReadValue>(initial, variable);
            auto updated = std::make_shared<ov::op::v1::Add>(read, hidden);
            sinks.push_back(std::make_shared<ov::op::v6::Assign>(updated, variable));
        } else {
            auto present = std::make_shared<ov::op::v1::Add>(
                hidden,
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}));
            present->output(0).set_names({"present.0.key"});
            outputs.push_back(present);
        }
        auto model = std::make_shared<ov::Model>(outputs, sinks, parameters);
        ov::save_model(model, m_model_dir / "openvino_decoder_model.xml", false);
    }

    void expect_format_error() {
        try {
            std::unique_ptr<ov::genai::Qwen3ASRDecoder> decoder;
            if (GetParam() == "NPU") {
                decoder = std::make_unique<ov::genai::Qwen3ASRDecoderNPU>(m_model_dir, ov::AnyMap{}, 128);
            } else {
                decoder = std::make_unique<ov::genai::Qwen3ASRDecoder>(m_model_dir, GetParam(), ov::AnyMap{});
            }
            FAIL() << "An incompatible decoder must be rejected before compilation.";
        } catch (const ov::Exception& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("must be a stateful model with KV-cache variables and a 'beam_idx' input"),
                      std::string::npos);
            EXPECT_NE(message.find("--task automatic-speech-recognition-with-past"), std::string::npos);
            EXPECT_NE(message.find(m_model_dir.string()), std::string::npos);
        }
    }
};

class CapacityLimitedCpuDecoder : public ov::genai::Qwen3ASRDecoder {
public:
    explicit CapacityLimitedCpuDecoder(const std::filesystem::path& models_path)
        : Qwen3ASRDecoder(models_path, "CPU", {}) {}

private:
    ov::genai::ASRGenerationConfig prepare_generation_config(
        size_t prompt_len,
        const ov::genai::ASRGenerationConfig& config) const override {
        return ov::genai::prepare_qwen3_asr_npu_generation(prompt_len, config, 2, 3);
    }
};

TEST_P(Qwen3ASRDecoderFormatTest, RejectsPrefillOnlyDecoder) {
    save_decoder(false, false);
    expect_format_error();
}

TEST_P(Qwen3ASRDecoderFormatTest, DummyBeamInputDoesNotReplaceKVCache) {
    save_decoder(false, true);
    expect_format_error();
}

TEST_P(Qwen3ASRDecoderFormatTest, RejectsStatefulDecoderWithoutBeamInput) {
    save_decoder(true, false);
    expect_format_error();
}

TEST_F(Qwen3ASRDecoderFormatTest, StatefulDecoderCompilesOnCPU) {
    save_decoder(true, true);
    EXPECT_NO_THROW(ov::genai::Qwen3ASRDecoder(m_model_dir, "CPU", {}));
}

TEST_F(Qwen3ASRDecoderFormatTest, SharedFunASRConstructorGeneratesOnCPU) {
    save_decoder(true, true);
    ov::genai::Qwen3ASRDecoder decoder(m_model_dir, "CPU", {});
    ov::Tensor input_ids(ov::element::i64, {1, 2});
    std::fill_n(input_ids.data<int64_t>(), input_ids.get_size(), 0);
    ov::Tensor encoder_hidden_states(ov::element::f32, {1, 1, 1});
    encoder_hidden_states.data<float>()[0] = 1.0f;
    ov::genai::ASRGenerationConfig config;
    config.set_eos_token_id(0);
    config.max_new_tokens = 1;
    ov::genai::RawPerfMetrics metrics;
    metrics.m_inference_durations = {{ov::genai::MicroSeconds(0.0f)}};
    ov::genai::ASRRawPerfMetrics asr_metrics;
    const auto result = decoder.generate(input_ids, encoder_hidden_states, config, metrics, asr_metrics);
    EXPECT_EQ(result.tokens.size(), 1u);
    EXPECT_EQ(asr_metrics.decode_inference_durations.size(), 1u);
}

TEST_F(Qwen3ASRDecoderFormatTest, NPUDecoderRequiresPositiveEncoderCapacity) {
    save_decoder(true, true);
    try {
        ov::genai::Qwen3ASRDecoderNPU decoder(m_model_dir, {}, 0);
        FAIL() << "A zero encoder capacity must be rejected.";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("NPUW_QWEN3_ASR_MAX_ENCODER_LEN must be > 0"), std::string::npos);
    }
}

TEST_F(Qwen3ASRDecoderFormatTest, RejectsOversizedFiniteBudgetBeforeInference) {
    save_decoder(true, true);
    CapacityLimitedCpuDecoder decoder(m_model_dir);
    ov::Tensor input_ids(ov::element::i64, {1, 2});
    std::fill_n(input_ids.data<int64_t>(), input_ids.get_size(), 0);
    ov::Tensor encoder_hidden_states(ov::element::f32, {1, 1, 1});
    encoder_hidden_states.data<float>()[0] = 1.0f;
    ov::genai::ASRGenerationConfig config;
    config.set_eos_token_id(0);
    config.ignore_eos = true;
    config.max_new_tokens = 4;
    ov::genai::RawPerfMetrics metrics;
    metrics.m_inference_durations = {{ov::genai::MicroSeconds(0.0f)}};
    ov::genai::ASRRawPerfMetrics asr_metrics;
    try {
        decoder.generate(input_ids, encoder_hidden_states, config, metrics, asr_metrics);
        FAIL() << "An explicit output budget beyond the KV capacity must be rejected.";
    } catch (const ov::Exception& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("requested 4 new tokens"), std::string::npos);
        EXPECT_NE(message.find("accommodate only 3"), std::string::npos);
    }
    EXPECT_TRUE(asr_metrics.decode_inference_durations.empty());
    EXPECT_EQ(config.max_new_tokens, 4u);

    config.max_new_tokens = 3;
    const auto result = decoder.generate(input_ids, encoder_hidden_states, config, metrics, asr_metrics);
    ASSERT_EQ(result.tokens.size(), 1u);
    EXPECT_EQ(result.tokens[0].size(), 3u);
    ASSERT_EQ(result.finish_reasons.size(), 1u);
    EXPECT_EQ(result.finish_reasons[0], ov::genai::GenerationFinishReason::LENGTH);
    EXPECT_EQ(asr_metrics.decode_inference_durations.size(), 3u);
}

TEST_F(Qwen3ASRDecoderFormatTest, UnboundedGenerationReturnsPartialAndPreservesEarlyEOS) {
    save_decoder(true, true);
    CapacityLimitedCpuDecoder decoder(m_model_dir);
    ov::Tensor input_ids(ov::element::i64, {1, 2});
    std::fill_n(input_ids.data<int64_t>(), input_ids.get_size(), 0);
    ov::Tensor encoder_hidden_states(ov::element::f32, {1, 1, 1});
    encoder_hidden_states.data<float>()[0] = 1.0f;
    ov::genai::ASRGenerationConfig config;
    config.set_eos_token_id(0);
    config.ignore_eos = true;
    ov::genai::RawPerfMetrics metrics;
    metrics.m_inference_durations = {{ov::genai::MicroSeconds(0.0f)}};
    ov::genai::ASRRawPerfMetrics asr_metrics;
    const auto bounded = decoder.generate(input_ids, encoder_hidden_states, config, metrics, asr_metrics);
    ASSERT_EQ(bounded.tokens.size(), 1u);
    EXPECT_EQ(bounded.tokens[0].size(), 3u);
    ASSERT_EQ(bounded.finish_reasons.size(), 1u);
    EXPECT_EQ(bounded.finish_reasons[0], ov::genai::GenerationFinishReason::LENGTH);

    config.ignore_eos = false;
    ov::genai::ASRRawPerfMetrics early_metrics;
    const auto early = decoder.generate(input_ids, encoder_hidden_states, config, metrics, early_metrics);
    ASSERT_EQ(early.tokens.size(), 1u);
    EXPECT_EQ(early.tokens[0].size(), 1u);
    ASSERT_EQ(early.finish_reasons.size(), 1u);
    EXPECT_EQ(early.finish_reasons[0], ov::genai::GenerationFinishReason::STOP);
}

TEST(Qwen3ASRNPUGenerationCapacityTest, AcceptsExactlyAvailableTokensAndRejectsOneMore) {
    ov::genai::ASRGenerationConfig config;
    config.ignore_eos = true;
    config.max_new_tokens = 65;
    EXPECT_EQ(ov::genai::prepare_qwen3_asr_npu_generation(128, config, 128, 191).max_new_tokens, 65u);
    config.max_new_tokens = 66;
    EXPECT_THROW(ov::genai::prepare_qwen3_asr_npu_generation(128, config, 128, 191), ov::Exception);
    EXPECT_EQ(config.max_new_tokens, 66u);
}

TEST(Qwen3ASRNPUGenerationCapacityTest, AppliesMaxLengthAndMaxNewTokensPrecedence) {
    ov::genai::ASRGenerationConfig config;
    config.max_length = 193;
    EXPECT_EQ(ov::genai::prepare_qwen3_asr_npu_generation(128, config, 128, 191).get_max_new_tokens(128), 65u);
    config.max_length = 194;
    EXPECT_THROW(ov::genai::prepare_qwen3_asr_npu_generation(128, config, 128, 191), ov::Exception);
    config.max_new_tokens = 1;
    EXPECT_EQ(ov::genai::prepare_qwen3_asr_npu_generation(128, config, 128, 191).get_max_new_tokens(128), 1u);
}

TEST(Qwen3ASRNPUGenerationCapacityTest, RejectsBeamSearchBeforeInference) {
    ov::genai::ASRGenerationConfig config;
    config.max_new_tokens = 1;
    config.num_beams = 2;
    try {
        ov::genai::prepare_qwen3_asr_npu_generation(2, config, 128, 191);
        FAIL() << "The NPU decoder only supports single-beam generation.";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("does not support beam search"), std::string::npos);
    }
}

TEST(Qwen3ASRNPUGenerationCapacityTest, CapsUnboundedGenerationAndRejectsInvalidInputs) {
    ov::genai::ASRGenerationConfig config;
    config.max_new_tokens = 1;
    EXPECT_THROW(ov::genai::prepare_qwen3_asr_npu_generation(0, config, 128, 191), ov::Exception);
    EXPECT_THROW(ov::genai::prepare_qwen3_asr_npu_generation(129, config, 128, 191), ov::Exception);
    config.max_new_tokens = SIZE_MAX;
    EXPECT_EQ(ov::genai::prepare_qwen3_asr_npu_generation(128, config, 128, 191).max_new_tokens, 65u);
    config.min_new_tokens = 66;
    EXPECT_THROW(ov::genai::prepare_qwen3_asr_npu_generation(128, config, 128, 191), ov::Exception);
}

TEST(Qwen3ASRNPUPropertiesTest, RejectsConflictingEncoderCapacity) {
    for (const ov::Any& value : {ov::Any{64}, ov::Any{int64_t{64}}, ov::Any{std::string{"64"}}}) {
        ov::AnyMap properties{{"NPUW_QWEN3_ASR_MAX_ENCODER_LEN", value}};
        EXPECT_THROW(ov::genai::prepare_qwen3_asr_npu_properties(properties, 128), ov::Exception);
    }
    for (const ov::Any& value : {ov::Any{128}, ov::Any{int64_t{128}}, ov::Any{std::string{"128"}}}) {
        ov::AnyMap properties{{"NPUW_QWEN3_ASR_MAX_ENCODER_LEN", value}};
        EXPECT_NO_THROW(ov::genai::prepare_qwen3_asr_npu_properties(properties, 128));
    }
    EXPECT_THROW(ov::genai::prepare_qwen3_asr_npu_properties(
                     {{"NPUW_QWEN3_ASR_MAX_ENCODER_LEN", "128extra"}}, 128),
                 ov::Exception);
}

TEST(Qwen3ASRNPUPropertiesTest, AddsDefaultPromptLengthOnlyForCompileOrExport) {
    const auto compiled = ov::genai::prepare_qwen3_asr_npu_properties({}, 128);
    EXPECT_EQ(compiled.at("MAX_PROMPT_LEN").as<int64_t>(), 256);
    const auto imported = ov::genai::prepare_qwen3_asr_npu_properties({{"BLOB_PATH", "model.blob"}}, 128);
    EXPECT_EQ(imported.count("MAX_PROMPT_LEN"), 0u);
    const auto exported =
        ov::genai::prepare_qwen3_asr_npu_properties({{"BLOB_PATH", "model.blob"}, {"EXPORT_BLOB", true}}, 128);
    EXPECT_EQ(exported.at("MAX_PROMPT_LEN").as<int64_t>(), 256);
}

INSTANTIATE_TEST_SUITE_P(Devices, Qwen3ASRDecoderFormatTest, ::testing::Values("CPU", "NPU"));

}  // namespace
