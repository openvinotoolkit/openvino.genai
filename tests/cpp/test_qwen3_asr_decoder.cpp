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

INSTANTIATE_TEST_SUITE_P(Devices, Qwen3ASRDecoderFormatTest, ::testing::Values("CPU", "NPU"));

}  // namespace
