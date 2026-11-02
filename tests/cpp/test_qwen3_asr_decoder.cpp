// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>

#include "automatic_speech_recognition/models/qwen3-asr/encoder_decoder/decoder.hpp"
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
            ov::genai::Qwen3ASRDecoder decoder(m_model_dir, GetParam(), {}, 128);
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
    EXPECT_NO_THROW(ov::genai::Qwen3ASRDecoder(m_model_dir, "CPU", {}, 128));
}

TEST_F(Qwen3ASRDecoderFormatTest, SharedFunASRConstructorCompilesOnCPU) {
    save_decoder(true, true);
    EXPECT_NO_THROW(ov::genai::Qwen3ASRDecoder(m_model_dir, "CPU", {}));
}

INSTANTIATE_TEST_SUITE_P(Devices, Qwen3ASRDecoderFormatTest, ::testing::Values("CPU", "NPU"));

}  // namespace
