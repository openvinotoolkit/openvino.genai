// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "utils.hpp"
#include "visual_language/lfm3_omni/speech_core.hpp"

using namespace ov::genai;

namespace {

// Prompt plus appended marker; max_new_tokens is set, so this length never caps the slots.
constexpr size_t PROMPT_LENGTH = 4;
constexpr float LOG_PROB = -0.5f;

class LFM3SpeechStateSlots : public ::testing::Test {
protected:
    LFM3SpeechStateSlots() {
        m_model_config.audio_out_token_id = 10;
        m_model_config.audio_start_token_id = 11;
        m_model_config.mixed_start_token_id = 12;
        m_model_config.text_end_token_id = 13;
    }

    LFM3SpeechRequest make_request(SpeechOutputMode mode, size_t max_new_tokens) const {
        OmniTalkerSpeechConfig speech_config;
        speech_config.output_mode = mode;
        speech_config.max_new_tokens = 8;
        auto text_config = utils::get_greedy_config();
        text_config.max_new_tokens = max_new_tokens;
        text_config.eos_token_id = 2;
        text_config.stop_token_ids = {2};
        return LFM3SpeechRequest(speech_config, text_config);
    }

    LFM3SpeechModelConfig m_model_config;
    // The state keeps references to the config and the streamer, so both outlive it.
    const OmniSpeechStreamerVariant m_streamer = std::monostate{};
};

TEST_F(LFM3SpeechStateSlots, counts_one_slot_per_token) {
    const auto request = make_request(SpeechOutputMode::INTERLEAVED, 5);
    LFM3SpeechState state(m_model_config, request, m_streamer);

    // The appended marker goes into the context but is not a generated slot.
    EXPECT_EQ(state.take_prompt_marker(1), m_model_config.mixed_start_token_id);
    EXPECT_EQ(state.generated_slots(), 0);

    state.start(PROMPT_LENGTH);
    state.accept_text(100, LOG_PROB, StreamingStatus::RUNNING);
    state.accept_text(100, LOG_PROB, StreamingStatus::RUNNING);

    EXPECT_FALSE(state.finished());
    EXPECT_EQ(state.generated_slots(), 2);
}

TEST_F(LFM3SpeechStateSlots, stops_counting_at_the_finishing_slot) {
    const auto request = make_request(SpeechOutputMode::SPEECH_ONLY, 2);
    LFM3SpeechState state(m_model_config, request, m_streamer);

    EXPECT_EQ(state.take_prompt_marker(1), m_model_config.audio_start_token_id);
    state.start(PROMPT_LENGTH);
    state.accept_text(100, LOG_PROB, StreamingStatus::RUNNING);
    state.accept_text(101, LOG_PROB, StreamingStatus::RUNNING);

    ASSERT_TRUE(state.finished());
    EXPECT_EQ(state.generated_slots(), 2);

    // A slot fed after the finishing one must not move the count, or it drifts from the sequence length.
    state.accept_text(m_model_config.text_end_token_id, LOG_PROB, StreamingStatus::RUNNING);
    EXPECT_EQ(state.generated_slots(), 2);
}

}  // namespace
