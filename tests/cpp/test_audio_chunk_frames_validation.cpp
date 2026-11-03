// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Regression tests for audio_chunk_frames bounds validation in
// validate_omni_talker_speech_config(). The criterion requires:
//   - audio_chunk_frames >= 1 when return_audio is true
//   - audio_chunk_frames <= 4096 when return_audio is true
//   - audio_chunk_left_context_frames <= 4096 when return_audio is true

#include <gtest/gtest.h>

#include "openvino/genai/omni/talker_speech_config.hpp"
#include "omni/talker_speech_config_utils.hpp"

namespace {

ov::genai::OmniTalkerSpeechConfig make_config_with_audio(std::size_t chunk_frames) {
    ov::genai::OmniTalkerSpeechConfig cfg;
    cfg.return_audio = true;
    cfg.audio_chunk_frames = chunk_frames;
    return cfg;
}

}  // namespace

TEST(AudioChunkFramesValidation, RejectsZeroChunkFramesWhenReturnAudioTrue) {
    auto cfg = make_config_with_audio(0);
    EXPECT_THROW(ov::genai::validate_omni_talker_speech_config(cfg), ov::Exception);
}

TEST(AudioChunkFramesValidation, RejectsExcessiveChunkFramesWhenReturnAudioTrue) {
    auto cfg = make_config_with_audio(9999);
    EXPECT_THROW(ov::genai::validate_omni_talker_speech_config(cfg), ov::Exception);
}

TEST(AudioChunkFramesValidation, AcceptsLeftContextFramesUpToBound) {
    auto cfg = make_config_with_audio(4);
    for (const std::size_t left_context_frames : {std::size_t{0}, std::size_t{25}, std::size_t{4096}}) {
        cfg.audio_chunk_left_context_frames = left_context_frames;
        EXPECT_NO_THROW(ov::genai::validate_omni_talker_speech_config(cfg)) << left_context_frames;
    }
}

TEST(AudioChunkFramesValidation, RejectsExcessiveLeftContextFrames) {
    auto cfg = make_config_with_audio(4);
    cfg.audio_chunk_left_context_frames = 4097;
    EXPECT_THROW(ov::genai::validate_omni_talker_speech_config(cfg), ov::Exception);
}

TEST(AudioChunkFramesValidation, ReadsLeftContextFramesFromProperties) {
    ov::genai::OmniTalkerSpeechConfig cfg;
    EXPECT_TRUE(ov::genai::is_omni_talker_speech_config_key("audio_chunk_left_context_frames"));
    ov::genai::update_omni_talker_speech_config(cfg, {{"audio_chunk_left_context_frames", std::size_t{3}}});
    EXPECT_EQ(cfg.audio_chunk_left_context_frames, 3);
}
