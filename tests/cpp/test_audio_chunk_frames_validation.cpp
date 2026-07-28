// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Regression tests for audio_chunk_frames bounds validation in
// validate_omni_talker_speech_config(). The criterion requires:
//   - audio_chunk_frames >= 1 when return_audio is true
//   - audio_chunk_frames <= 4096 when return_audio is true

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
