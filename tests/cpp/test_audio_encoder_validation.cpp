// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Regression tests for unbounded audio tensor / chunk_frames input
// validation in AudioEncoderQwen3Omni and validate_omni_talker_speech_config().
//
// These tests pre-validate inputs at the API boundary before allocation, so they do
// not need a fully-constructed pipeline. We test the boundary-validation helper
// directly. The helper is exposed for testing via the impl header; production code
// invokes it from AudioEncoderQwen3Omni::preprocess_audio().

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "openvino/genai/omni/talker_speech_config.hpp"
#include "omni/talker_speech_config_utils.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/qwen3_omni/audio_encoder.hpp"

namespace {

constexpr size_t kMaxAudioSamples = 480'000'000;  // ~500 min @ 16 kHz, matches impl

// Build a float32 audio tensor with the given shape filled with silence. Used to
// exercise the validate_audio_input() boundary check; we do not run the encoder
// (which requires a compiled model) — only the validation entry point.
ov::Tensor make_audio(std::initializer_list<size_t> shape) {
    return ov::Tensor(ov::element::f32, ov::Shape(shape));
}

}  // namespace

// ---- C6 audio tensor validation ----

TEST(AudioEncoderValidation, RejectsRank2Audio) {
    // 2-D tensor must be rejected before any allocation; impl currently passes
    // through silently and only fails deep inside WhisperFeatureExtractor.
    auto t = make_audio({2, 100});
    EXPECT_THROW({ ov::genai::AudioEncoderQwen3Omni::validate_audio_input(t); }, ov::Exception);
}

TEST(AudioEncoderValidation, RejectsEmptyAudio) {
    // Shape {0} is rank-1 but has 0 samples. Empty input must be rejected at the
    // boundary, not allowed to construct an empty std::vector and underflow later.
    auto t = make_audio({0});
    EXPECT_THROW({ ov::genai::AudioEncoderQwen3Omni::validate_audio_input(t); }, ov::Exception);
}

TEST(AudioEncoderValidation, RejectsHugeAudio) {
    // Use the shape-only overload so we never allocate kMaxAudioSamples + 1 floats:
    // the validator must reject based on the declared shape, before allocation.
    ov::Shape too_big{kMaxAudioSamples + 1};
    EXPECT_THROW({ ov::genai::AudioEncoderQwen3Omni::validate_audio_shape(too_big, ov::element::f32); },
                 ov::Exception);
}

TEST(AudioEncoderValidation, AcceptsValidShape) {
    // 1 second of audio at 16 kHz: 1-D, 16 000 samples, well within bounds.
    auto t = make_audio({16000});
    EXPECT_NO_THROW({ ov::genai::AudioEncoderQwen3Omni::validate_audio_input(t); });
}

TEST(AudioEncoderValidation, RejectsWrongElementType) {
    // PCM contract is f32; reject other types at the boundary so callers see a
    // clear error instead of a confusing "audio_data is null" deep inside.
    ov::Tensor t(ov::element::i16, {16000});
    EXPECT_THROW({ ov::genai::AudioEncoderQwen3Omni::validate_audio_input(t); }, ov::Exception);
}

// ---- audio_chunk_frames validation in validate_omni_talker_speech_config() ----

// Helper: the standalone OmniTalkerSpeechConfig has no GenerationConfig
// termination invariant to satisfy, so a default-constructed instance is valid
// out of the gate. All chunk-frame tests reuse this factory so the contract
// stays explicit.
ov::genai::OmniTalkerSpeechConfig make_speech_config() {
    return ov::genai::OmniTalkerSpeechConfig{};
}

TEST(OmniTalkerSpeechConfigAudioChunkFrames, RejectsZeroWhenReturnAudioTrue) {
    auto cfg = make_speech_config();
    cfg.return_audio = true;
    cfg.audio_chunk_frames = 0;
    EXPECT_THROW({ ov::genai::validate_omni_talker_speech_config(cfg); }, ov::Exception);
}

TEST(OmniTalkerSpeechConfigAudioChunkFrames, RejectsHugeChunkFramesWhenReturnAudioTrue) {
    auto cfg = make_speech_config();
    cfg.return_audio = true;
    cfg.audio_chunk_frames = std::numeric_limits<size_t>::max() / 2;
    EXPECT_THROW({ ov::genai::validate_omni_talker_speech_config(cfg); }, ov::Exception);
}

TEST(OmniTalkerSpeechConfigAudioChunkFrames, AcceptsValidChunkFrames) {
    auto cfg = make_speech_config();
    cfg.return_audio = true;
    cfg.audio_chunk_frames = 1;
    EXPECT_NO_THROW({ ov::genai::validate_omni_talker_speech_config(cfg); });
    cfg.audio_chunk_frames = 100;
    EXPECT_NO_THROW({ ov::genai::validate_omni_talker_speech_config(cfg); });
}

TEST(OmniTalkerSpeechConfigAudioChunkFrames, IgnoresChunkFramesWhenReturnAudioFalse) {
    // When return_audio is false, audio_chunk_frames is unused;
    // validate_omni_talker_speech_config() must not reject 0 / huge values
    // (they're irrelevant to text-only generation).
    auto cfg = make_speech_config();
    cfg.return_audio = false;
    cfg.audio_chunk_frames = 0;
    EXPECT_NO_THROW({ ov::genai::validate_omni_talker_speech_config(cfg); });
    cfg.audio_chunk_frames = std::numeric_limits<size_t>::max();
    EXPECT_NO_THROW({ ov::genai::validate_omni_talker_speech_config(cfg); });
}
