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

#include <cstdint>
#include <limits>
#include <numeric>
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

// ---- CVS-193623: mel chunking must not duplicate audio ----

namespace {

using Encoder = ov::genai::AudioEncoderQwen3Omni;

// Every shipped Qwen3-Omni config uses n_window = 50, so the chunk width is 100 frames.
constexpr size_t kNWindow = 50;
constexpr size_t kFramesPerSecond = 100;  // 16 kHz / hop 160

// Python floors on integer division, C++ truncates. The formula below feeds
// negative numerators, so the difference changes the result.
int64_t floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

// Port of _get_feat_extract_output_lengths() from
// transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py, so the test
// checks against upstream instead of restating our own code. The 100 and 13 are
// upstream literals and assume n_window == 50.
int64_t upstream_output_length(int64_t n_frames) {
    const int64_t leave = n_frames % 100;
    const int64_t feat = floor_div(leave - 1, 2) + 1;
    return floor_div(floor_div(feat - 1, 2) + 1 - 1, 2) + 1 + floor_div(n_frames, 100) * 13;
}

size_t total_audio_tokens(size_t n_frames) {
    size_t total = 0;
    for (size_t len : Encoder::plan_chunk_frame_lens(n_frames, kNWindow)) {
        total += Encoder::get_feat_extract_output_length(len);
    }
    return total;
}

}  // namespace

TEST(AudioEncoderChunking, ChunksAreDisjointAndCoverEveryFrameExactlyOnce) {
    // More than n_frames means duplicated audio, less means dropped audio.
    for (size_t n_frames : {1u, 50u, 99u, 100u, 101u, 200u, 1442u, 4000u}) {
        const auto lens = Encoder::plan_chunk_frame_lens(n_frames, kNWindow);
        const size_t covered = std::accumulate(lens.begin(), lens.end(), size_t{0});
        EXPECT_EQ(covered, n_frames) << "n_frames=" << n_frames;

        // Only the tail may be short; every other chunk is a full n_window * 2.
        for (size_t i = 0; i + 1 < lens.size(); i++) {
            EXPECT_EQ(lens[i], kNWindow * 2) << "n_frames=" << n_frames << " chunk=" << i;
        }
        ASSERT_FALSE(lens.empty());
        EXPECT_GT(lens.back(), 0u);
        EXPECT_LE(lens.back(), kNWindow * 2);
    }
}

TEST(AudioEncoderChunking, TokenCountMatchesTransformers) {
    for (size_t n_frames = 1; n_frames <= 4000; n_frames++) {
        EXPECT_EQ(static_cast<int64_t>(total_audio_tokens(n_frames)), upstream_output_length(n_frames))
            << "n_frames=" << n_frames;
    }
}

TEST(AudioEncoderChunking, TokenCountForKnownDurations) {
    // HF processor values. A drift here means the model gets a different amount
    // of audio than it saw in training.
    EXPECT_EQ(total_audio_tokens(kFramesPerSecond / 2), 7u);   // 0.5 s
    EXPECT_EQ(total_audio_tokens(kFramesPerSecond), 13u);      // 1 s
    EXPECT_EQ(total_audio_tokens(kFramesPerSecond * 2), 26u);  // 2 s
    EXPECT_EQ(total_audio_tokens(kFramesPerSecond * 4), 52u);  // 4 s

    // The CVS-193623 utterance (LibriSpeech test-clean 3575-170457-0015, 14.425 s).
    // Overlapping windows gave 337 tokens at n_window_infer=200 and 262 at 800.
    EXPECT_EQ(total_audio_tokens(1442), 188u);
}

TEST(AudioEncoderChunking, RejectsDegenerateInput) {
    EXPECT_THROW({ Encoder::plan_chunk_frame_lens(0, kNWindow); }, ov::Exception);
    EXPECT_THROW({ Encoder::plan_chunk_frame_lens(100, 0); }, ov::Exception);
}
