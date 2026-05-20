// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Regression tests for unbounded audio tensor / chunk_frames input
// validation in AudioEncoderQwen3Omni and GenerationConfig::validate().
//
// These tests pre-validate inputs at the API boundary before allocation, so they do
// not need a fully-constructed pipeline. We test the boundary-validation helper
// directly. The helper is exposed for testing via the impl header; production code
// invokes it from AudioEncoderQwen3Omni::preprocess_audio().

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/qwen3_omni/audio_encoder.hpp"

namespace {

constexpr size_t kSamplingRate = 16000;
constexpr size_t kMaxAudioSamples = 480'000'000;  // 30 min @ 16 kHz, matches impl

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
    // We don't actually allocate kMaxAudioSamples + 1 floats (would OOM). The
    // validator must reject based on declared shape, before allocation.
    // Build a small placeholder tensor and forge a shape via reshape() -- but
    // ov::Tensor doesn't allow lying about shape, so we build a dummy with the
    // exact threshold + 1 size. Use a small element_type to limit allocation cost.
    // Workaround: skip allocation; construct a shape directly and call the static
    // helper that takes a Shape, not a Tensor. Since the helper takes a Tensor, we
    // build the smallest tensor whose nominal size exceeds the cap. We cannot —
    // ov::Tensor allocates eagerly. Instead, the boundary helper must also expose
    // a shape-only overload for testing oversize cases.
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

// ---- C6 audio_chunk_frames validation in GenerationConfig::validate() ----

// Helper: a minimally-valid GenerationConfig that satisfies validate()'s
// termination-invariant check (one of eos_token_id, stop_token_ids, stop_strings,
// max_new_tokens, max_length must be set). All audio_chunk_frames tests build on
// this so we only exercise the audio-specific assertion, not the unrelated
// termination check.
ov::genai::GenerationConfig make_terminable_config() {
    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = 32;
    return cfg;
}

TEST(GenerationConfigAudioChunkFrames, RejectsZeroWhenReturnAudioTrue) {
    // The >=1 check used to happen at speech-generation time; we want it at
    // config validation time so users get the error at config construction.
    auto cfg = make_terminable_config();
    cfg.return_audio = true;
    cfg.audio_chunk_frames = 0;
    EXPECT_THROW({ cfg.validate(); }, ov::Exception);
}

TEST(GenerationConfigAudioChunkFrames, RejectsHugeChunkFramesWhenReturnAudioTrue) {
    // SIZE_MAX / 2 used to pass the >=1 check and later underflow the
    // streaming-chunk subtraction → never streams → OOM. Reject up front.
    auto cfg = make_terminable_config();
    cfg.return_audio = true;
    cfg.audio_chunk_frames = std::numeric_limits<size_t>::max() / 2;
    EXPECT_THROW({ cfg.validate(); }, ov::Exception);
}

TEST(GenerationConfigAudioChunkFrames, AcceptsValidChunkFrames) {
    auto cfg = make_terminable_config();
    cfg.return_audio = true;
    cfg.audio_chunk_frames = 1;
    EXPECT_NO_THROW({ cfg.validate(); });
    cfg.audio_chunk_frames = 100;
    EXPECT_NO_THROW({ cfg.validate(); });
}

TEST(GenerationConfigAudioChunkFrames, IgnoresChunkFramesWhenReturnAudioFalse) {
    // When return_audio is false, audio_chunk_frames is unused; validate() must
    // not reject 0 / huge values (they're irrelevant to text-only generation).
    auto cfg = make_terminable_config();
    cfg.return_audio = false;
    cfg.audio_chunk_frames = 0;
    EXPECT_NO_THROW({ cfg.validate(); });
    cfg.audio_chunk_frames = std::numeric_limits<size_t>::max();
    EXPECT_NO_THROW({ cfg.validate(); });
}
