// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "audio_utils.hpp"

namespace {

using ov::genai::audio_utils::MelSpectrogramExtractor;

constexpr size_t kWhisperBins = 80;
constexpr size_t kQwen3OmniBins = 128;
constexpr size_t kSamplingRate = 16000;
constexpr size_t kNFft = 400;
constexpr size_t kHopLength = 160;
constexpr size_t kWhisperMinLength = kSamplingRate * 30;  // 480000

std::vector<float> make_sine_wave(size_t num_samples, float freq_hz, float sample_rate) {
    std::vector<float> out(num_samples);
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    for (size_t i = 0; i < num_samples; ++i) {
        const float t = static_cast<float>(i) / sample_rate;
        const float envelope = 1.0f - static_cast<float>(i) / static_cast<float>(num_samples);
        out[i] = envelope * std::sin(two_pi * freq_hz * t);
    }
    return out;
}

// Frame count formula. Matches MelSpectrogramExtractor semantics:
//   padded_size = max(raw_size, min_length) + 2 * (n_fft / 2)
//   n_frames    = (padded_size - n_fft) / hop_length
constexpr size_t expected_n_frames(size_t raw_size, size_t min_length, size_t n_fft, size_t hop) {
    const size_t effective = raw_size > min_length ? raw_size : min_length;
    const size_t padded = effective + 2 * (n_fft / 2);
    return (padded - n_fft) / hop;
}

}  // namespace

TEST(MelSpectrogramExtractor, ConstructsWithWhisperDefaults) {
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    EXPECT_EQ(ex.num_mel_bins(), kWhisperBins);
    EXPECT_EQ(ex.sampling_rate(), kSamplingRate);
    EXPECT_EQ(ex.n_fft(), kNFft);
    EXPECT_EQ(ex.hop_length(), kHopLength);
}

TEST(MelSpectrogramExtractor, ConstructsWithQwen3OmniDefaults) {
    MelSpectrogramExtractor ex(kQwen3OmniBins, kSamplingRate, kNFft, kHopLength);
    EXPECT_EQ(ex.num_mel_bins(), kQwen3OmniBins);
    EXPECT_EQ(ex.sampling_rate(), kSamplingRate);
    EXPECT_EQ(ex.n_fft(), kNFft);
    EXPECT_EQ(ex.hop_length(), kHopLength);
}

TEST(MelSpectrogramExtractor, ExtractProducesExpectedShape_ShortAudio) {
    // 1s of 440 Hz sine: 16000 samples.
    const size_t raw_size = 16000;
    const auto raw = make_sine_wave(raw_size, 440.0f, 16000.0f);
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n_frames = 0;
    const auto out = ex.extract(raw, n_frames);
    EXPECT_EQ(n_frames, expected_n_frames(raw_size, 0, kNFft, kHopLength));
    EXPECT_EQ(out.size(), kWhisperBins * n_frames);
}

TEST(MelSpectrogramExtractor, ExtractAssertsOnEmptyInput) {
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n_frames = 0;
    EXPECT_ANY_THROW({ (void)ex.extract({}, n_frames); });
}

TEST(MelSpectrogramExtractor, ExtractWithMinLengthAssertsOnEmptyInput) {
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n_frames = 0, n_active = 0;
    EXPECT_ANY_THROW({ (void)ex.extract({}, n_frames, kWhisperMinLength, &n_active); });
}

TEST(MelSpectrogramExtractor, ExtractAssertsOnAudioShorterThanReflectPad) {
    // n_fft=400 -> reflect_pad_size = 200. Audio of 100 samples is too short
    // for reflect padding; extract() (min_length=0) must assert to avoid UB.
    const auto raw = make_sine_wave(100, 440.0f, 16000.0f);
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n_frames = 0;
    EXPECT_ANY_THROW({ (void)ex.extract(raw, n_frames); });
}

TEST(MelSpectrogramExtractor, ExtractAppliesMinLengthForShortAudio) {
    // 0.5s of audio; min_length forces up to 30s worth of frames.
    const size_t raw_size = 8000;
    const auto raw = make_sine_wave(raw_size, 440.0f, 16000.0f);
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n_frames = 0, n_active = 0;
    const auto out = ex.extract(raw, n_frames, kWhisperMinLength, &n_active);
    EXPECT_EQ(n_frames, expected_n_frames(raw_size, kWhisperMinLength, kNFft, kHopLength));
    EXPECT_EQ(n_active, raw_size / kHopLength);  // 50
    EXPECT_EQ(out.size(), kWhisperBins * n_frames);
}

TEST(MelSpectrogramExtractor, ExtractWithAudioExactlyMinLength) {
    // Boundary: raw_size == min_length exactly. Neither zero-pads nor truncates.
    const auto raw = make_sine_wave(kWhisperMinLength, 440.0f, 16000.0f);
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n_frames = 0, n_active = 0;
    const auto out = ex.extract(raw, n_frames, kWhisperMinLength, &n_active);
    EXPECT_EQ(n_frames, expected_n_frames(kWhisperMinLength, kWhisperMinLength, kNFft, kHopLength));
    EXPECT_EQ(n_active, kWhisperMinLength / kHopLength);
    EXPECT_EQ(n_frames, n_active);  // aligned lengths: should match exactly
    EXPECT_EQ(out.size(), kWhisperBins * n_frames);
}

TEST(MelSpectrogramExtractor, ExtractWithLongAudio) {
    // 40s of audio > min_length=30s. min_length is ignored.
    const size_t raw_size = 640000;
    const auto raw = make_sine_wave(raw_size, 440.0f, 16000.0f);
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n_frames = 0, n_active = 0;
    const auto out = ex.extract(raw, n_frames, kWhisperMinLength, &n_active);
    EXPECT_EQ(n_frames, expected_n_frames(raw_size, kWhisperMinLength, kNFft, kHopLength));
    EXPECT_EQ(n_active, raw_size / kHopLength);  // 4000
    EXPECT_EQ(out.size(), kWhisperBins * n_frames);
}

TEST(MelSpectrogramExtractor, ExtractDefaultArgsMatchExplicitZeroMinLength) {
    // extract(raw, n_frames) must be identical to
    // extract(raw, n_frames, /*min_length=*/0, /*n_active_frames=*/&active).
    // Guards the default-argument contract.
    const auto raw = make_sine_wave(32000, 220.0f, 16000.0f);
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n_frames_a = 0;
    const auto out_a = ex.extract(raw, n_frames_a);
    size_t n_frames_b = 0, n_active_b = 0;
    const auto out_b = ex.extract(raw, n_frames_b, 0, &n_active_b);
    EXPECT_EQ(n_frames_a, n_frames_b);
    ASSERT_EQ(out_a.size(), out_b.size());
    for (size_t i = 0; i < out_a.size(); ++i) {
        EXPECT_FLOAT_EQ(out_a[i], out_b[i]) << "mismatch at index " << i;
    }
}

TEST(MelSpectrogramExtractor, ExtractIsDeterministic) {
    // Internal parallelism partitions frames across threads, but each frame's
    // reduction order is thread-local and deterministic, so repeated calls
    // must produce bit-identical output.
    const auto raw = make_sine_wave(16000, 440.0f, 16000.0f);
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n1 = 0, n2 = 0;
    const auto out1 = ex.extract(raw, n1);
    const auto out2 = ex.extract(raw, n2);
    EXPECT_EQ(n1, n2);
    ASSERT_EQ(out1.size(), out2.size());
    for (size_t i = 0; i < out1.size(); ++i) {
        EXPECT_FLOAT_EQ(out1[i], out2[i]);
    }
}

TEST(MelSpectrogramExtractor, ExtractNormalizationRange) {
    // Whisper normalization: clamp values below (max - 8), then apply (x + 4) / 4.
    // After normalization, all values must lie in [(max - 4) / 4, (max + 4) / 4]
    // — a tight algebraic bound, not a sanity range.
    const auto raw = make_sine_wave(16000, 440.0f, 16000.0f);
    MelSpectrogramExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    size_t n = 0, a = 0;
    const auto out = ex.extract(raw, n, kWhisperMinLength, &a);
    ASSERT_FALSE(out.empty());

    const float out_max = *std::max_element(out.begin(), out.end());
    const float out_min = *std::min_element(out.begin(), out.end());
    // After the (x + 4) / 4 step, the max of the output equals (orig_max + 4) / 4,
    // and the min equals (orig_max - 4) / 4. So out_max - out_min must equal 2.0
    // exactly (for unclamped distributions) or be strictly less than 2.0 if the
    // actual range was narrower than 8 before clamping.
    EXPECT_LE(out_max - out_min, 2.0f + 1e-5f);
    // orig_max = 4 * out_max - 4 (inverse of (x + 4) / 4).
    const float orig_max = 4.0f * out_max - 4.0f;
    const float expected_min = (orig_max - 4.0f) / 4.0f;
    const float expected_max = (orig_max + 4.0f) / 4.0f;
    EXPECT_GE(out_min, expected_min - 1e-5f);
    EXPECT_LE(out_max, expected_max + 1e-5f);
}

TEST(MelSpectrogramExtractor, ShapeDependsOnNumMelBins) {
    const auto raw = make_sine_wave(16000, 440.0f, 16000.0f);
    size_t n_whisper = 0;
    size_t n_qwen = 0;
    MelSpectrogramExtractor wh(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    MelSpectrogramExtractor qw(kQwen3OmniBins, kSamplingRate, kNFft, kHopLength);
    const auto out_wh = wh.extract(raw, n_whisper);
    const auto out_qw = qw.extract(raw, n_qwen);
    EXPECT_EQ(n_whisper, n_qwen);  // n_frames depends only on n_fft/hop, not bins
    EXPECT_EQ(out_wh.size(), kWhisperBins * n_whisper);
    EXPECT_EQ(out_qw.size(), kQwen3OmniBins * n_qwen);
}
