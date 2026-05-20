// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "whisper/feature_extractor.hpp"

namespace {

using ov::genai::WhisperFeatureExtractor;
using ov::genai::WhisperFeatures;

constexpr float kPi = M_PI;
constexpr size_t kWhisperBins = 80;
constexpr size_t kQwen3OmniBins = 128;
constexpr size_t kSamplingRate = 16000;
constexpr size_t kNFft = 400;
constexpr size_t kHopLength = 160;
constexpr size_t kWhisperNSamples = kSamplingRate * 30;  // 480000

std::vector<float> make_sine_wave(size_t num_samples, float freq_hz, float sample_rate) {
    std::vector<float> out(num_samples);
    const float two_pi = 2.0f * kPi;
    for (size_t i = 0; i < num_samples; ++i) {
        const float t = static_cast<float>(i) / sample_rate;
        const float envelope = 1.0f - static_cast<float>(i) / static_cast<float>(num_samples);
        out[i] = envelope * std::sin(two_pi * freq_hz * t);
    }
    return out;
}

// Frame count formula. Matches WhisperFeatureExtractor semantics:
//   padded_size = max(raw_size, n_samples) + 2 * (n_fft / 2)
//   n_frames    = (padded_size - n_fft) / hop
constexpr size_t expected_n_frames(size_t raw_size, size_t n_samples, size_t n_fft, size_t hop) {
    const size_t effective = raw_size > n_samples ? raw_size : n_samples;
    const size_t padded = effective + 2 * (n_fft / 2);
    return (padded - n_fft) / hop;
}

// Build a Whisper-style extractor that pre-pads to 30 s by overriding n_samples
// on top of the explicit-numeric ctor (which defaults to n_samples = 0).
WhisperFeatureExtractor make_whisper_extractor(size_t feature_size = kWhisperBins) {
    WhisperFeatureExtractor ex(feature_size, kSamplingRate, kNFft, kHopLength);
    ex.n_samples = kWhisperNSamples;
    return ex;
}

}  // namespace

TEST(WhisperFeatureExtractor, ConstructsWithWhisperDefaults) {
    WhisperFeatureExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    EXPECT_EQ(ex.feature_size, kWhisperBins);
    EXPECT_EQ(ex.sampling_rate, kSamplingRate);
    EXPECT_EQ(ex.n_fft, kNFft);
    EXPECT_EQ(ex.hop_length, kHopLength);
    // Explicit-numeric ctor disables pre-padding (matches the Qwen3-Omni configuration
    // where preprocessor_config.json declares WhisperFeatureExtractor without fixing
    // chunk_length).
    EXPECT_EQ(ex.n_samples, 0u);
}

TEST(WhisperFeatureExtractor, ConstructsWithQwen3OmniDefaults) {
    WhisperFeatureExtractor ex(kQwen3OmniBins, kSamplingRate, kNFft, kHopLength);
    EXPECT_EQ(ex.feature_size, kQwen3OmniBins);
    EXPECT_EQ(ex.sampling_rate, kSamplingRate);
    EXPECT_EQ(ex.n_fft, kNFft);
    EXPECT_EQ(ex.hop_length, kHopLength);
}

TEST(WhisperFeatureExtractor, ExtractProducesExpectedShape_ShortAudio) {
    // 1s of 440 Hz sine: 16000 samples. Qwen3-style (n_samples = 0): no pre-pad.
    const size_t raw_size = 16000;
    const auto raw = make_sine_wave(raw_size, 440.0f, 16000.0f);
    WhisperFeatureExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    const auto features = ex.extract(raw);
    EXPECT_EQ(features.feature_size, kWhisperBins);
    EXPECT_EQ(features.n_frames, expected_n_frames(raw_size, 0, kNFft, kHopLength));
    EXPECT_EQ(features.data.size(), kWhisperBins * features.n_frames);
}

TEST(WhisperFeatureExtractor, ExtractAssertsOnEmptyInput) {
    WhisperFeatureExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    EXPECT_ANY_THROW({ (void)ex.extract({}); });
}

TEST(WhisperFeatureExtractor, ExtractWithWhisperPrepadAssertsOnEmptyInput) {
    auto ex = make_whisper_extractor();
    EXPECT_ANY_THROW({ (void)ex.extract({}); });
}

TEST(WhisperFeatureExtractor, ExtractAssertsOnAudioShorterThanReflectPad) {
    // n_fft=400 -> reflect_pad_size = 200. Audio of 100 samples is too short
    // for reflect padding; extract() (n_samples=0) must assert to avoid UB.
    const auto raw = make_sine_wave(100, 440.0f, 16000.0f);
    WhisperFeatureExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    EXPECT_ANY_THROW({ (void)ex.extract(raw); });
}

TEST(WhisperFeatureExtractor, ExtractAppliesPrepadForShortAudio) {
    // 0.5s of audio; Whisper-style n_samples forces up to 30s worth of frames.
    const size_t raw_size = 8000;
    const auto raw = make_sine_wave(raw_size, 440.0f, 16000.0f);
    auto ex = make_whisper_extractor();
    const auto features = ex.extract(raw);
    EXPECT_EQ(features.n_frames, expected_n_frames(raw_size, kWhisperNSamples, kNFft, kHopLength));
    EXPECT_EQ(features.n_active_frames, raw_size / kHopLength);  // 50
    EXPECT_EQ(features.data.size(), kWhisperBins * features.n_frames);
}

TEST(WhisperFeatureExtractor, ExtractWithAudioExactlyNSamples) {
    // Boundary: raw_size == n_samples exactly. Neither zero-pads nor truncates.
    auto ex = make_whisper_extractor();
    const auto raw = make_sine_wave(kWhisperNSamples, 440.0f, 16000.0f);
    const auto features = ex.extract(raw);
    EXPECT_EQ(features.n_frames, expected_n_frames(kWhisperNSamples, kWhisperNSamples, kNFft, kHopLength));
    EXPECT_EQ(features.n_active_frames, kWhisperNSamples / kHopLength);
    EXPECT_EQ(features.n_frames, features.n_active_frames);  // aligned lengths: should match exactly
    EXPECT_EQ(features.data.size(), kWhisperBins * features.n_frames);
}

TEST(WhisperFeatureExtractor, ExtractWithLongAudio) {
    // 40s of audio > n_samples=30s. n_samples is ignored by max() in pad_with_reflect.
    const size_t raw_size = 640000;
    const auto raw = make_sine_wave(raw_size, 440.0f, 16000.0f);
    auto ex = make_whisper_extractor();
    const auto features = ex.extract(raw);
    EXPECT_EQ(features.n_frames, expected_n_frames(raw_size, kWhisperNSamples, kNFft, kHopLength));
    EXPECT_EQ(features.n_active_frames, raw_size / kHopLength);  // 4000
    EXPECT_EQ(features.data.size(), kWhisperBins * features.n_frames);
}

TEST(WhisperFeatureExtractor, ExtractMatchesAcrossExtractorsWithSameConfig) {
    // Two extractors with identical config must produce identical output for the
    // same input — guards constructor determinism (sin/cos/mel tables are rebuilt
    // from scratch and must agree bit-for-bit).
    const auto raw = make_sine_wave(32000, 220.0f, 16000.0f);
    WhisperFeatureExtractor ex_a(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    WhisperFeatureExtractor ex_b(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    const auto fa = ex_a.extract(raw);
    const auto fb = ex_b.extract(raw);
    EXPECT_EQ(fa.n_frames, fb.n_frames);
    ASSERT_EQ(fa.data.size(), fb.data.size());
    for (size_t i = 0; i < fa.data.size(); ++i) {
        EXPECT_FLOAT_EQ(fa.data[i], fb.data[i]) << "mismatch at index " << i;
    }
}

TEST(WhisperFeatureExtractor, ExtractIsDeterministic) {
    // Internal parallelism partitions frames across threads, but each frame's
    // reduction order is thread-local and deterministic, so repeated calls
    // must produce bit-identical output.
    const auto raw = make_sine_wave(16000, 440.0f, 16000.0f);
    WhisperFeatureExtractor ex(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    const auto f1 = ex.extract(raw);
    const auto f2 = ex.extract(raw);
    EXPECT_EQ(f1.n_frames, f2.n_frames);
    ASSERT_EQ(f1.data.size(), f2.data.size());
    for (size_t i = 0; i < f1.data.size(); ++i) {
        EXPECT_FLOAT_EQ(f1.data[i], f2.data[i]);
    }
}

TEST(WhisperFeatureExtractor, ExtractNormalizationRange) {
    // Whisper normalization: clamp values below (max - 8), then apply (x + 4) / 4.
    // After normalization, all values must lie in [(max - 4) / 4, (max + 4) / 4]
    // — a tight algebraic bound, not a sanity range.
    const auto raw = make_sine_wave(16000, 440.0f, 16000.0f);
    auto ex = make_whisper_extractor();
    const auto features = ex.extract(raw);
    ASSERT_FALSE(features.data.empty());

    const float out_max = *std::max_element(features.data.begin(), features.data.end());
    const float out_min = *std::min_element(features.data.begin(), features.data.end());
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

TEST(WhisperFeatureExtractor, ShapeDependsOnFeatureSize) {
    const auto raw = make_sine_wave(16000, 440.0f, 16000.0f);
    WhisperFeatureExtractor wh(kWhisperBins, kSamplingRate, kNFft, kHopLength);
    WhisperFeatureExtractor qw(kQwen3OmniBins, kSamplingRate, kNFft, kHopLength);
    const auto fw = wh.extract(raw);
    const auto fq = qw.extract(raw);
    EXPECT_EQ(fw.n_frames, fq.n_frames);  // n_frames depends only on n_fft/hop, not bins
    EXPECT_EQ(fw.data.size(), kWhisperBins * fw.n_frames);
    EXPECT_EQ(fq.data.size(), kQwen3OmniBins * fq.n_frames);
}
