// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/forced_aligner.hpp"

// Constructing from a directory without a valid forced-aligner export must fail rather than crash.
// temp_directory_path() exists but lacks config.json / model files, exercising the public ctor -> config validation.
TEST(ASRForcedAligner, IncompleteModelDirThrows) {
    EXPECT_ANY_THROW(ov::genai::ASRForcedAligner(std::filesystem::temp_directory_path(), "CPU"));
}

// Real-model standalone alignment smoke. Skipped unless a converted Qwen3-ForcedAligner directory is provided.
TEST(ASRForcedAligner, AlignSmoke) {
    const char* model_dir = std::getenv("QWEN3_FORCED_ALIGNER_MODEL");
    if (!model_dir || !std::filesystem::exists(model_dir)) {
        GTEST_SKIP() << "Set QWEN3_FORCED_ALIGNER_MODEL to a converted Qwen3-ForcedAligner directory.";
    }

    constexpr size_t sampling_rate = 16000;
    std::vector<float> audio(sampling_rate);
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] = 0.01f * std::sin(two_pi * 220.0f * static_cast<float>(i) / static_cast<float>(sampling_rate));
    }

    ov::genai::ASRForcedAligner aligner(model_dir, "CPU");
    const std::vector<ov::genai::ASRDecodedResultChunk> words =
        aligner.align(audio, "how are you doing today", "english");

    ASSERT_FALSE(words.empty());
    float previous_end = -1.0f;
    for (const ov::genai::ASRDecodedResultChunk& word : words) {
        EXPECT_TRUE(std::isfinite(word.start_ts) && std::isfinite(word.end_ts));
        EXPECT_LE(word.start_ts, word.end_ts);
        EXPECT_GE(word.start_ts, previous_end - 1e-3f);
        previous_end = word.end_ts;
    }
}
