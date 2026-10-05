// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "visual_language/qwen3_omni/classes.hpp"

using ov::genai::EncodedAudio;
using ov::genai::qwen3_omni::expand_audio_tags;
using ov::genai::qwen3_omni::merge_audio_embeddings;

namespace {

constexpr int64_t AUDIO_ID = 7;
constexpr int64_t TEXT_ID = 1;
constexpr size_t HIDDEN = 2;
constexpr float TEXT_VALUE = -1.0f;

std::string tag(size_t pads) {
    std::string result{ov::genai::qwen3_omni::AUDIO_START_TAG};
    for (size_t i = 0; i < pads; i++) {
        result += ov::genai::qwen3_omni::AUDIO_PAD_TAG;
    }
    return result += ov::genai::qwen3_omni::AUDIO_END_TAG;
}

// Row t of audio `label` holds label * 100 + t, so every merged row names its source.
EncodedAudio make_audio(size_t label, size_t tokens) {
    if (tokens == 0) {
        return {ov::Tensor(), 0};
    }
    ov::Tensor features{ov::element::f32, {tokens, HIDDEN}};
    auto* data = features.data<float>();
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < HIDDEN; h++) {
            data[t * HIDDEN + h] = static_cast<float>(label * 100 + t);
        }
    }
    return {features, tokens};
}

ov::Tensor make_embeds(size_t seq_len) {
    ov::Tensor embeds{ov::element::f32, {1, seq_len, HIDDEN}};
    std::fill_n(embeds.data<float>(), embeds.get_size(), TEXT_VALUE);
    return embeds;
}

// First column of each row; all columns of a row hold the same value.
std::vector<float> rows(const ov::Tensor& embeds) {
    const auto seq_len = embeds.get_shape()[1];
    const auto* data = embeds.data<const float>();
    std::vector<float> result;
    for (size_t i = 0; i < seq_len; i++) {
        result.push_back(data[i * HIDDEN]);
    }
    return result;
}

}  // namespace

// ------------------------------------------------------------------------- expansion

TEST(Qwen3OmniAudioExpansion, EachTagGetsItsOwnPadCount) {
    std::string prompt = "A " + tag(1) + " B " + tag(1);
    expand_audio_tags(prompt, {make_audio(0, 2), make_audio(1, 3)}, {0, 1}, 0);
    EXPECT_EQ(prompt, "A " + tag(2) + " B " + tag(3));
}

// A one-pad expansion is byte-identical to the tag. Searching from zero would find slot 0 again
// and give it the second audio's pads.
TEST(Qwen3OmniAudioExpansion, OnePadAudioDoesNotSwallowNextTag) {
    std::string prompt = "A " + tag(1) + " B " + tag(1);
    expand_audio_tags(prompt, {make_audio(0, 1), make_audio(1, 3)}, {0, 1}, 0);
    EXPECT_EQ(prompt, "A " + tag(1) + " B " + tag(3));
}

TEST(Qwen3OmniAudioExpansion, ReorderedSequenceSizesBySequenceNotPosition) {
    std::string prompt = "A " + tag(1) + " B " + tag(1);
    expand_audio_tags(prompt, {make_audio(0, 2), make_audio(1, 3)}, {1, 0}, 0);
    EXPECT_EQ(prompt, "A " + tag(3) + " B " + tag(2));
}

TEST(Qwen3OmniAudioExpansion, DuplicateIndexExpandsBoth) {
    std::string prompt = tag(1) + " and " + tag(1);
    expand_audio_tags(prompt, {make_audio(0, 2)}, {0, 0}, 0);
    EXPECT_EQ(prompt, tag(2) + " and " + tag(2));
}

TEST(Qwen3OmniAudioExpansion, BaseIdMapsAbsoluteIndexToThisTurn) {
    std::string prompt = "x " + tag(1);
    expand_audio_tags(prompt, {make_audio(0, 4)}, {3}, 3);
    EXPECT_EQ(prompt, "x " + tag(4));
}

TEST(Qwen3OmniAudioExpansion, ZeroTokenAudioKeepsOnlyStartAndEnd) {
    std::string prompt = tag(1) + "x";
    expand_audio_tags(prompt, {make_audio(0, 0)}, {0}, 0);
    EXPECT_EQ(prompt, tag(0) + "x");
}

TEST(Qwen3OmniAudioExpansion, IndexOutOfRangeIsRejected) {
    std::string prompt = tag(1);
    EXPECT_THROW(expand_audio_tags(prompt, {make_audio(0, 2)}, {1}, 0), ov::Exception);
    EXPECT_THROW(expand_audio_tags(prompt, {make_audio(0, 2)}, {0}, 1), ov::Exception) << "index below base";
}

TEST(Qwen3OmniAudioExpansion, FewerTagsThanSequenceIsRejected) {
    std::string prompt = tag(1);
    EXPECT_THROW(expand_audio_tags(prompt, {make_audio(0, 2)}, {0, 0}, 0), ov::Exception);
}

// ----------------------------------------------------------------------------- merge

TEST(Qwen3OmniAudioMerge, EachRunTakesItsOwnAudio) {
    const std::vector<int64_t> ids{TEXT_ID, AUDIO_ID, AUDIO_ID, TEXT_ID, AUDIO_ID, AUDIO_ID, AUDIO_ID};
    auto embeds = make_embeds(ids.size());
    merge_audio_embeddings(embeds, ids, {make_audio(0, 2), make_audio(1, 3)}, {0, 1}, AUDIO_ID);
    EXPECT_EQ(rows(embeds), (std::vector<float>{TEXT_VALUE, 0, 1, TEXT_VALUE, 100, 101, 102}));
}

// Binding in document order would copy audio 0 (2 rows) into the 3-token run and fail.
TEST(Qwen3OmniAudioMerge, ReorderedSequenceBindsRunToIndex) {
    const std::vector<int64_t> ids{TEXT_ID, AUDIO_ID, AUDIO_ID, AUDIO_ID, TEXT_ID, AUDIO_ID, AUDIO_ID};
    auto embeds = make_embeds(ids.size());
    merge_audio_embeddings(embeds, ids, {make_audio(0, 2), make_audio(1, 3)}, {1, 0}, AUDIO_ID);
    EXPECT_EQ(rows(embeds), (std::vector<float>{TEXT_VALUE, 100, 101, 102, TEXT_VALUE, 0, 1}));
}

TEST(Qwen3OmniAudioMerge, OnePadRunsStaySeparate) {
    const std::vector<int64_t> ids{AUDIO_ID, TEXT_ID, AUDIO_ID};
    auto embeds = make_embeds(ids.size());
    merge_audio_embeddings(embeds, ids, {make_audio(0, 1), make_audio(1, 1)}, {0, 1}, AUDIO_ID);
    EXPECT_EQ(rows(embeds), (std::vector<float>{0, TEXT_VALUE, 100}));
}

TEST(Qwen3OmniAudioMerge, DuplicateIndexFillsBothRuns) {
    const std::vector<int64_t> ids{AUDIO_ID, AUDIO_ID, TEXT_ID, AUDIO_ID, AUDIO_ID};
    auto embeds = make_embeds(ids.size());
    merge_audio_embeddings(embeds, ids, {make_audio(0, 2)}, {0, 0}, AUDIO_ID);
    EXPECT_EQ(rows(embeds), (std::vector<float>{0, 1, TEXT_VALUE, 0, 1}));
}

// A zero-token audio has no run. It must be skipped, or run k stops matching audios_sequence[k].
TEST(Qwen3OmniAudioMerge, ZeroTokenAudioIsSkippedAnywhere) {
    const std::vector<int64_t> ids{TEXT_ID, AUDIO_ID, AUDIO_ID};
    for (const std::vector<size_t>& sequence : {std::vector<size_t>{0, 1}, std::vector<size_t>{1, 0}}) {
        auto embeds = make_embeds(ids.size());
        merge_audio_embeddings(embeds, ids, {make_audio(0, 0), make_audio(1, 2)}, sequence, AUDIO_ID);
        EXPECT_EQ(rows(embeds), (std::vector<float>{TEXT_VALUE, 100, 101}));
    }
}

TEST(Qwen3OmniAudioMerge, EmptySequenceLeavesEmbedsAlone) {
    const std::vector<int64_t> ids{TEXT_ID, TEXT_ID};
    auto embeds = make_embeds(ids.size());
    merge_audio_embeddings(embeds, ids, {}, {}, -1);
    EXPECT_EQ(rows(embeds), (std::vector<float>{TEXT_VALUE, TEXT_VALUE}));
}

TEST(Qwen3OmniAudioMerge, MissingAudioTokenIdIsRejected) {
    const std::vector<int64_t> ids{AUDIO_ID, AUDIO_ID};
    auto embeds = make_embeds(ids.size());
    EXPECT_THROW(merge_audio_embeddings(embeds, ids, {make_audio(0, 2)}, {0}, -1), ov::Exception);
}

TEST(Qwen3OmniAudioMerge, RunLengthMismatchIsRejected) {
    const std::vector<int64_t> ids{AUDIO_ID, AUDIO_ID, AUDIO_ID};
    auto embeds = make_embeds(ids.size());
    EXPECT_THROW(merge_audio_embeddings(embeds, ids, {make_audio(0, 2)}, {0}, AUDIO_ID), ov::Exception);
}

TEST(Qwen3OmniAudioMerge, MoreRunsThanAudiosIsRejected) {
    const std::vector<int64_t> ids{AUDIO_ID, TEXT_ID, AUDIO_ID};
    auto embeds = make_embeds(ids.size());
    EXPECT_THROW(merge_audio_embeddings(embeds, ids, {make_audio(0, 1)}, {0}, AUDIO_ID), ov::Exception);
}

TEST(Qwen3OmniAudioMerge, FewerRunsThanAudiosIsRejected) {
    const std::vector<int64_t> ids{AUDIO_ID, TEXT_ID};
    auto embeds = make_embeds(ids.size());
    EXPECT_THROW(merge_audio_embeddings(embeds, ids, {make_audio(0, 1), make_audio(1, 1)}, {0, 1}, AUDIO_ID),
                 ov::Exception);
}

TEST(Qwen3OmniAudioMerge, SequenceIndexOutOfRangeIsRejected) {
    const std::vector<int64_t> ids{AUDIO_ID};
    auto embeds = make_embeds(ids.size());
    EXPECT_THROW(merge_audio_embeddings(embeds, ids, {make_audio(0, 1)}, {1}, AUDIO_ID), ov::Exception);
}
