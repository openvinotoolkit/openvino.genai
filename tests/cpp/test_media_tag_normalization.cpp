// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vlm_utils.hpp"

using ov::genai::ModalityType;
using ov::genai::normalize_media_tags;
using ov::genai::vlm_utils::rebase_media_sequence;

namespace {

// Qwen3-Omni's audio marker. Single pad here; the per-audio expansion happens later.
const std::string AUDIO_TAG = "<|audio_start|><|audio_pad|><|audio_end|>";

std::pair<std::string, std::vector<size_t>> normalize_audio(const std::string& prompt,
                                                            size_t base_idx,
                                                            size_t n_audios) {
    return normalize_media_tags(prompt, AUDIO_TAG, AUDIO_TAG, base_idx, n_audios, ModalityType::AUDIO);
}

}  // namespace

// ---------------------------------------------------------------- no tags: prepend

TEST(MediaTagNormalization, NoTagPrependsInInputOrder) {
    const auto [prompt, sequence] = normalize_audio("Describe", 0, 2);
    EXPECT_EQ(prompt, AUDIO_TAG + AUDIO_TAG + "Describe");
    EXPECT_EQ(sequence, (std::vector<size_t>{0, 1}));
}

TEST(MediaTagNormalization, NoTagNoMediaLeavesPromptAlone) {
    const auto [prompt, sequence] = normalize_audio("Describe", 0, 0);
    EXPECT_EQ(prompt, "Describe");
    EXPECT_TRUE(sequence.empty());
}

// ------------------------------------------------------- universal tags keep position

TEST(MediaTagNormalization, UniversalTagAtFrontMatchesPrependDefault) {
    const auto tagged = normalize_audio("<ov_genai_audio_0>Describe", 0, 1);
    const auto defaulted = normalize_audio("Describe", 0, 1);
    EXPECT_EQ(tagged.first, defaulted.first) << "an explicit leading tag must equal the prepend default";
    EXPECT_EQ(tagged.second, defaulted.second);
}

TEST(MediaTagNormalization, UniversalTagAppended) {
    const auto [prompt, sequence] = normalize_audio("Describe this: <ov_genai_audio_0>", 0, 1);
    EXPECT_EQ(prompt, "Describe this: " + AUDIO_TAG);
    EXPECT_EQ(sequence, (std::vector<size_t>{0}));
}

// The case no modality covered before this branch: a tag with text on BOTH sides.
TEST(MediaTagNormalization, UniversalTagInterleaved) {
    const auto [prompt, sequence] = normalize_audio("text <ov_genai_audio_0> more text", 0, 1);
    EXPECT_EQ(prompt, "text " + AUDIO_TAG + " more text");
    EXPECT_EQ(sequence, (std::vector<size_t>{0}));
}

TEST(MediaTagNormalization, TwoUniversalTagsKeepTheirOwnPositions) {
    const auto [prompt, sequence] = normalize_audio("A <ov_genai_audio_0> B <ov_genai_audio_1> C", 0, 2);
    EXPECT_EQ(prompt, "A " + AUDIO_TAG + " B " + AUDIO_TAG + " C");
    EXPECT_EQ(sequence, (std::vector<size_t>{0, 1}));
}

// Reversed indices must produce a reversed *sequence* while the tag positions stay put — that is
// what makes per-item binding observable at all.
TEST(MediaTagNormalization, ReversedIndicesReverseTheSequenceNotThePositions) {
    const auto forward = normalize_audio("A <ov_genai_audio_0> B <ov_genai_audio_1>", 0, 2);
    const auto reversed = normalize_audio("A <ov_genai_audio_1> B <ov_genai_audio_0>", 0, 2);
    EXPECT_EQ(forward.first, reversed.first) << "same skeleton, so the rendered prompt is identical";
    EXPECT_EQ(forward.second, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(reversed.second, (std::vector<size_t>{1, 0}));
}

TEST(MediaTagNormalization, DuplicateIndexRendersTwice) {
    const auto [prompt, sequence] = normalize_audio("<ov_genai_audio_0> and <ov_genai_audio_0>", 0, 1);
    EXPECT_EQ(prompt, AUDIO_TAG + " and " + AUDIO_TAG);
    EXPECT_EQ(sequence, (std::vector<size_t>{0, 0}));
}

// ----------------------------------------------------------------- base index offset

TEST(MediaTagNormalization, BaseIndexOffsetsTheSequence) {
    const auto [prompt, sequence] = normalize_audio("<ov_genai_audio_3>x", 3, 1);
    EXPECT_EQ(prompt, AUDIO_TAG + "x");
    EXPECT_EQ(sequence, (std::vector<size_t>{3}));
}

TEST(MediaTagNormalization, NoTagPrependStartsAtBaseIndex) {
    const auto [prompt, sequence] = normalize_audio("x", 5, 2);
    EXPECT_EQ(sequence, (std::vector<size_t>{5, 6}));
}

// ------------------------------------------------------------------------ error paths

TEST(MediaTagNormalization, IndexBelowBaseIsRejected) {
    EXPECT_THROW(normalize_audio("<ov_genai_audio_0>x", 1, 1), ov::Exception);
}

TEST(MediaTagNormalization, IndexAtOrAboveRangeIsRejected) {
    EXPECT_THROW(normalize_audio("<ov_genai_audio_5>x", 0, 1), ov::Exception);
}

TEST(MediaTagNormalization, TagWithNoMediaIsRejected) {
    EXPECT_THROW(normalize_audio("<ov_genai_audio_0>x", 0, 0), ov::Exception);
}

TEST(MediaTagNormalization, MixingUniversalAndNativeIsRejected) {
    EXPECT_THROW(normalize_audio(AUDIO_TAG + "<ov_genai_audio_0>", 0, 1), ov::Exception);
}

TEST(MediaTagNormalization, NativeTagCountMustMatchMediaCount) {
    EXPECT_THROW(normalize_audio(AUDIO_TAG + AUDIO_TAG, 0, 1), ov::Exception);
    EXPECT_THROW(normalize_audio(AUDIO_TAG, 0, 2), ov::Exception);
}

// Error messages are load-bearing: the Python suite pins these substrings, and a reworded message
// sends users to the wrong modality (plan risk R2).
TEST(MediaTagNormalization, ErrorMessagesNameAllThreeModalities) {
    try {
        normalize_audio("<ov_genai_audio_0>x", 1, 1);
        FAIL() << "expected a throw";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("images/videos/audios"), std::string::npos) << e.what();
    }
    try {
        normalize_audio("<ov_genai_audio_5>x", 0, 1);
        FAIL() << "expected a throw";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("Missing image/video/audio with index"), std::string::npos) << e.what();
    }
    try {
        normalize_audio(AUDIO_TAG + AUDIO_TAG, 0, 1);
        FAIL() << "expected a throw";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("native media tags"), std::string::npos) << e.what();
    }
}

// ------------------------------------------------- audio must not share the video regex

// Guards the ternary that adding AUDIO exposed: `modality == IMAGE ? IMAGE : VIDEO` routed audio
// through the video regex.
TEST(MediaTagNormalization, AudioModalityIgnoresImageAndVideoTags) {
    const auto [prompt, sequence] = normalize_audio("<ov_genai_video_0><ov_genai_image_0>x", 0, 1);
    EXPECT_EQ(prompt, AUDIO_TAG + "<ov_genai_video_0><ov_genai_image_0>x")
        << "video/image tags are inert under AUDIO, so the audio falls back to prepend";
    EXPECT_EQ(sequence, (std::vector<size_t>{0}));
}

// ------------------------------------------- rebasing absolute indices onto one turn

// The SDPA prompt path shipped audio without this rebase, so turn 2 asked for index 1 of a
// one-element list. Gated here because the end-to-end SDPA path cannot run.
TEST(MediaTagNormalization, RebaseShiftsIndicesOntoThisTurn) {
    std::vector<size_t> sequence{3, 4, 5};
    rebase_media_sequence(sequence, 3);
    EXPECT_EQ(sequence, (std::vector<size_t>{0, 1, 2}));
}

TEST(MediaTagNormalization, RebaseWithZeroBaseIsIdentity) {
    std::vector<size_t> sequence{0, 1, 2};
    rebase_media_sequence(sequence, 0);
    EXPECT_EQ(sequence, (std::vector<size_t>{0, 1, 2}));
}

TEST(MediaTagNormalization, RebaseHandlesEmptyAndOutOfOrderSequences) {
    std::vector<size_t> empty;
    rebase_media_sequence(empty, 7);
    EXPECT_TRUE(empty.empty());

    // Out-of-order indices must keep their order; only the offset changes.
    std::vector<size_t> reversed{5, 4};
    rebase_media_sequence(reversed, 4);
    EXPECT_EQ(reversed, (std::vector<size_t>{1, 0}));
}

// An index below the base would underflow size_t and index far out of bounds, so it must throw
// rather than wrap.
TEST(MediaTagNormalization, RebaseRejectsIndexBelowBase) {
    std::vector<size_t> sequence{2};
    EXPECT_THROW(rebase_media_sequence(sequence, 3), ov::Exception);
}

// The full chain a second turn goes through: absolute indices out of normalize, then rebased.
TEST(MediaTagNormalization, SecondTurnAbsoluteIndicesRebaseToLocalSlots) {
    // Turn 2 supplies one audio; its absolute index is 1 because turn 1 consumed index 0.
    auto [prompt, sequence] = normalize_audio("Second <ov_genai_audio_1>", 1, 1);
    EXPECT_EQ(sequence, (std::vector<size_t>{1})) << "normalize returns absolute indices";

    rebase_media_sequence(sequence, 1);
    EXPECT_EQ(sequence, (std::vector<size_t>{0})) << "rebased onto this turn's single-element list";
}

// ------------------------------------------------- known gap, pinned rather than fixed

// Under-tagging drops the unreferenced item silently, because verify_ids only checks the indices
// present. Shared by every modality, so pinned rather than fixed — tightening it needs its own ticket.
TEST(MediaTagNormalization, UnderTaggingSilentlyDropsMediaForEveryModality) {
    // Two audios supplied, only index 0 referenced (twice). Audio 1 is dropped without error.
    const auto audio = normalize_audio("<ov_genai_audio_0> and <ov_genai_audio_0>", 0, 2);
    EXPECT_EQ(audio.second, (std::vector<size_t>{0, 0})) << "audio 1 is silently absent";

    // Identical for images, which is the point: audio did not introduce this.
    const std::string img = "<img>";
    const auto image = normalize_media_tags("<ov_genai_image_0> and <ov_genai_image_0>", img, img, 0, 2,
                                            ModalityType::IMAGE);
    EXPECT_EQ(image.second, (std::vector<size_t>{0, 0})) << "image 1 is silently absent";

    // A native tag in the same position DOES enforce the count, so the two schemes differ.
    EXPECT_THROW(normalize_audio(AUDIO_TAG, 0, 2), ov::Exception);
}

TEST(MediaTagNormalization, EachModalityResolvesOnlyItsOwnTag) {
    const std::string img = "<img>";
    const std::string vid = "<vid>";
    const auto as_image = normalize_media_tags("<ov_genai_image_0>x", img, img, 0, 1, ModalityType::IMAGE);
    EXPECT_EQ(as_image.first, img + "x");

    const auto as_video = normalize_media_tags("<ov_genai_video_0>x", vid, vid, 0, 1, ModalityType::VIDEO);
    EXPECT_EQ(as_video.first, vid + "x");

    // An image tag under VIDEO is inert, so the video prepends instead of substituting.
    const auto crossed = normalize_media_tags("<ov_genai_image_0>x", vid, vid, 0, 1, ModalityType::VIDEO);
    EXPECT_EQ(crossed.first, vid + "<ov_genai_image_0>x");
}
