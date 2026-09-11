// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "automatic_speech_recognition/models/qwen3-asr/qwen3_forced_aligner.hpp"

using ov::genai::fix_timestamp;
using ov::genai::normalize_alignment_language;
using ov::genai::segment_alignment_units;

using Units = std::vector<std::string>;

TEST(Qwen3ForcedAlignerFixTimestamp, MonotoneUnchanged) {
    const std::vector<int64_t> data{480, 560, 560, 720, 720, 800, 800, 1120, 1200, 1600};
    EXPECT_EQ(fix_timestamp(data), data);
}

TEST(Qwen3ForcedAlignerFixTimestamp, ShortAnomalySnapsToNearerAnchor) {
    EXPECT_EQ(fix_timestamp({10, 100, 20, 30}), (std::vector<int64_t>{10, 10, 20, 30}));
}

TEST(Qwen3ForcedAlignerFixTimestamp, LongAnomalyRunInterpolates) {
    EXPECT_EQ(fix_timestamp({10, 500, 500, 500, 20, 30, 40, 50}),
              (std::vector<int64_t>{10, 12, 15, 17, 20, 30, 40, 50}));
}

TEST(Qwen3ForcedAlignerFixTimestamp, OneSidedLeadingRunFillsFromRight) {
    EXPECT_EQ(fix_timestamp({50, 10, 10, 10}), (std::vector<int64_t>{10, 10, 10, 10}));
    EXPECT_EQ(fix_timestamp({}), (std::vector<int64_t>{}));
}

TEST(Qwen3ForcedAlignerFixTimestamp, OneSidedTrailingRunFillsFromLeft) {
    EXPECT_EQ(fix_timestamp({10, 20, 30, 5, 5, 5}),
              (std::vector<int64_t>{10, 20, 30, 30, 30, 30}));
}

// Reference class IDs from qwen-asr==0.0.6 for "how are you doing today" (80 ms per class).
TEST(Qwen3ForcedAlignerFixTimestamp, ReferenceClassIdsMapToSpans) {
    const std::vector<int64_t> class_ids{6, 7, 7, 9, 9, 10, 10, 14, 15, 20};
    std::vector<int64_t> ms;
    for (int64_t id : class_ids) {
        ms.push_back(id * 80);
    }
    const std::vector<int64_t> fixed = fix_timestamp(ms);
    EXPECT_EQ(fixed, ms);
    const std::vector<std::pair<int64_t, int64_t>> expected{
        {480, 560}, {560, 720}, {720, 800}, {800, 1120}, {1200, 1600}};
    for (size_t unit = 0; unit < expected.size(); ++unit) {
        EXPECT_EQ(fixed[unit * 2], expected[unit].first);
        EXPECT_EQ(fixed[unit * 2 + 1], expected[unit].second);
    }
}

TEST(Qwen3ForcedAlignerTokenize, EnglishWhitespacePunctuationApostropheHyphen) {
    EXPECT_EQ(segment_alignment_units("how are you doing today", "english"), (Units{"how", "are", "you", "doing", "today"}));
    EXPECT_EQ(segment_alignment_units("it's a test.", "english"), (Units{"it's", "a", "test"}));
    // Reference preprocessing removes hyphens rather than splitting on them.
    EXPECT_EQ(segment_alignment_units("well-known thing", "english"), (Units{"wellknown", "thing"}));
    EXPECT_EQ(segment_alignment_units("  spaced   out  ", "english"), (Units{"spaced", "out"}));
    EXPECT_TRUE(segment_alignment_units("", "english").empty());
    EXPECT_TRUE(segment_alignment_units("!!! ??? ...", "english").empty());
}

TEST(Qwen3ForcedAlignerTokenize, ChineseAndMixedLatin) {
    EXPECT_EQ(segment_alignment_units("\u4eca\u5929\u5929\u6c14", "chinese"), (Units{"\u4eca", "\u5929", "\u5929", "\u6c14"}));
    EXPECT_EQ(segment_alignment_units("hello \u4e16\u754c", "chinese"), (Units{"hello", "\u4e16", "\u754c"}));
    EXPECT_EQ(segment_alignment_units("\u4eca\u59293\u70b9", "chinese"), (Units{"\u4eca", "\u5929", "3", "\u70b9"}));
}

TEST(Qwen3ForcedAlignerTokenize, AccentedLatinAndCyrillicKept) {
    EXPECT_EQ(segment_alignment_units("Gr\u00fc\u00dfe, Welt!", "english"), (Units{"Gr\u00fc\u00dfe", "Welt"}));
    EXPECT_EQ(segment_alignment_units("\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440", "english"), (Units{"\u041f\u0440\u0438\u0432\u0435\u0442", "\u043c\u0438\u0440"}));
}

TEST(Qwen3ForcedAlignerTokenize, UnicodeWhitespaceSplitsWords) {
    // Matches Python str.split() whitespace semantics used by the reference processor.
    EXPECT_EQ(segment_alignment_units("hello\u3000world", "english"), (Units{"hello", "world"}));  // ideographic space
    EXPECT_EQ(segment_alignment_units("hello\u00a0world", "english"), (Units{"hello", "world"}));  // no-break space
    EXPECT_EQ(segment_alignment_units("hello\u202fworld", "english"), (Units{"hello", "world"}));  // narrow no-break space
    EXPECT_EQ(segment_alignment_units("hello\u2009world", "english"), (Units{"hello", "world"}));  // thin space
    // Python str.split() also treats U+001C-U+001F as whitespace.
    EXPECT_EQ(segment_alignment_units("hello\x1f" "world", "english"), (Units{"hello", "world"}));
    EXPECT_EQ(segment_alignment_units("a b\tc\nd", "english"), (Units{"a", "b", "c", "d"}));
}

TEST(Qwen3ForcedAlignerTokenize, UnicodeNumbersKeptAndCombiningMarksDropped) {
    // Keep representative Unicode category-N code points used by the reference processor.
    EXPECT_EQ(segment_alignment_units("\u00bd", "english"), (Units{"\u00bd"}));    // vulgar fraction one half
    EXPECT_EQ(segment_alignment_units("x\u00b2", "english"), (Units{"x\u00b2"}));  // superscript two
    EXPECT_EQ(segment_alignment_units("\u2163", "english"), (Units{"\u2163"}));    // roman numeral four
    // U+3007 is category Nl and remains a separate CJK alignment unit.
    EXPECT_EQ(segment_alignment_units("\u4e8c\u3007\u4e8c\u4e94", "english"),
              (Units{"\u4e8c", "\u3007", "\u4e8c", "\u4e94"}));
    // Combining marks are not Unicode L/N and are removed.
    EXPECT_EQ(segment_alignment_units("e\u0301", "english"), (Units{"e"}));
}

TEST(Qwen3ForcedAlignerSegment, UnimplementedLanguageStrategiesRejected) {
    EXPECT_ANY_THROW(segment_alignment_units("\u3053\u3093\u306b\u3061\u306f", "japanese"));
    EXPECT_ANY_THROW(segment_alignment_units("\uc548\ub155", "korean"));
}

TEST(Qwen3ForcedAlignerLanguage, NormalizationAliases) {
    EXPECT_EQ(normalize_alignment_language("English"), "english");
    EXPECT_EQ(normalize_alignment_language("CHINESE"), "chinese");
    EXPECT_EQ(normalize_alignment_language("en"), "english");
    EXPECT_EQ(normalize_alignment_language("zh"), "chinese");
    // Surrounding whitespace is trimmed before normalization.
    EXPECT_EQ(normalize_alignment_language("  English  "), "english");
    EXPECT_EQ(normalize_alignment_language("\ten\t"), "english");
    EXPECT_EQ(normalize_alignment_language("Japanese"), "japanese");
    EXPECT_EQ(normalize_alignment_language("ja"), "japanese");
    EXPECT_EQ(normalize_alignment_language("Korean"), "korean");
    EXPECT_EQ(normalize_alignment_language("ko"), "korean");
    // Recognized Qwen3-ASR languages remain canonicalized even when the aligner does not support them.
    EXPECT_EQ(normalize_alignment_language("Thai"), "thai");
    // Unknown or missing languages have no canonical value.
    EXPECT_EQ(normalize_alignment_language("Klingon"), "");
    EXPECT_EQ(normalize_alignment_language(""), "");
    EXPECT_EQ(normalize_alignment_language("   "), "");
}
