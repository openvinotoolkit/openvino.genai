// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <random>

#include "openvino/genai/tokenizer.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_text_committer.hpp"

using ov::genai::qwen3_tts::TextCommitter;
using ov::genai::qwen3_tts::tokenize_target_segment;

namespace {

std::string drain(TextCommitter& committer) {
    return committer.take();
}

}  // namespace

TEST(Qwen3TTSTextCommitter, HoldsBackThePartialWordAndTheSpaceInFrontOfIt) {
    TextCommitter committer;
    committer.push("Hello wor");
    EXPECT_EQ(drain(committer), "Hello");
    EXPECT_EQ(drain(committer), "");
    committer.push("ld, how");
    EXPECT_EQ(drain(committer), " world,");
    committer.end();
    EXPECT_EQ(drain(committer), " how");
    EXPECT_EQ(drain(committer), "");
}

TEST(Qwen3TTSTextCommitter, NeverCutsInsideAWord) {
    TextCommitter committer;
    committer.push("anti");
    EXPECT_EQ(drain(committer), "");
    committer.push("disestablish");
    EXPECT_EQ(drain(committer), "");
    committer.push("ment ");
    EXPECT_EQ(drain(committer), "antidisestablishment");
    EXPECT_EQ(committer.pending_bytes(), 1u);
}

TEST(Qwen3TTSTextCommitter, KeepsAWholeWhitespaceRunTogether) {
    TextCommitter committer;
    committer.push("a   b");
    EXPECT_EQ(drain(committer), "a");
    committer.push(" c");
    EXPECT_EQ(drain(committer), "   b");
}

TEST(Qwen3TTSTextCommitter, ALineBreakAfterPunctuationIsNotACutPoint) {
    TextCommitter committer;
    committer.push("Hi.\nNext");
    EXPECT_EQ(drain(committer), "");  // ".\n" is one pre-token
    committer.push(" more");
    EXPECT_EQ(drain(committer), "Hi.\nNext");

    TextCommitter after_word;
    after_word.push("line one\nline");
    EXPECT_EQ(drain(after_word), "line one");  // a line break after a word is a boundary
}

TEST(Qwen3TTSTextCommitter, CommitsALongRunWithoutSpacesAtTheCap) {
    TextCommitter committer(8);
    committer.push("123456789");  // 9 characters held back, above the cap
    EXPECT_EQ(drain(committer), "123456789");

    TextCommitter chinese(4);
    chinese.push("\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96");  // 3 characters: held back
    EXPECT_EQ(drain(chinese), "");
    chinese.push("\xE7\x95\x8C\xE4\xBD\xA0");  // 5 characters
    EXPECT_EQ(drain(chinese), "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C\xE4\xBD\xA0");
}

TEST(Qwen3TTSTextCommitter, NeverCommitsAHalfCharacter) {
    TextCommitter committer(0);  // no holdback: everything complete is committed at once
    committer.push("ab\xE4\xBD");  // the first two bytes of a three-byte character
    EXPECT_EQ(drain(committer), "ab");
    committer.push("\xA0 x");
    EXPECT_EQ(drain(committer), "\xE4\xBD\xA0 x");
}

TEST(Qwen3TTSTextCommitter, EndReleasesEverythingAndClosesTheInput) {
    TextCommitter committer;
    committer.push("tail");
    EXPECT_EQ(drain(committer), "");
    committer.end();
    EXPECT_TRUE(committer.ended());
    EXPECT_EQ(drain(committer), "tail");
    EXPECT_THROW(committer.push("more"), ov::Exception);
    EXPECT_THROW(committer.end(), ov::Exception);
}

TEST(Qwen3TTSTextCommitter, TheSegmentsAddUpToTheText) {
    const std::string text = "The quick brown fox, jumps over 12 lazy dogs.\nThen it   sleeps. ";
    std::mt19937 rng(7);
    for (int trial = 0; trial < 200; ++trial) {
        TextCommitter committer;
        std::string rebuilt;
        size_t pos = 0;
        while (pos < text.size()) {
            const size_t n = 1 + rng() % 9;
            committer.push(text.substr(pos, n));
            pos += n;
            rebuilt += committer.take();
        }
        committer.end();
        rebuilt += committer.take();
        ASSERT_EQ(rebuilt, text);
    }
}

TEST(Qwen3TTSTextCommitter, TargetSegmentKeepsTheTokensBetweenTheMarkers) {
    auto encode = [](const std::string& text) {
        // A stand-in tokenizer: one token per character, plus markers for the template tokens.
        std::vector<int64_t> ids{1000};
        for (unsigned char c : text.substr(12, text.size() - 12 - 10)) {
            ids.push_back(c);
        }
        ids.push_back(1001);
        return ids;
    };
    EXPECT_EQ(tokenize_target_segment(encode, "ab"), (std::vector<int64_t>{'a', 'b'}));
    EXPECT_TRUE(tokenize_target_segment(encode, "").empty());
}

// With a real tokenizer (QWEN3_TTS_TOKENIZER_DIR: a Qwen3-TTS OpenVINO model directory with openvino_tokenizer.xml):
// the tokens of the committed segments must equal the tokens of the complete text, for any way the text arrives.
TEST(Qwen3TTSTextCommitter, SegmentsTokenizeLikeTheCompleteText) {
    const char* dir = std::getenv("QWEN3_TTS_TOKENIZER_DIR");
    if (dir == nullptr) {
        GTEST_SKIP() << "QWEN3_TTS_TOKENIZER_DIR is not set";
    }
    ov::genai::Tokenizer tokenizer(dir);
    auto encode = [&](const std::string& text) {
        const ov::Tensor ids = tokenizer.encode(text).input_ids;
        const int64_t* data = ids.data<const int64_t>();
        return std::vector<int64_t>(data, data + ids.get_shape()[1]);
    };
    // The tokens of the target text as the complete-text prompt has them (see Qwen3TTSCodecEngine::build_prefill).
    auto complete_text_tokens = [&](const std::string& text) {
        const auto ids = encode("<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n");
        return std::vector<int64_t>(ids.begin() + 3, ids.end() - 5);
    };

    const std::vector<std::string> texts = {
        "Hello, this is a test of incremental speech synthesis. Does it sound the same?",
        "It's 3:45 p.m. on 12/31/2026 -- don't forget the (very) important meeting!",
        "First line.\nSecond line, with   extra   spaces.\n\nA new paragraph \"quoted\" and more...",
        "A longword_with_underscores and CamelCaseIdentifiers and URLs like https://example.com/a?b=c#d.",
        "Price: $1,234.56; ratio 3.14159; tabs\tand\tmore\ttabs; trailing space ",
        "Caf\xC3\xA9 cr\xC3\xA8me br\xC3\xBBl\xC3\xA9""e \xE2\x80\x94 na\xC3\xAFve fa\xC3\xA7""ade, \xE2\x82\xAC""5.",
    };
    std::mt19937 rng(3);
    for (const auto& text : texts) {
        const auto expected = complete_text_tokens(text);
        for (int trial = 0; trial < 60; ++trial) {
            TextCommitter committer;  // default holdback; the words here are shorter than the cap
            std::vector<int64_t> got;
            auto add = [&](const std::string& segment) {
                const auto ids = tokenize_target_segment(encode, segment);
                got.insert(got.end(), ids.begin(), ids.end());
            };
            size_t pos = 0;
            const size_t max_piece = trial < 20 ? 1 : (trial < 40 ? 5 : 17);
            while (pos < text.size()) {
                const size_t n = 1 + rng() % max_piece;
                committer.push(text.substr(pos, n));
                pos += n;
                add(committer.take());
            }
            committer.end();
            add(committer.take());
            ASSERT_EQ(got, expected) << "text: " << text << "\ntrial " << trial;
        }
    }

    // Text with no spaces and a cap that is never reached is committed at end() as one segment: same tokens.
    const std::string chinese = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xBB\x8A\xE5\xA4\xA9\xE5\xA4\xA9\xE6\xB0\x94\xE5\xBE\x88\xE5\xA5\xBD\xE3\x80\x82";
    TextCommitter committer(1000);
    std::vector<int64_t> got;
    for (size_t pos = 0; pos < chinese.size(); pos += 3) {
        committer.push(chinese.substr(pos, 3));
        const auto ids = tokenize_target_segment(encode, committer.take());
        got.insert(got.end(), ids.begin(), ids.end());
    }
    committer.end();
    const auto ids = tokenize_target_segment(encode, committer.take());
    got.insert(got.end(), ids.begin(), ids.end());
    EXPECT_EQ(got, complete_text_tokens(chinese));
}
