// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <functional>

#include "automatic_speech_recognition/models/sensevoice/windowing.hpp"

using ov::genai::AudioWindow;
using ov::genai::SenseVoiceTranscript;

namespace {

constexpr size_t kSamplingRate = 16000;

SenseVoiceTranscript make_transcript(const std::string& text,
                                     const std::string& language,
                                     std::optional<std::string> emotion,
                                     std::optional<std::string> event,
                                     size_t samples) {
    SenseVoiceTranscript transcript;
    transcript.text = text;
    transcript.language = language;
    transcript.emotion = std::move(emotion);
    transcript.event = std::move(event);
    transcript.samples = samples;
    return transcript;
}

}  // namespace

TEST(SenseVoiceWindowing, AudioWithinLimitIsBorrowed) {
    const std::vector<float> audio(5 * kSamplingRate, 0.0f);
    const ov::genai::AudiosView audios{std::cref(audio)};

    const std::vector<AudioWindow> windows = ov::genai::split_audio_into_windows(audios, 5 * kSamplingRate);

    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0].audio_index, 0u);
    EXPECT_EQ(&windows[0].samples(), &audio);
}

TEST(SenseVoiceWindowing, LongAudioIsSplitIntoOwnedWindowsWithinLimit) {
    const std::vector<float> short_audio(kSamplingRate, 0.0f);
    const std::vector<float> long_audio(25 * kSamplingRate, 0.0f);
    const size_t limit = 10 * kSamplingRate;
    const ov::genai::AudiosView audios{std::cref(long_audio), std::cref(short_audio)};

    const std::vector<AudioWindow> windows = ov::genai::split_audio_into_windows(audios, limit);

    size_t long_audio_samples = 0;
    size_t long_audio_windows = 0;
    size_t short_audio_windows = 0;
    for (const AudioWindow& window : windows) {
        EXPECT_LE(window.samples().size(), limit);
        if (window.audio_index == 0) {
            EXPECT_EQ(window.borrowed, nullptr);
            long_audio_samples += window.samples().size();
            ++long_audio_windows;
        } else {
            EXPECT_EQ(&window.samples(), &short_audio);
            ++short_audio_windows;
        }
    }
    EXPECT_GT(long_audio_windows, 1u);
    EXPECT_EQ(long_audio_samples, long_audio.size());
    EXPECT_EQ(short_audio_windows, 1u);
    EXPECT_EQ(windows.back().audio_index, 1u);
}

TEST(SenseVoiceWindowing, ShortTailKeepsItsRealLength) {
    const std::vector<float> audio(kSamplingRate + kSamplingRate / 10, 0.0f);
    const ov::genai::AudiosView audios{std::cref(audio)};

    const std::vector<AudioWindow> windows = ov::genai::split_audio_into_windows(audios, kSamplingRate);

    ASSERT_EQ(windows.size(), 2u);
    EXPECT_EQ(windows[0].samples().size(), kSamplingRate);
    EXPECT_EQ(windows[1].samples().size(), kSamplingRate / 10);
}

TEST(SenseVoiceWindowing, OneSampleTailIsDropped) {
    const std::vector<float> audio(kSamplingRate + 1, 0.0f);
    const ov::genai::AudiosView audios{std::cref(audio)};

    const std::vector<AudioWindow> windows = ov::genai::split_audio_into_windows(audios, kSamplingRate);

    ASSERT_EQ(windows.size(), 1u);
    EXPECT_EQ(windows[0].samples().size(), kSamplingRate);
}

TEST(SenseVoiceMerge, JoinsTextsWithSpaceAndSkipsEmptyWindows) {
    const std::vector<SenseVoiceTranscript> windows = {
        make_transcript("hello", "en", "NEUTRAL", "Speech", 100),
        make_transcript("", "en", "NEUTRAL", "Speech", 100),
        make_transcript("world", "en", "NEUTRAL", "Speech", 100),
    };

    const SenseVoiceTranscript merged = ov::genai::merge_transcripts(windows, std::nullopt);

    EXPECT_EQ(merged.text, "hello world");
    EXPECT_EQ(merged.samples, 300u);
}

TEST(SenseVoiceMerge, JoinsWithoutSpaceForLanguagesWithoutWordSeparators) {
    for (const std::string language : {"zh", "ja", "yue"}) {
        const std::vector<SenseVoiceTranscript> windows = {
            make_transcript("ab", language, std::nullopt, std::nullopt, 100),
            make_transcript("cd", language, std::nullopt, std::nullopt, 100),
        };
        EXPECT_EQ(ov::genai::merge_transcripts(windows, std::nullopt).text, "abcd") << language;
    }
}

TEST(SenseVoiceMerge, LanguageIsChosenByTotalDuration) {
    const std::vector<SenseVoiceTranscript> windows = {
        make_transcript("a", "en", std::nullopt, std::nullopt, 100),
        make_transcript("b", "zh", std::nullopt, std::nullopt, 150),
        make_transcript("c", "en", std::nullopt, std::nullopt, 100),
        make_transcript("d", "", std::nullopt, std::nullopt, 1000),
    };

    EXPECT_EQ(ov::genai::merge_transcripts(windows, std::nullopt).language, "en");
}

TEST(SenseVoiceMerge, ForcedLanguageIsUsedUnlessAuto) {
    const std::vector<SenseVoiceTranscript> windows = {
        make_transcript("a", "en", std::nullopt, std::nullopt, 100),
    };

    EXPECT_EQ(ov::genai::merge_transcripts(windows, "ja").language, "ja");
    EXPECT_EQ(ov::genai::merge_transcripts(windows, "auto").language, "en");
}

TEST(SenseVoiceMerge, UnknownEmotionAndSpeechEventAreFallbacksOnly) {
    const std::vector<SenseVoiceTranscript> informative = {
        make_transcript("a", "en", "EMO_UNKNOWN", "Speech", 900),
        make_transcript("b", "en", "HAPPY", "Laughter", 100),
    };
    const SenseVoiceTranscript merged = ov::genai::merge_transcripts(informative, std::nullopt);
    EXPECT_EQ(merged.emotion, "HAPPY");
    EXPECT_EQ(merged.event, "Laughter");

    const std::vector<SenseVoiceTranscript> only_defaults = {
        make_transcript("a", "en", "EMO_UNKNOWN", "Speech", 100),
        make_transcript("b", "en", "EMO_UNKNOWN", "Speech", 100),
    };
    const SenseVoiceTranscript fallback = ov::genai::merge_transcripts(only_defaults, std::nullopt);
    EXPECT_EQ(fallback.emotion, "EMO_UNKNOWN");
    EXPECT_EQ(fallback.event, "Speech");
}

TEST(SenseVoiceMerge, MissingMetadataStaysUnset) {
    const std::vector<SenseVoiceTranscript> windows = {
        make_transcript("a", "en", std::nullopt, std::nullopt, 100),
    };

    const SenseVoiceTranscript merged = ov::genai::merge_transcripts(windows, std::nullopt);

    EXPECT_FALSE(merged.emotion.has_value());
    EXPECT_FALSE(merged.event.has_value());
}
