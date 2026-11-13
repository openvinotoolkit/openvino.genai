// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "automatic_speech_recognition/audio_chunk.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"

namespace ov::genai {

// Transcription of a window of audio or of a whole audio. `language` is empty if it was not recognized.
struct SenseVoiceTranscript {
    std::string text;
    std::string language;
    std::optional<std::string> emotion;
    std::optional<std::string> event;
    size_t samples = 0;
};

/**
 * A window of audio. Audio that fits into one window is borrowed from the input and must outlive the window,
 * a window of a longer audio owns its samples.
 */
struct AudioWindow {
    size_t audio_index = 0;
    const std::vector<float>* borrowed = nullptr;
    std::vector<float> owned;

    const std::vector<float>& samples() const {
        return borrowed ? *borrowed : owned;
    }
};

/**
 * Splits each audio longer than `max_window_samples` into windows at low-energy positions.
 * Shorter audio is a single window. `audio_index` of a window is the index of its audio.
 */
std::vector<AudioWindow> split_audio_into_windows(const AudiosView& audios, size_t max_window_samples);

/**
 * Merges transcripts of the windows of one audio. Texts are joined, language, emotion and event are chosen by the
 * largest total window duration. A language other than "auto" in `forced_language` is used as is.
 *
 * Limitation: when streaming, tokens of consecutive windows are written without a separator, so a space between
 * windows of space-delimited languages can be missing in the streamed text compared to the merged text.
 */
SenseVoiceTranscript merge_transcripts(const std::vector<SenseVoiceTranscript>& windows,
                                       const std::optional<std::string>& forced_language);

}  // namespace ov::genai
