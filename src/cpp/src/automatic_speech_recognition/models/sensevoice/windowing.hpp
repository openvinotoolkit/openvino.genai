// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "automatic_speech_recognition/pipeline_base.hpp"

namespace ov::genai {

// Part of an audio that is transcribed by a single model inference.
struct SenseVoiceWindow {
    size_t audio_index;
    const std::vector<float>* samples;
};

// Windows of all audios in the input order. `storage` owns samples of windows cut from longer audio.
struct SenseVoiceWindows {
    std::vector<SenseVoiceWindow> windows;
    std::deque<std::vector<float>> storage;
};

// Transcription of a window or of a whole audio. `language` is empty if it was not recognized.
struct SenseVoiceTranscript {
    std::string text;
    std::string language;
    std::optional<std::string> emotion;
    std::optional<std::string> event;
    size_t samples = 0;
};

/**
 * Defines how audio is fed to the model: which inputs the model accepts and how audio that does not fit them
 * is split into windows and merged back.
 */
class SenseVoiceWindowing {
public:
    virtual ~SenseVoiceWindowing() = default;

    // Maximum number of windows in one inference.
    virtual size_t max_batch() const = 0;
    // Features of shorter windows are zero-padded to this number of frames.
    virtual size_t padded_frames() const = 0;
    virtual SenseVoiceWindows split(const AudiosView& audios) const = 0;
    // Merges transcripts of all windows of one audio. A language other than "auto" in `forced_language` is used as is.
    virtual SenseVoiceTranscript merge(const std::vector<SenseVoiceTranscript>& windows,
                                       const std::optional<std::string>& forced_language) const = 0;
};

}  // namespace ov::genai
