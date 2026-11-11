// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/models/sensevoice/windowing.hpp"

#include <algorithm>
#include <utility>

#include "automatic_speech_recognition/models/fun-asr/feature_extractor.hpp"

namespace {

using Votes = std::vector<std::pair<std::string, size_t>>;

// Returns the value with the largest total weight. Values listed in `ignored` are used only if nothing else is voted.
std::optional<std::string> weighted_vote(const Votes& votes, const std::vector<std::string>& ignored) {
    const auto tally = [&](bool skip_ignored) -> std::optional<std::string> {
        Votes totals;
        for (const auto& [value, weight] : votes) {
            if (skip_ignored && std::find(ignored.begin(), ignored.end(), value) != ignored.end()) {
                continue;
            }
            const auto it = std::find_if(totals.begin(), totals.end(), [&](const auto& total) {
                return total.first == value;
            });
            if (it == totals.end()) {
                totals.emplace_back(value, weight);
            } else {
                it->second += weight;
            }
        }
        if (totals.empty()) {
            return std::nullopt;
        }
        return std::max_element(totals.begin(),
                                totals.end(),
                                [](const auto& a, const auto& b) {
                                    return a.second < b.second;
                                })
            ->first;
    };
    if (const auto value = tally(true)) {
        return value;
    }
    return tally(false);
}

}  // namespace

namespace ov::genai {

std::vector<AudioWindow> split_audio_into_windows(const AudiosView& audios, size_t max_window_samples) {
    constexpr size_t sampling_rate = FunASRFeatureExtractor::sampling_rate;
    std::vector<AudioWindow> windows;
    for (size_t audio_index = 0; audio_index < audios.size(); ++audio_index) {
        const std::vector<float>& audio = audios[audio_index];
        if (audio.size() <= max_window_samples) {
            windows.push_back({audio_index, &audio, {}});
            continue;
        }
        const size_t window_seconds = max_window_samples / sampling_rate;
        // A chunk can exceed its nominal duration by up to the boundary search window.
        const size_t search_seconds = std::min(CHUNK_BOUNDARY_SEARCH_SECONDS, window_seconds / 4);
        std::vector<AudioChunk> chunks =
            split_audio_into_chunks({audio}, sampling_rate, window_seconds - search_seconds, search_seconds);
        // Only the last chunk can be shorter than the splitter's minimum length and padded.
        size_t consumed = 0;
        for (size_t i = 0; i + 1 < chunks.size(); ++i) {
            consumed += chunks[i].wav.size();
        }
        chunks.back().wav.resize(audio.size() - consumed);
        // Feature extraction needs at least two samples.
        if (chunks.back().wav.size() < 2) {
            chunks.pop_back();
        }
        for (AudioChunk& chunk : chunks) {
            windows.push_back({audio_index, nullptr, std::move(chunk.wav)});
        }
    }
    return windows;
}

SenseVoiceTranscript merge_transcripts(const std::vector<SenseVoiceTranscript>& windows,
                                       const std::optional<std::string>& forced_language) {
    Votes languages;
    Votes emotions;
    Votes events;
    SenseVoiceTranscript result;
    for (const SenseVoiceTranscript& window : windows) {
        result.samples += window.samples;
        if (!window.language.empty()) {
            languages.emplace_back(window.language, window.samples);
        }
        if (window.emotion.has_value()) {
            emotions.emplace_back(*window.emotion, window.samples);
        }
        if (window.event.has_value()) {
            events.emplace_back(*window.event, window.samples);
        }
    }

    result.language = (forced_language.has_value() && *forced_language != "auto")
                          ? *forced_language
                          : weighted_vote(languages, {}).value_or("");
    result.emotion = weighted_vote(emotions, {"EMO_UNKNOWN"});
    result.event = weighted_vote(events, {"Speech"});

    // Words are not separated by spaces in these languages.
    const bool joined_without_space = result.language == "zh" || result.language == "ja" || result.language == "yue";
    for (const SenseVoiceTranscript& window : windows) {
        if (window.text.empty()) {
            continue;
        }
        if (!result.text.empty() && !joined_without_space) {
            result.text += ' ';
        }
        result.text += window.text;
    }
    return result;
}

}  // namespace ov::genai
