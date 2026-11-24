// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "automatic_speech_recognition/vad/silero_vad.hpp"

namespace {

struct Segment {
    int64_t start = 0;
    int64_t end = 0;
};

double samples_from_ms(size_t milliseconds) {
    return static_cast<double>(ov::genai::silero_vad::SAMPLE_RATE) * static_cast<double>(milliseconds) / 1000.0;
}

}  // namespace

namespace ov::genai::silero_vad {

// Adapted from get_speech_timestamps_from_probs() in Silero VAD v6.3 (MIT License,
// Copyright (c) 2020-present Silero Team):
// https://github.com/snakers4/silero-vad/blob/63d4a2a7bfce8cd9bb39ed8fa0afcb5236d73e01/src/silero_vad/utils_vad.py#L421-L591
std::vector<SpeechSegment> segments_from_probabilities(const std::vector<float>& probabilities,
                                                       size_t audio_length_samples,
                                                       const SileroVADConfig& config) {
    const int64_t window = static_cast<int64_t>(WINDOW_SAMPLES);
    const int64_t audio_length = static_cast<int64_t>(audio_length_samples);
    const double min_speech_samples = samples_from_ms(config.min_speech_duration_ms);
    const double speech_pad_samples = samples_from_ms(config.speech_pad_ms);
    const double max_speech_samples = static_cast<double>(SAMPLE_RATE) * config.max_speech_duration_s -
                                      static_cast<double>(window) - 2.0 * speech_pad_samples;
    const double min_silence_samples = samples_from_ms(config.min_silence_duration_ms);
    const double min_silence_samples_at_max_speech = samples_from_ms(config.min_silence_at_max_speech_ms);
    const double threshold = config.threshold;
    const double neg_threshold = config.neg_threshold.value_or(std::max(threshold - 0.15, 0.01));

    bool triggered = false;
    std::vector<Segment> speeches;
    std::optional<Segment> current;
    int64_t temp_end = 0;
    std::vector<std::pair<int64_t, int64_t>> possible_ends;

    const auto reset_split_state = [&]() {
        temp_end = 0;
        possible_ends.clear();
    };

    for (size_t index = 0; index < probabilities.size(); ++index) {
        const double probability = probabilities[index];
        const int64_t cur_sample = window * static_cast<int64_t>(index);

        // Speech resumed after a pause too short to end the segment; a pause longer than
        // min_silence_at_max_speech_ms becomes a candidate split point for max_speech_duration_s.
        if (probability >= threshold && temp_end != 0) {
            const int64_t silence = cur_sample - temp_end;
            if (static_cast<double>(silence) > min_silence_samples_at_max_speech) {
                possible_ends.emplace_back(temp_end, silence);
            }
            temp_end = 0;
        }

        if (probability >= threshold && !triggered) {
            triggered = true;
            current = Segment{cur_sample, 0};
            continue;
        }

        // Split an over-long segment at its longest candidate pause. Without one, hard-cut here and,
        // unless already in a pause or below neg_threshold, start the next segment at the cut.
        if (triggered && static_cast<double>(cur_sample - current->start) > max_speech_samples) {
            if (!possible_ends.empty()) {
                const auto longest = std::max_element(possible_ends.begin(),
                                                      possible_ends.end(),
                                                      [](const auto& lhs, const auto& rhs) {
                                                          return lhs.second < rhs.second;
                                                      });
                const int64_t prev_end = longest->first;
                const int64_t next_start = prev_end + longest->second;
                current->end = prev_end;
                speeches.push_back(*current);
                current.reset();
                if (next_start < prev_end + cur_sample) {
                    current = Segment{next_start, 0};
                } else {
                    triggered = false;
                }
                reset_split_state();
            } else {
                current->end = cur_sample;
                speeches.push_back(*current);
                current.reset();
                if (temp_end != 0 || probability < neg_threshold) {
                    triggered = false;
                } else {
                    current = Segment{cur_sample, 0};
                }
                reset_split_state();
                continue;
            }
        }

        if (probability < neg_threshold && triggered) {
            if (temp_end == 0) {
                temp_end = cur_sample;
            }
            const int64_t silence_now = cur_sample - temp_end;
            if (static_cast<double>(silence_now) < min_silence_samples) {
                continue;
            }
            current->end = temp_end;
            if (static_cast<double>(current->end - current->start) > min_speech_samples) {
                speeches.push_back(*current);
            }
            current.reset();
            reset_split_state();
            triggered = false;
            continue;
        }
    }

    if (current && static_cast<double>(audio_length - current->start) > min_speech_samples) {
        current->end = audio_length;
        speeches.push_back(*current);
    }

    const auto pad_start = [&](int64_t start) {
        return static_cast<int64_t>(std::max(0.0, static_cast<double>(start) - speech_pad_samples));
    };
    const auto pad_end = [&](int64_t end) {
        return static_cast<int64_t>(
            std::min(static_cast<double>(audio_length), static_cast<double>(end) + speech_pad_samples));
    };

    for (size_t index = 0; index < speeches.size(); ++index) {
        Segment& speech = speeches[index];
        if (index == 0) {
            speech.start = pad_start(speech.start);
        }
        if (index + 1 != speeches.size()) {
            Segment& next = speeches[index + 1];
            const int64_t silence = next.start - speech.end;
            // Split gaps narrower than both paddings between the neighbors so they cannot overlap.
            if (static_cast<double>(silence) < 2.0 * speech_pad_samples) {
                speech.end += silence / 2;
                next.start -= silence / 2;
            } else {
                speech.end = pad_end(speech.end);
                next.start = pad_start(next.start);
            }
        } else {
            speech.end = pad_end(speech.end);
        }
    }

    std::vector<SpeechSegment> segments;
    segments.reserve(speeches.size());
    for (const Segment& speech : speeches) {
        segments.push_back({static_cast<size_t>(speech.start), static_cast<size_t>(speech.end)});
    }
    return segments;
}

}  // namespace ov::genai::silero_vad
