// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/models/sensevoice/npu/static_model.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include "automatic_speech_recognition/audio_chunk.hpp"
#include "automatic_speech_recognition/models/sensevoice/feature_extractor.hpp"
#include "openvino/core/model.hpp"
#include "utils.hpp"

namespace {

using ov::genai::SenseVoiceTranscript;

// Largest audio length in samples whose features fit into `frames` post-LFR frames.
constexpr size_t max_audio_samples_for_frames(size_t frames) {
    using ov::genai::FunASRFeatureExtractor;
    const size_t max_fbank_frames = frames * FunASRFeatureExtractor::lfr_stride;
    return (max_fbank_frames - 1) * FunASRFeatureExtractor::frame_shift + FunASRFeatureExtractor::frame_length +
           FunASRFeatureExtractor::frame_shift - 1;
}

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

class StaticWindowing final : public ov::genai::SenseVoiceWindowing {
public:
    StaticWindowing(size_t max_batch, size_t padded_frames)
        : m_max_batch(max_batch),
          m_padded_frames(padded_frames),
          m_max_audio_samples(max_audio_samples_for_frames(padded_frames)) {}

    size_t max_batch() const override {
        return m_max_batch;
    }

    size_t padded_frames() const override {
        return m_padded_frames;
    }

    ov::genai::SenseVoiceWindows split(const ov::genai::AudiosView& audios) const override {
        using ov::genai::FunASRFeatureExtractor;
        const size_t sampling_rate = FunASRFeatureExtractor::sampling_rate;
        const size_t window_seconds = m_max_audio_samples / sampling_rate;
        // A chunk can exceed its nominal duration by up to the boundary search window.
        const size_t search_seconds = std::min(ov::genai::CHUNK_BOUNDARY_SEARCH_SECONDS, window_seconds / 4);

        ov::genai::SenseVoiceWindows result;
        for (size_t audio_index = 0; audio_index < audios.size(); ++audio_index) {
            const std::vector<float>& audio = audios[audio_index];
            if (audio.size() <= m_max_audio_samples) {
                result.windows.push_back({audio_index, &audio});
                continue;
            }
            for (ov::genai::AudioChunk& chunk :
                 ov::genai::split_audio_into_chunks({audio}, sampling_rate, window_seconds - search_seconds, search_seconds)) {
                result.storage.push_back(std::move(chunk.wav));
                result.windows.push_back({audio_index, &result.storage.back()});
            }
        }
        return result;
    }

    SenseVoiceTranscript merge(const std::vector<SenseVoiceTranscript>& windows,
                               const std::optional<std::string>& forced_language) const override {
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
        const bool joined_without_space =
            result.language == "zh" || result.language == "ja" || result.language == "yue";
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

private:
    size_t m_max_batch;
    size_t m_padded_frames;
    size_t m_max_audio_samples;
};

}  // namespace

namespace ov::genai {

ov::CompiledModel compile_sensevoice_for_npu(const std::filesystem::path& model_path, const ov::AnyMap& properties) {
    constexpr size_t max_audio_seconds = 30;
    constexpr size_t max_audio_samples = max_audio_seconds * FunASRFeatureExtractor::sampling_rate;
    constexpr size_t max_fbank_frames =
        1 + (max_audio_samples - FunASRFeatureExtractor::frame_length) / FunASRFeatureExtractor::frame_shift;
    constexpr size_t max_lfr_frames =
        (max_fbank_frames + FunASRFeatureExtractor::lfr_stride - 1) / FunASRFeatureExtractor::lfr_stride;
    // The model prepends 4 query tokens (language, event/emotion, textnorm) to the encoder sequence.
    // NPU prefers sequence lengths aligned to 16, so the padded input is sized to make the encoder sequence aligned.
    constexpr size_t prefix_query_tokens = 4;
    constexpr size_t sequence_alignment = 16;
    constexpr size_t padded_frames =
        (max_lfr_frames + prefix_query_tokens + sequence_alignment - 1) / sequence_alignment * sequence_alignment -
        prefix_query_tokens;

    ov::Core& core = utils::singleton_core();
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    model->reshape({{"input_features", ov::PartialShape{1, padded_frames, SenseVoiceSmallFeatureExtractor::feature_size}},
                    {"speech_lengths", ov::PartialShape{1}},
                    {"language", ov::PartialShape{1}},
                    {"textnorm", ov::PartialShape{1}}});

    return core.compile_model(model, "NPU", properties);
}

std::unique_ptr<SenseVoiceWindowing> create_static_windowing(size_t max_batch, size_t padded_frames) {
    return std::make_unique<StaticWindowing>(max_batch, padded_frames);
}

}  // namespace ov::genai
