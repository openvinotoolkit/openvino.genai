// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

/**
 * @brief Speech interval on the input audio timeline in seconds, [start_ts, end_ts).
 */
struct SpeechTimestamp {
    double start_ts = 0.0;
    double end_ts = 0.0;
};

/**
 * @brief Segmentation options for the Silero voice activity detector.
 *
 * Segmentation parameters follow upstream Silero VAD v6.3 get_speech_timestamps() for 16 kHz audio and
 * 512-sample windows.
 */
struct SileroVADConfig {
    /**
     * @brief Speech probability at or above which a window starts speech.
     *
     * Must be in (0, 1). Default: 0.5.
     */
    double threshold = 0.5;

    /**
     * @brief Speech probability below which a window counts as silence.
     *
     * When unset, max(threshold - 0.15, 0.01) is used. When set, must be in (0, threshold].
     */
    std::optional<double> neg_threshold = std::nullopt;

    /**
     * @brief Minimum speech segment duration in milliseconds.
     *
     * Segments not longer than this are discarded before padding is applied. Default: 250.
     */
    size_t min_speech_duration_ms = 250;

    /**
     * @brief Minimum silence duration in milliseconds that ends a speech segment.
     *
     * A speech segment ends only after silence persists for this duration. Default: 100.
     */
    size_t min_silence_duration_ms = 100;

    /**
     * @brief Padding in milliseconds added to both sides of each speech segment.
     *
     * When neighbouring segments are closer than twice this value, the gap between them is split evenly.
     * Padding is clamped to the audio bounds. Default: 30.
     */
    size_t speech_pad_ms = 30;

    /**
     * @brief Speech duration limit in seconds used to split long speech regions.
     *
     * Because splitting is aligned to windows and padding is applied afterwards, returned segments may be
     * longer than this value. Must be positive; infinity disables splitting. Default: infinity.
     */
    double max_speech_duration_s = std::numeric_limits<double>::infinity();

    /**
     * @brief Minimum silence duration in milliseconds for a silence to be a candidate split point when
     * speech exceeds max_speech_duration_s.
     *
     * Default: 98.
     */
    size_t min_silence_at_max_speech_ms = 98;
};

/**
 * @brief Silero segmentation options for VoiceActivityDetector::get_speech_timestamps().
 *
 * Call-time property carrying a SileroVADConfig. Defaults are used when it is not set.
 */
static constexpr ov::Property<SileroVADConfig> silero_vad_config{"silero_vad_config"};

/**
 * @brief Standalone voice activity detector for mono 16 kHz audio.
 *
 * Currently supports Silero VAD v6.3: the model directory must contain silero_vad_16k_sequence.onnx.
 * Recurrent state is reset on every get_speech_timestamps() call.
 *
 * @note Not thread-safe. An instance may be reused for sequential get_speech_timestamps() calls but must not
 * be called concurrently; use a separate instance per thread.
 */
class OPENVINO_GENAI_EXPORTS VoiceActivityDetector {
public:
    /**
     * @brief Constructs the voice activity detector from a model directory.
     * @param models_path Directory containing the detector model.
     * @param device Device to run inference on (e.g. "CPU").
     * @param properties OpenVINO compile properties.
     */
    VoiceActivityDetector(const std::filesystem::path& models_path,
                          const std::string& device,
                          const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VoiceActivityDetector(const std::filesystem::path& models_path,
                          const std::string& device,
                          Properties&&... properties)
        : VoiceActivityDetector(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~VoiceActivityDetector();

    /**
     * @brief Detects speech in mono 16 kHz float PCM.
     * @param audio Mono 16 kHz float PCM samples.
     * @param properties Optional backend-specific segmentation properties: ov::genai::silero_vad_config for
     * Silero. Any other property is rejected.
     * @return Sorted, non-overlapping speech intervals in seconds, not rounded; empty when no speech is detected.
     * @throws ov::Exception if a property is unsupported or a configuration value is out of range.
     */
    std::vector<SpeechTimestamp> get_speech_timestamps(const std::vector<float>& audio,
                                                       const ov::AnyMap& properties = {});

    template <typename... Properties>
    util::EnableIfAllStringAny<std::vector<SpeechTimestamp>, Properties...> get_speech_timestamps(
        const std::vector<float>& audio,
        Properties&&... properties) {
        return get_speech_timestamps(audio, ov::AnyMap{std::forward<Properties>(properties)...});
    }

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
