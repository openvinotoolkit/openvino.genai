// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper/feature_extractor.hpp"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <openvino/core/except.hpp>
#include <thread>

#include "json_utils.hpp"
#include "openvino/genai/visibility.hpp"
#include "whisper/audio_utils.hpp"

namespace ov {
namespace genai {

std::vector<float> WhisperFeatures::get_data_with_offset(const size_t frame_offset, const size_t min_frames) {
    OPENVINO_ASSERT(n_frames > frame_offset);

    size_t copy_size = std::min(n_frames - frame_offset, min_frames);
    std::vector<float> offset_data;

    for (size_t i = 0; i < feature_size; i++) {
        size_t offset = frame_offset + (i * n_frames);
        std::copy(data.begin() + offset, data.begin() + offset + copy_size, std::back_inserter(offset_data));
        if (copy_size < min_frames) {
            std::fill_n(std::back_inserter(offset_data), min_frames - copy_size, 0);
        }
    }

    return offset_data;
}

WhisperFeatureExtractor::WhisperFeatureExtractor(const std::filesystem::path& preprocessor_json_path) {
    init_parameters(preprocessor_json_path);
    // HF WhisperFeatureExtractor pre-pads to n_samples = sampling_rate * chunk_length
    // (default 30 s = 480 000 samples at 16 kHz). Re-derive after JSON load so an
    // overridden chunk_length wins over a stale default.
    n_samples = sampling_rate * chunk_length;
    rebuild_tables();
}

WhisperFeatureExtractor::WhisperFeatureExtractor(size_t feature_size,
                                                 size_t sampling_rate,
                                                 size_t n_fft,
                                                 size_t hop_length)
    : feature_size(feature_size),
      sampling_rate(sampling_rate),
      hop_length(hop_length),
      n_fft(n_fft),
      // chunk_length is unused when n_samples == 0; keep it nonzero so any
      // accidental "chunk_length / X" division stays safe.
      chunk_length(0),
      n_samples(0),
      nb_max_frames(0) {
    rebuild_tables();
}

void WhisperFeatureExtractor::init_parameters(const std::filesystem::path& preprocessor_json_path) {
    // preprocessor_config.json not found. Skip parameters initialization from file, use defaults.
    if (!std::filesystem::exists(preprocessor_json_path)) {
        return;
    }

    using ov::genai::utils::read_json_param;

    std::ifstream f(preprocessor_json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", preprocessor_json_path, "' with preprocessor config");

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "feature_size", feature_size);
    read_json_param(data, "sampling_rate", sampling_rate);
    read_json_param(data, "hop_length", hop_length);
    read_json_param(data, "n_fft", n_fft);
    read_json_param(data, "chunk_length", chunk_length);
    read_json_param(data, "n_samples", n_samples);
    read_json_param(data, "nb_max_frames", nb_max_frames);
}

void WhisperFeatureExtractor::rebuild_tables() {
    m_sin_vals = audio_utils::build_sin_table(n_fft);
    m_cos_vals = audio_utils::build_cos_table(n_fft);
    m_mel_filter = audio_utils::build_mel_filter(1 + n_fft / 2, feature_size, sampling_rate);
}

std::vector<float> WhisperFeatureExtractor::pad_with_reflect(const std::vector<float>& raw_speech) const {
    const size_t reflect_pad_size = n_fft / 2;
    // Reflect padding copies raw tail samples symmetrically; if raw is shorter than the
    // reflect window, the copy would read into zero-padded slots and bake silence into
    // the reflected head/tail. Reject that configuration explicitly instead of producing
    // subtly wrong features.
    OPENVINO_ASSERT(raw_speech.size() > reflect_pad_size,
                    "raw_speech too short for reflect padding: size=",
                    raw_speech.size(),
                    " required > ",
                    reflect_pad_size);

    const size_t total_pad_length = std::max(raw_speech.size(), n_samples) + 2 * reflect_pad_size;
    std::vector<float> padded(total_pad_length, 0.0f);

    std::copy(raw_speech.begin(), raw_speech.end(), padded.begin() + reflect_pad_size);

    std::reverse_copy(padded.begin() + reflect_pad_size + 1,
                      padded.begin() + reflect_pad_size + 1 + reflect_pad_size,
                      padded.begin());

    std::reverse_copy(padded.end() - reflect_pad_size - 1 - reflect_pad_size,
                      padded.end() - reflect_pad_size - 1,
                      padded.end() - reflect_pad_size);

    return padded;
}

std::vector<float> WhisperFeatureExtractor::compute_mel(const std::vector<float>& padded,
                                                        size_t n_samples_real,
                                                        size_t n_frames) const {
    std::vector<float> output(feature_size * n_frames, 0.0f);

    const size_t n_threads =
        std::max(size_t{1}, std::min(size_t{4}, static_cast<size_t>(std::thread::hardware_concurrency())));

    const auto hann = audio_utils::hann_window(n_fft);

    // n_samples_real is the logical sample count the worker treats as "real" data: raw +
    // front reflect pad only. Everything past that (the tail reflect and any zero-fill
    // between raw and n_samples) is zeroed inside the FFT frame. Matches the pre-refactor
    // Whisper form bit-for-bit — without this, tail-of-signal frames would pick up the
    // reflected tail samples instead of log10(1e-10) silence.
    std::vector<std::thread> workers(n_threads - 1);
    for (size_t iw = 0; iw < n_threads - 1; ++iw) {
        workers[iw] = std::thread(audio_utils::mel_worker,
                                  static_cast<int>(iw + 1),
                                  std::cref(hann),
                                  std::cref(padded),
                                  static_cast<int>(n_samples_real),
                                  static_cast<int>(n_fft),
                                  static_cast<int>(hop_length),
                                  static_cast<int>(n_threads),
                                  std::cref(m_mel_filter),
                                  feature_size,
                                  n_frames,
                                  std::ref(output),
                                  std::cref(m_sin_vals),
                                  std::cref(m_cos_vals));
    }

    audio_utils::mel_worker(0,
                            hann,
                            padded,
                            static_cast<int>(n_samples_real),
                            static_cast<int>(n_fft),
                            static_cast<int>(hop_length),
                            static_cast<int>(n_threads),
                            m_mel_filter,
                            feature_size,
                            n_frames,
                            output,
                            m_sin_vals,
                            m_cos_vals);

    for (auto& w : workers) {
        w.join();
    }

    // Whisper-style clamping and normalization: clamp below (max - 8), then (x + 4) / 4.
    float mmax = -1e20f;
    for (const auto val : output) {
        mmax = std::max(mmax, val);
    }
    mmax -= 8.0f;
    for (auto& val : output) {
        val = std::max(val, mmax);
        val = (val + 4.0f) / 4.0f;
    }

    return output;
}

WhisperFeatures WhisperFeatureExtractor::extract(const std::vector<float>& raw_speech) {
    OPENVINO_ASSERT(!raw_speech.empty(), "Cannot extract mel spectrogram from empty audio input");

    WhisperFeatures features;
    features.feature_size = feature_size;
    features.n_active_frames = raw_speech.size() / hop_length;

    const size_t reflect_pad_size = n_fft / 2;
    auto padded = pad_with_reflect(raw_speech);
    features.n_frames = (padded.size() - n_fft) / hop_length;
    if (features.n_frames == 0) {
        features.data = {};
        return features;
    }
    // The "real" sample count excludes the tail reflect pad so frames past the real signal
    // zero-fill with log10(1e-10) silence via mel_worker's short-circuit (matches the HF
    // WhisperFeatureExtractor reference for both default and n_samples == 0 configs).
    features.data = compute_mel(padded, raw_speech.size() + reflect_pad_size, features.n_frames);
    return features;
}

}  // namespace genai
}  // namespace ov
