// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <variant>

#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

/**
 * @brief Order of text and speech in the answer of a model that emits both in one token stream (LFM3-Omni).
 *
 * @note This is a preview API and is subject to change.
 */
enum class SpeechOutputMode {
    /// Text and speech alternate in short blocks, so speech starts early.
    INTERLEAVED,
    /// The answer is speech only, for example text-to-speech.
    SPEECH_ONLY,
};

/**
 * @brief Speech-side generation config of OmniPipeline.
 *
 * Qwen3-Omni speaks with a separate talker after the thinker text decode. LFM3-Omni emits text and
 * speech frames in one stream: its frames are sampled by a depth decoder, which the `cp_*` fields
 * control, and the `talker_*` fields and `speaker` are rejected.
 *
 * Standalone POD struct — does NOT inherit from GenerationConfig. The text decode is steered by a
 * separate `GenerationConfig text_config` argument to OmniPipeline::generate; this struct only
 * carries speech fields.
 *
 * @note This is a preview API and is subject to change.
 */
struct OPENVINO_GENAI_EXPORTS OmniTalkerSpeechConfig {
    OmniTalkerSpeechConfig() = default;

    /**
     * @brief Read defaults from `<models_dir>/config.json`.
     *
     * Currently picks up `talker_config.speaker_id` (single-string case) into
     * `speaker`. Multi-speaker models leave `speaker` empty so the talker resolves
     * a default at generate time. Sampling defaults (`talker_temperature`,
     * `talker_top_k`, etc.) live in the speech pipeline's m_config (loaded from
     * `generation_config.json`) and the std::optional override fields stay unset
     * here so the pipeline keeps using those checkpoint defaults.
     */
    explicit OmniTalkerSpeechConfig(const std::filesystem::path& models_dir);

    /// @brief Enable speech output. Defaults to true; set false to short-circuit
    /// the talker and produce text only.
    bool return_audio = true;

    /// @brief Order of text and speech, for models that emit both in one stream (LFM3-Omni).
    /// Models with a separate talker (Qwen3-Omni) accept only the default.
    SpeechOutputMode output_mode = SpeechOutputMode::INTERLEAVED;

    /// @brief Speaker identity: either a name (std::string) looked up in
    /// `talker_config.speaker_id`, or an explicit embedding tensor
    /// (`[1, 1, talker_hidden_size]`, f32). Default is empty string which
    /// selects the model's default speaker.
    /// Use `pipe.get_talker()->get_speaker_embedding(name)` to fetch precomputed
    /// tensors and blend them yourself (e.g. weighted sum) to mix voices.
    std::variant<std::string, ov::Tensor> speaker;

    /// @brief Number of codec frames accumulated before streaming each audio chunk.
    /// Must be >= 1. At steady state each frame decodes to 1920 samples (80ms at 24 kHz),
    /// but the code2wav vocoder trims its convolutional warmup from the first frame of every
    /// decode call, so a chunk of N frames yields 1920*N - 555 samples, not 1920*N. This is a
    /// property of the vocoder graph, not a miscount. Larger chunks amortize the fixed warmup
    /// cost; very small chunks (e.g. 1) also risk audible seams between independently decoded
    /// chunks in streaming mode.
    ///
    /// Default is 4: the per-frame talker+code_predictor decode cost is fixed (~77ms on GPU),
    /// so a chunk must yield at least that much audio to keep up in real time. A single frame
    /// yields only ~57ms of audio (warmup trim dominates), so `audio_chunk_frames = 1` runs at
    /// ~1.36x real time (falls behind); 4 frames yield ~74ms/frame and reach ~1.04x (keeps up).
    /// Smaller values lower time-to-first-audio at the cost of falling behind real time.
    std::size_t audio_chunk_frames = 4;

    /// @brief Cap on talker AR steps. Independent of `text_config.max_new_tokens`
    /// (which caps the thinker text decode). For LFM3-Omni it caps the speech frames, while
    /// `text_config.max_new_tokens` caps every generated slot, text tokens and speech frames together.
    std::size_t max_new_tokens = std::numeric_limits<std::size_t>::max();

    /// @brief RNG seed for talker + CodePredictor sampling (LFM3-Omni: the depth decoder; sampled text uses
    /// `text_config.rng_seed`).
    std::size_t rng_seed = 0;

    // Qwen3-Omni speech generation has two AR stages:
    // 1. **Talker** — generates semantic tokens (speech content).
    // 2. **CodePredictor (cp)** — converts semantic tokens to codec tokens (audio waveform).
    // Fields below control sampling for each stage independently.
    // Checkpoint defaults loaded from generation_config.json when present.

    /// @brief Talker sampling temperature override (must be > 0 when set).
    /// Checkpoint default from `generation_config.json -> talker_temperature` if present.
    std::optional<float> talker_temperature;

    /// @brief Talker top-k override (must be >= 1 when set).
    /// Checkpoint default from `generation_config.json -> talker_top_k` if present.
    std::optional<std::size_t> talker_top_k;

    /// @brief Talker repetition penalty override (must be > 0 when set; 1.0 = no penalty).
    /// Checkpoint default from `generation_config.json -> talker_repetition_penalty` if present.
    std::optional<float> talker_repetition_penalty;

    // LFM3-Omni samples every codebook of a speech frame with its depth decoder, which the cp_* fields
    // control; its checkpoint defaults are the audio_* keys of generation_config.json.

    /// @brief CodePredictor sampling temperature override (must be > 0 when set).
    /// Checkpoint default from `generation_config.json -> cp_temperature` (LFM3-Omni: `audio_temperature`)
    /// if present.
    std::optional<float> cp_temperature;

    /// @brief CodePredictor top-k override (must be >= 1 when set).
    /// Checkpoint default from `generation_config.json -> cp_top_k` (LFM3-Omni: `audio_top_k`) if present.
    std::optional<std::size_t> cp_top_k;

    /// @brief Depth decoder repetition penalty over the codes each codebook emitted in the last
    /// `audio_repetition_window` frames (must be > 0 when set; 1.0 = no penalty). LFM3-Omni only;
    /// checkpoint default from `generation_config.json -> audio_repetition_penalty` if present.
    std::optional<float> cp_repetition_penalty;
};

}  // namespace genai
}  // namespace ov
