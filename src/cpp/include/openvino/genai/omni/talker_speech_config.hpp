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
    /// Must be >= 1. For Qwen3-Omni, each frame decodes to S samples, the product of the checkpoint's code2wav
    /// `upsample_rates` and `upsampling_ratios`. The code2wav vocoder trims its convolutional
    /// warmup of W samples once per utterance, so the first chunk of N frames yields S*N - W
    /// samples and every later chunk yields S*N. Streamed audio has the same length as
    /// non-streamed audio. This holds while `audio_chunk_left_context_frames` > 0. For the
    /// standard Qwen3-Omni, S = 1920 (80ms at 24 kHz) and W = 555, so the default first chunk is
    /// about 297ms and later chunks are 320ms.
    ///
    /// Every chunk after the first is decoded together with up to `audio_chunk_left_context_frames`
    /// earlier frames, so the vocoder sees their history. That context is decoded again for each
    /// chunk, so smaller chunks lower time-to-first-audio but spend more vocoder time per second of
    /// audio and can fall behind real time.
    ///
    /// For LFM3-Omni, a frame is `stft_frame_stack * stft_hop_length` samples of the codec config (1920 for
    /// LFM3-Omni-2B). The ISTFT overlap at the end of each chunk is held back until the next chunk, and the
    /// last chunk flushes it, so all chunks together give exactly one frame of samples per frame.
    std::size_t audio_chunk_frames = 4;

    /// @brief Number of earlier codec frames the code2wav vocoder decodes together with each
    /// streamed chunk. Their samples are dropped from the output; they only give the chunk's first
    /// samples the history a non-streamed decode would see. The first chunk has no history, so the
    /// value is capped by the number of frames already streamed. 0 decodes every chunk on its own:
    /// each chunk then loses the code2wav warmup samples, so streamed audio is shorter than
    /// non-streamed audio.
    ///
    /// Larger values give smoother seams between chunks but cost more vocoder work per chunk.
    /// Default is 25, the `left_context_size` of transformers `Qwen3OmniMoeCode2Wav.chunked_decode`.
    /// Ignored when no speech streamer is set.
    std::size_t audio_chunk_left_context_frames = 25;

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
