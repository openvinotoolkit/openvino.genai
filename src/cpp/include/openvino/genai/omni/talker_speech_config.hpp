// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/genai/llm_pipeline.hpp"  // util::EnableIfAllStringAny
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

/**
 * @brief Speech-side generation config for the Qwen3-Omni talker (and future omni
 * models with a similar two-stage thinker + talker architecture).
 *
 * Standalone POD struct — does NOT inherit from GenerationConfig. The thinker text
 * decode is steered by a separate `GenerationConfig text_config` argument to
 * OmniPipeline::generate; this struct only carries fields the talker actually
 * consumes.
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

    void update_generation_config(const ov::AnyMap& properties);

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief Validate talker-only invariants. Cross-config rules (e.g. return_audio
    /// vs beam search on the text_config) are checked by OmniPipelineImpl.
    /// @throws Exception if config is invalid.
    void validate() const;

    /// @brief Enable speech output. Defaults to true; set false to short-circuit
    /// the talker and produce text only.
    bool return_audio = true;

    /// @brief Speaker name. Empty selects the model's default speaker.
    /// Available names are listed under `talker_config.speaker_id` in
    /// `config.json`. Ignored when `speaker_embedding` is non-empty.
    std::string speaker;

    /// @brief Optional explicit talker speaker embedding (`[1, 1, talker_hidden_size]`, f32).
    /// Overrides `speaker` when non-empty. Use `OmniPipeline::get_speaker_embedding`
    /// to fetch precomputed tensors and blend them yourself (e.g. weighted sum) to
    /// mix voices.
    ov::Tensor speaker_embedding;

    /// @brief Number of codec frames accumulated before streaming each audio chunk.
    /// Must be >= 1. Each frame is 80ms of audio at 24 kHz (1920 samples).
    std::size_t audio_chunk_frames = 1;

    /// @brief Cap on talker AR steps. Independent of `text_config.max_new_tokens`
    /// (which caps the thinker text decode). The talker pipeline takes the min of
    /// this value and the model's `talker_config.talker_max_new_tokens`.
    std::size_t max_new_tokens = std::numeric_limits<std::size_t>::max();

    /// @brief RNG seed for deterministic talker + CodePredictor sampling.
    std::size_t rng_seed = 0;

    /// @brief Talker sampling temperature override (must be > 0 when set).
    /// Checkpoint default lives in `generation_config.json -> talker_temperature`.
    std::optional<float> talker_temperature;

    /// @brief Talker top-k override (must be >= 1 when set). Checkpoint default
    /// lives in `generation_config.json -> talker_top_k`.
    std::optional<std::size_t> talker_top_k;

    /// @brief Talker repetition penalty override (must be > 0 when set; 1.0 = no penalty).
    /// Checkpoint default lives in `generation_config.json -> talker_repetition_penalty`.
    std::optional<float> talker_repetition_penalty;

    /// @brief CodePredictor sampling temperature override (must be > 0 when set).
    /// Checkpoint default lives in `generation_config.json -> cp_temperature`.
    std::optional<float> cp_temperature;

    /// @brief CodePredictor top-k override (must be >= 1 when set).
    /// Checkpoint default lives in `generation_config.json -> cp_top_k`.
    std::optional<std::size_t> cp_top_k;

    /// @brief CodePredictor repetition penalty override (must be > 0 when set; 1.0 = no penalty).
    /// Checkpoint default lives in `generation_config.json -> cp_repetition_penalty`.
    std::optional<float> cp_repetition_penalty;
};

}  // namespace genai
}  // namespace ov
