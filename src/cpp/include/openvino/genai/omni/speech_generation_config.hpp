// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>

#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

/**
 * @brief Generation config for Qwen3-Omni speech output.
 *
 */
class OPENVINO_GENAI_EXPORTS OmniSpeechGenerationConfig : public GenerationConfig {
public:
    OmniSpeechGenerationConfig() = default;

    /**
     * @brief Load both inherited GenerationConfig fields (from `generation_config.json`)
     * and Omni-specific defaults (from `config.json`'s `talker_config` subtree).
     *
     * Missing keys silently fall back to the field defaults; missing `config.json`
     * means the directory is not an Omni model and Omni-specific defaults stay as-is.
     */
    explicit OmniSpeechGenerationConfig(const std::filesystem::path& models_dir);

    void update_generation_config(const ov::AnyMap& properties);

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief Validate inherited fields plus Omni-specific invariants.
    /// @throws Exception if config is invalid.
    void validate() const;

    /// @brief Enable speech output generation. Defaults to true; set false to short-circuit
    /// the talker and produce text only.
    bool return_audio = true;

    /// @brief Speaker name for speech output. Empty selects the model's default speaker.
    /// Available names are listed under `talker_config.speaker_id` in the model's `config.json`.
    /// Ignored when `speaker_embedding` is non-empty.
    std::string speaker;

    /// @brief Optional explicit talker speaker embedding (`[1, 1, talker_hidden_size]`, f32).
    /// Overrides `speaker` when non-empty. Use `OmniPipeline::get_speaker_embedding()` to fetch
    /// the precomputed tensors for named speakers and blend them yourself (e.g. weighted sum)
    /// to mix voices.
    ov::Tensor speaker_embedding;

    /// @brief Number of codec frames accumulated before streaming each audio chunk. Must be >= 1.
    /// Each frame is 80ms of audio at 24 kHz (1920 samples).
    size_t audio_chunk_frames = 1;

    /// @brief Talker sampling temperature override (must be > 0 when set). Higher = more
    /// variation in voice timing/prosody. Checkpoint default is in `generation_config.json`
    /// under `talker_temperature` (typically ~0.9).
    std::optional<float> talker_temperature;

    /// @brief Talker top-k override (must be >= 1 when set). Checkpoint default is in
    /// `generation_config.json` under `talker_top_k` (typically 50).
    std::optional<size_t> talker_top_k;

    /// @brief Talker repetition penalty override (must be > 0 when set; 1.0 = no penalty).
    /// Checkpoint default is in `generation_config.json` under `talker_repetition_penalty`
    /// (typically 1.0).
    std::optional<float> talker_repetition_penalty;

    /// @brief CodePredictor sampling temperature override (must be > 0 when set).
    /// Checkpoint default is in `generation_config.json` under `cp_temperature` (typically 1.0).
    std::optional<float> cp_temperature;

    /// @brief CodePredictor top-k override (must be >= 1 when set). Checkpoint default is
    /// in `generation_config.json` under `cp_top_k` (typically 50).
    std::optional<size_t> cp_top_k;

    /// @brief CodePredictor repetition penalty override (must be > 0 when set; 1.0 = no penalty).
    /// Checkpoint default is in `generation_config.json` under `cp_repetition_penalty`
    /// (typically 1.0).
    std::optional<float> cp_repetition_penalty;
};

}  // namespace genai
}  // namespace ov
