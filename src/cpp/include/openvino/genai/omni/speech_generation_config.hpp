// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "openvino/genai/generation_config.hpp"

namespace ov {
namespace genai {

/**
 * @brief Generation config for Qwen3-Omni speech output.
 *
 * Adds the three Omni-specific knobs (`return_audio`, `speaker`, `audio_chunk_frames`)
 * on top of the inherited GenerationConfig surface. `validate()` enforces the
 * speech-output invariants — beam search and prompt lookup are incompatible with
 * a single talker hidden-state stream, so they are rejected when `return_audio`
 * is true.
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
    std::string speaker;

    /// @brief Number of codec frames accumulated before streaming each audio chunk. Must be >= 1.
    /// Each frame is 80ms of audio at 24 kHz (1920 samples).
    size_t audio_chunk_frames = 1;
};

}  // namespace genai
}  // namespace ov
