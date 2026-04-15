// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>

#include "openvino/genai/speech_recognition/asr_types.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {

/**
 * @brief Abstract base class for ASR pipeline implementations.
 *
 * Both the Whisper and Paraformer back-ends derive from this.  The
 * ASRPipeline front-end holds a shared_ptr<ASRPipelineImplBase> and
 * forwards all calls through it.
 */
class ASRPipelineImplBase {
public:
    ASRPipelineImplBase() = default;

    ASRPipelineImplBase(const std::filesystem::path& model_dir,
                        const std::string& device,
                        const ov::AnyMap& properties)
        : m_model_dir(model_dir), m_device(device), m_properties(properties) {}

    virtual ~ASRPipelineImplBase() = default;

    // ── Core inference (Generic API) ────────────────────────────────────

    virtual ASRDecodedResults generate(
        const RawSpeechInput& raw_speech_input,
        const ASRGenerationConfig& config,
        const std::shared_ptr<StreamerBase> streamer) = 0;

    // ── Accessors (overridden per back-end) ─────────────────────────────

    virtual Tokenizer get_tokenizer() = 0;
    virtual ASRGenerationConfig get_generation_config() const = 0;
    virtual void set_generation_config(const ASRGenerationConfig& config) = 0;
    
    /**
     * @brief Check if this implementation supports Whisper-specific features.
     * @return true for Whisper models, false otherwise
     */
    virtual bool supports_whisper_interface() const { return false; }

    /** @brief Model load time in milliseconds. */
    float m_load_time_ms = 0;

protected:
    std::filesystem::path m_model_dir;
    std::string m_device;
    ov::AnyMap m_properties;
};

}  // namespace genai
}  // namespace ov