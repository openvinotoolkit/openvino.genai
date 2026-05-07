// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>

#include "openvino/genai/speech_recognition/asr_types.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {

// forward declaration
class ASRPipelineImplBase;

/**
 * @brief Unified Automatic Speech Recognition (ASR) Pipeline.
 * 
 * Supports multiple ASR model architectures:
 * - Whisper (encoder-decoder)
 * - Paraformer (CTC-based)
 * 
 * The model type is auto-detected from config.json in the model directory.
 */
class OPENVINO_GENAI_EXPORTS ASRPipeline {
public:
    enum class ModelType {
        UNKNOWN,
        WHISPER,
        PARAFORMER
    };

    /**
     * @brief Construct ASR pipeline with auto-detection of model type.
     * @param model_dir Path to model directory containing config.json
     * @param device Target device (CPU, GPU, NPU, etc.)
     * @param properties Optional device properties
     */
    explicit ASRPipeline(const std::filesystem::path& model_dir,
                         const std::string& device,
                         const ov::AnyMap& properties = {});

    ~ASRPipeline();

    // ── Core inference (Generic API) ────────────────────────────────────

    /**
     * @brief Generate transcription from audio input using stored config.
     * @param raw_speech_input Audio input (tensor or vector<float>)
     * @param streamer Optional streamer for incremental output
     * @return Decoded transcription results
     */
    ASRDecodedResults generate(
        const RawSpeechInput& raw_speech_input,
        const std::shared_ptr<StreamerBase> streamer = nullptr);

    /**
     * @brief Generate transcription from audio input with explicit config.
     * @param raw_speech_input Audio input (tensor or vector<float>)
     * @param config Generation configuration
     * @param streamer Optional streamer for incremental output
     * @return Decoded transcription results
     */
    ASRDecodedResults generate(
        const RawSpeechInput& raw_speech_input,
        const ASRGenerationConfig& config,
        const std::shared_ptr<StreamerBase> streamer = nullptr);

    /**
     * @brief Generate transcription with config map.
     * @param raw_speech_input Audio input
     * @param config_map Configuration as key-value map
     * @return Decoded transcription results
     */
    ASRDecodedResults generate(
        const RawSpeechInput& raw_speech_input,
        const ov::AnyMap& config_map);

    // ── Legacy Whisper-compatible API (for backward compatibility) ──────

    /**
     * @brief Generate with Whisper-compatible interface.
     * @deprecated Use generate() with ASRGenerationConfig instead
     */
    WhisperDecodedResults generate(
        const RawSpeechInput& raw_speech_input,
        OptionalWhisperGenerationConfig generation_config,
        const std::shared_ptr<StreamerBase> streamer);

    // ── Accessors ───────────────────────────────────────────────────────

    /**
     * @brief Get the tokenizer (if available for this model type).
     * @throws For models without tokenizer support (e.g., some Paraformer variants)
     */
    Tokenizer get_tokenizer();

    ASRGenerationConfig get_generation_config() const;
    void set_generation_config(const ASRGenerationConfig& config);

    // Legacy accessors for backward compatibility
    WhisperGenerationConfig get_whisper_generation_config() const;
    void set_whisper_generation_config(const WhisperGenerationConfig& config);

    // ── Model type detection ────────────────────────────────────────────

    ModelType get_model_type() const;
    bool is_whisper() const;
    bool is_paraformer() const;
    
    /**
     * @brief Get model type string from config.json.
     * @return Model type string (e.g., "whisper", "paraformer")
     */
    std::string get_model_type_string() const;

private:
    std::shared_ptr<ASRPipelineImplBase> m_impl;
    ModelType m_model_type = ModelType::UNKNOWN;
    std::string m_model_type_string;
    ASRGenerationConfig m_generation_config;
};

}  // namespace genai
}  // namespace ov
