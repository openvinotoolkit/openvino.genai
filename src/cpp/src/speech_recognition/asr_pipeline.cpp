// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_recognition/asr_pipeline.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "speech_recognition/paraformer_impl.hpp"
#include "speech_recognition/whisper_pipeline_wrapper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

namespace {

/**
 * @brief Detect model type from config.json model_type field.
 * Falls back to file-based detection if config.json is not available.
 */
std::pair<ASRPipeline::ModelType, std::string> detect_model_type(const std::filesystem::path& model_dir) {
    // Try to read model_type from config.json
    auto config_path = model_dir / "config.json";
    if (std::filesystem::exists(config_path)) {
        try {
            std::ifstream config_file(config_path);
            nlohmann::json config = nlohmann::json::parse(config_file);
            
            if (config.contains("model_type")) {
                std::string model_type = config["model_type"].get<std::string>();
                
                // Map model_type string to enum
                if (model_type == "whisper") {
                    return {ASRPipeline::ModelType::WHISPER, model_type};
                } else if (model_type == "paraformer" || model_type == "funasr" || 
                           model_type == "Paraformer") {
                    return {ASRPipeline::ModelType::PARAFORMER, model_type};
                }
                // Unknown model type in config.json - try file-based detection
            }
        } catch (const std::exception&) {
            // JSON parsing failed - fall back to file-based detection
        }
    }
    
    // Fallback: file-based detection for backward compatibility
    bool has_whisper = std::filesystem::exists(model_dir / "openvino_encoder_model.xml")
                    && std::filesystem::exists(model_dir / "openvino_decoder_model.xml");

    // Paraformer only requires openvino_model.xml (tokens.json is optional)
    bool has_paraformer = std::filesystem::exists(model_dir / "openvino_model.xml");

    if (has_whisper) {
        return {ASRPipeline::ModelType::WHISPER, "whisper"};
    }
    if (has_paraformer) {
        return {ASRPipeline::ModelType::PARAFORMER, "paraformer"};
    }
    return {ASRPipeline::ModelType::UNKNOWN, ""};
}

}  // namespace

ASRPipeline::ASRPipeline(const std::filesystem::path& model_dir,
                         const std::string& device,
                         const ov::AnyMap& properties) {
    OPENVINO_ASSERT(std::filesystem::exists(model_dir),
                    "ASRPipeline: model directory does not exist: ", model_dir);

    auto [type, type_string] = detect_model_type(model_dir);
    m_model_type = type;
    m_model_type_string = type_string;

    switch (m_model_type) {
        case ModelType::WHISPER:
            // Use WhisperPipeline directly via wrapper
            m_impl = std::make_shared<WhisperPipelineWrapper>(model_dir, device, properties);
            break;
        case ModelType::PARAFORMER:
            m_impl = std::make_shared<ParaformerImpl>(model_dir, device, properties);
            break;
        default:
            OPENVINO_THROW("ASRPipeline: unable to detect model type. "
                          "Expected 'whisper' or 'paraformer' in config.json model_type field, "
                          "or appropriate model files in: ", model_dir);
    }

    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: failed to create implementation");
    
    // Initialize stored config from impl
    m_generation_config = m_impl->get_generation_config();
}

ASRPipeline::~ASRPipeline() = default;

// ── Core inference (Generic API) ────────────────────────────────────────

ASRDecodedResults ASRPipeline::generate(
    const RawSpeechInput& raw_speech_input,
    const std::shared_ptr<StreamerBase> streamer) {
    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: implementation not initialized");
    return m_impl->generate(raw_speech_input, m_generation_config, streamer);
}

ASRDecodedResults ASRPipeline::generate(
    const RawSpeechInput& raw_speech_input,
    const ASRGenerationConfig& config,
    const std::shared_ptr<StreamerBase> streamer) {
    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: implementation not initialized");
    return m_impl->generate(raw_speech_input, config, streamer);
}

ASRDecodedResults ASRPipeline::generate(
    const RawSpeechInput& raw_speech_input,
    const ov::AnyMap& config_map) {
    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: implementation not initialized");
    ASRGenerationConfig config = m_impl->get_generation_config();
    config.update(config_map);
    return m_impl->generate(raw_speech_input, config, nullptr);
}

// ── Legacy Whisper-compatible API ───────────────────────────────────────

WhisperDecodedResults ASRPipeline::generate(
    const RawSpeechInput& raw_speech_input,
    OptionalWhisperGenerationConfig generation_config,
    const std::shared_ptr<StreamerBase> streamer) {
    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: implementation not initialized");
    return m_impl->generate_whisper(raw_speech_input, generation_config, streamer);
}

// ── Accessors ───────────────────────────────────────────────────────────

Tokenizer ASRPipeline::get_tokenizer() {
    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: implementation not initialized");
    return m_impl->get_tokenizer();
}

ASRGenerationConfig ASRPipeline::get_generation_config() const {
    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: implementation not initialized");
    // Return stored config (kept in sync with impl)
    return m_generation_config;
}

void ASRPipeline::set_generation_config(const ASRGenerationConfig& config) {
    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: implementation not initialized");
    m_generation_config = config;
    m_impl->set_generation_config(config);
}

WhisperGenerationConfig ASRPipeline::get_whisper_generation_config() const {
    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: implementation not initialized");
    return m_impl->get_whisper_generation_config();
}

void ASRPipeline::set_whisper_generation_config(const WhisperGenerationConfig& config) {
    OPENVINO_ASSERT(m_impl != nullptr, "ASRPipeline: implementation not initialized");
    m_impl->set_whisper_generation_config(config);
}

// ── Model type detection ────────────────────────────────────────────────

ASRPipeline::ModelType ASRPipeline::get_model_type() const {
    return m_model_type;
}

bool ASRPipeline::is_whisper() const {
    return m_model_type == ModelType::WHISPER;
}

bool ASRPipeline::is_paraformer() const {
    return m_model_type == ModelType::PARAFORMER;
}

std::string ASRPipeline::get_model_type_string() const {
    return m_model_type_string;
}

// ── ASRGenerationConfig implementation ──────────────────────────────────

void ASRGenerationConfig::update(const ov::AnyMap& config_map) {
    using ov::genai::utils::read_anymap_param;
    
    read_anymap_param(config_map, "max_new_tokens", max_new_tokens);
    read_anymap_param(config_map, "language", language);
    read_anymap_param(config_map, "task", task);
    read_anymap_param(config_map, "return_timestamps", return_timestamps);
    read_anymap_param(config_map, "temperature", temperature);
    read_anymap_param(config_map, "top_k", top_k);
    read_anymap_param(config_map, "top_p", top_p);
    read_anymap_param(config_map, "num_beams", num_beams);
    read_anymap_param(config_map, "suppress_blank", suppress_blank);
}

}  // namespace genai
}  // namespace ov