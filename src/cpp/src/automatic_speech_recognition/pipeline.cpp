// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "automatic_speech_recognition/pipeline_base.hpp"
#include "automatic_speech_recognition/whisper_asr_pipeline_adapter.hpp"
#include "utils.hpp"

namespace {

enum class ASRModelType { WHISPER };

ASRModelType read_model_type(const std::filesystem::path& models_path) {
    auto config_path = models_path / "config.json";
    std::ifstream stream(config_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", config_path, "'");
    nlohmann::json parsed = nlohmann::json::parse(stream);
    OPENVINO_ASSERT(parsed.contains("model_type"),
                    "config.json must contain 'model_type' field for ASRPipeline model dispatch");
    std::string value = parsed.at("model_type").get<std::string>();

    static const std::unordered_map<std::string, ASRModelType> model_types_map = {
        {"whisper", ASRModelType::WHISPER},
    };

    auto it = model_types_map.find(value);
    if (it != model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' ASR model type");
}

std::optional<ov::genai::ASRGenerationConfig> get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count(ov::genai::utils::CONFIG_ARG_NAME)) {
        return config_map.at(ov::genai::utils::CONFIG_ARG_NAME).as<ov::genai::ASRGenerationConfig>();
    }
    return std::nullopt;
}

}  // namespace

namespace ov::genai {

ASRPipeline::ASRPipeline(const std::filesystem::path& models_path,
                         const std::string& device,
                         const ov::AnyMap& properties) {
    const ASRModelType model_type = read_model_type(models_path);

    if (model_type == ASRModelType::WHISPER) {
        m_impl = std::make_unique<ASRPipeline::WhisperASRPipelineAdapter>(models_path, device, properties);
    } else {
        OPENVINO_THROW("Unsupported ASR model type");
    }
}

ASRDecodedResults ASRPipeline::generate(const RawSpeechInput& raw_speech_input,
                                        std::optional<ASRGenerationConfig> generation_config,
                                        StreamerVariant streamer) {
    const std::shared_ptr<StreamerBase> base_streamer = utils::create_streamer(streamer, m_impl->m_tokenizer);
    return m_impl->generate(raw_speech_input, std::move(generation_config), base_streamer);
}

ASRDecodedResults ASRPipeline::generate(const RawSpeechInput& raw_speech_input, const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    ASRGenerationConfig config = config_arg.has_value() ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    StreamerVariant streamer_variant = utils::get_streamer_from_map(config_map);
    const std::shared_ptr<StreamerBase> base_streamer = utils::create_streamer(streamer_variant, m_impl->m_tokenizer);

    return m_impl->generate(raw_speech_input, config, base_streamer);
}

Tokenizer ASRPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

ASRGenerationConfig ASRPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

void ASRPipeline::set_generation_config(const ASRGenerationConfig& config) {
    m_impl->m_generation_config = config;
    m_impl->m_generation_config.validate();
}

ASRPipeline::~ASRPipeline() = default;

}  // namespace ov::genai
