// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "automatic_speech_recognition/models/fun-asr/pipeline.hpp"
#include "automatic_speech_recognition/models/qwen3-asr/encoder_decoder/pipeline.hpp"
#include "automatic_speech_recognition/models/qwen3-asr/split/pipeline.hpp"
#include "automatic_speech_recognition/models/sensevoice/pipeline.hpp"
#include "automatic_speech_recognition/models/whisper/pipeline.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"
#include "utils.hpp"

namespace {

enum class ASRModelType { whisper, qwen3_asr, fun_asr, sense_voice };

ASRModelType read_model_type(const std::filesystem::path& models_path) {
    auto config_path = models_path / "config.json";
    std::ifstream stream(config_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", config_path, "'");
    nlohmann::json parsed = nlohmann::json::parse(stream);
    OPENVINO_ASSERT(parsed.contains("model_type"),
                    "config.json must contain 'model_type' field for ASRPipeline model dispatch");
    std::string value = parsed.at("model_type").get<std::string>();

    static const std::unordered_map<std::string, ASRModelType> model_types_map = {
        {"whisper", ASRModelType::whisper},
        {"qwen3_asr", ASRModelType::qwen3_asr},
        {"fun_asr", ASRModelType::fun_asr},
        {"sense_voice", ASRModelType::sense_voice},
    };

    auto it = model_types_map.find(value);
    if (it != model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' ASR model type");
}

bool is_qwen3_asr_split_export(const std::filesystem::path& models_path) {
    return std::filesystem::exists(models_path / "openvino_audio_encoder_model.xml");
}

// Only Qwen3-ASR split export implements PagedAttention backend. Other models run SDPA-like stateful decoders.
ov::AnyMap remove_sdpa_attention_backend(const ov::AnyMap& properties) {
    const auto attention_backend_it = properties.find("ATTENTION_BACKEND");
    if (attention_backend_it == properties.end()) {
        return properties;
    }
    // Reject PA before extract_attention_backend() checks PagedAttention platform availability.
    OPENVINO_ASSERT(attention_backend_it->second.as<std::string>() != ov::genai::PA_BACKEND,
                    "'",
                    ov::genai::PA_BACKEND,
                    "' attention backend is supported only for Qwen3-ASR split export");
    return ov::genai::utils::extract_attention_backend(properties).first;
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
    const auto start_time = std::chrono::steady_clock::now();
    const ASRModelType model_type = read_model_type(models_path);

    OPENVINO_ASSERT(!properties.count(ov::genai::forced_aligner.name()) || model_type == ASRModelType::qwen3_asr,
                    "'forced_aligner' is only supported for Qwen3-ASR models.");

    // Split export: audio encoder, text embeddings and language model.
    if (model_type == ASRModelType::qwen3_asr && is_qwen3_asr_split_export(models_path)) {
        m_impl = create_qwen3_asr_split_pipeline(models_path, device, properties);
    } else {
        const ov::AnyMap impl_properties = remove_sdpa_attention_backend(properties);
        switch (model_type) {
        case ASRModelType::whisper: {
            m_impl = std::make_unique<WhisperASRPipelineAdapter>(models_path, device, impl_properties);
            break;
        }
        case ASRModelType::qwen3_asr: {
            m_impl = std::make_unique<Qwen3EncoderDecoderASR>(models_path, device, impl_properties);
            break;
        }
        case ASRModelType::fun_asr: {
            m_impl = std::make_unique<FunASR>(models_path, device, impl_properties);
            break;
        }
        case ASRModelType::sense_voice: {
            m_impl = std::make_unique<SenseVoiceSmall>(models_path, device, impl_properties);
            break;
        }
        }
    }
    const auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ASRDecodedResults ASRPipeline::generate(const AudioInputs& audio_inputs,
                                        const std::optional<ASRGenerationConfig>& generation_config,
                                        StreamerVariant streamer) {
    const std::shared_ptr<StreamerBase> base_streamer = utils::create_streamer(streamer, m_impl->m_tokenizer);
    return std::visit(
        utils::overloaded{
            [&](const std::vector<float>& audio) {
                return m_impl->generate(audio, generation_config, base_streamer);
            },
            [&](const std::vector<std::vector<float>>& audio_batch) {
                OPENVINO_ASSERT(!audio_batch.empty(), "ASRPipeline::generate() requires a non-empty audio batch");
                return m_impl->generate_batch(audio_batch, generation_config, base_streamer);
            },
        },
        audio_inputs);
}

ASRDecodedResults ASRPipeline::generate(const AudioInputs& audio_inputs, const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    ASRGenerationConfig config = config_arg.has_value() ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return generate(audio_inputs, config, utils::get_streamer_from_map(config_map));
}

Tokenizer ASRPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

ASRGenerationConfig ASRPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

void ASRPipeline::set_generation_config(const ASRGenerationConfig& config) {
    m_impl->set_generation_config(config);
}

ASRPipeline::~ASRPipeline() = default;

}  // namespace ov::genai
