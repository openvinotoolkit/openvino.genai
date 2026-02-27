// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>

#include "json_utils.hpp"
#include "kokoro_tts_model.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "speecht5_tts_model.hpp"
#include "utils.hpp"

namespace {

const std::string get_class_name(const std::filesystem::path& root_dir) {
    const std::filesystem::path model_index_path = root_dir / "config.json";
    std::ifstream file(model_index_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using ov::genai::utils::read_json_param;

    std::string arch = "none";
    if (data.contains("architectures") && data["architectures"].is_array() &&
        data["architectures"].get<std::vector<std::string>>().size() > 0) {
        arch = data["architectures"].get<std::vector<std::string>>()[0];
    }

    return arch;
}

enum class SpeechBackend {
    SpeechT5,
    Kokoro,
};

SpeechBackend resolve_backend(const std::filesystem::path& root_dir,
                             const ov::AnyMap& properties,
                             const ov::genai::SpeechGenerationConfig& config,
                             const std::string& class_name) {
    std::string model_type = config.model_type;
    using ov::genai::utils::read_anymap_param;
    read_anymap_param(properties, "speech_model_type", model_type);

    if (model_type == "speecht5_tts") {
        return SpeechBackend::SpeechT5;
    }
    if (model_type == "kokoro") {
        return SpeechBackend::Kokoro;
    }

    if (class_name == "SpeechT5ForTextToSpeech") {
        return SpeechBackend::SpeechT5;
    }

    if (std::filesystem::exists(root_dir / "openvino_model.xml")) {
        return SpeechBackend::Kokoro;
    }

    OPENVINO_THROW("Unsupported text to speech generation pipeline '", class_name,
                   "'. Set speech_model_type to one of: speecht5_tts, kokoro");
}

}  // namespace

namespace ov {
namespace genai {

Text2SpeechPipeline::Text2SpeechPipeline(const std::filesystem::path& root_dir,
                                         const std::string& device,
                                         const ov::AnyMap& properties)
    : m_speech_gen_config(utils::from_config_json_if_exists<SpeechGenerationConfig>(root_dir)) {
    const std::string class_name = get_class_name(root_dir);

    const auto backend = resolve_backend(root_dir, properties, m_speech_gen_config, class_name);
    if (backend == SpeechBackend::SpeechT5) {
        auto tokenizer = ov::genai::Tokenizer(root_dir);
        m_impl = std::make_shared<SpeechT5TTSImpl>(root_dir, device, properties, tokenizer);
    } else {
        m_impl = std::make_shared<KokoroTTSImpl>(root_dir, device, properties);
    }
}

Text2SpeechDecodedResults Text2SpeechPipeline::generate(const std::vector<std::string>& texts,
                                                        const ov::Tensor& speaker_embedding,
                                                        const ov::AnyMap& properties) {
    SpeechGenerationConfig request_config = m_speech_gen_config;
    request_config.update_generation_config(properties);
    request_config.validate();
    return m_impl->generate(texts, speaker_embedding, request_config);
}

SpeechGenerationConfig Text2SpeechPipeline::get_generation_config() const {
    return m_speech_gen_config;
}

void Text2SpeechPipeline::set_generation_config(const SpeechGenerationConfig& new_config) {
    m_speech_gen_config = new_config;
}

}  // namespace genai
}  // namespace ov
