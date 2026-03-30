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
                             const std::string& class_name) {
    if (class_name == "SpeechT5ForTextToSpeech") {
        return SpeechBackend::SpeechT5;
    }

    const bool has_openvino_model = std::filesystem::exists(root_dir / "openvino_model.xml");
    const bool has_voices_dir = std::filesystem::exists(root_dir / "voices") &&
                                std::filesystem::is_directory(root_dir / "voices");
    const bool architecture_mentions_kokoro = class_name.find("Kokoro") != std::string::npos;

    bool has_kokoro_vocab = false;
    const std::filesystem::path config_json_path = root_dir / "config.json";
    if (std::filesystem::exists(config_json_path)) {
        std::ifstream config_file(config_json_path);
        if (config_file.is_open()) {
            const nlohmann::json config = nlohmann::json::parse(config_file, nullptr, false);
            has_kokoro_vocab = !config.is_discarded() && config.contains("vocab") && config["vocab"].is_object();
        }
    }

    if (has_openvino_model && (architecture_mentions_kokoro || (has_voices_dir && has_kokoro_vocab))) {
        return SpeechBackend::Kokoro;
    }

    OPENVINO_THROW("Unsupported text to speech generation pipeline '", class_name,
                   "'. Unable to auto-detect a supported backend from model metadata/files. "
                   "Expected SpeechT5 architecture metadata or Kokoro-specific model layout");
}

}  // namespace

namespace ov {
namespace genai {

Text2SpeechPipeline::Text2SpeechPipeline(const std::filesystem::path& root_dir,
                                         const std::string& device,
                                         const ov::AnyMap& properties)
    : m_speech_gen_config(utils::from_config_json_if_exists<SpeechGenerationConfig>(root_dir)) {
    const std::string class_name = get_class_name(root_dir);

    const auto backend = resolve_backend(root_dir, class_name);
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

Text2SpeechDecodedResults Text2SpeechPipeline::generate_from_phonemes(const std::vector<std::string>& phoneme_chunks,
                                                                      const ov::Tensor& speaker_embedding,
                                                                      const ov::AnyMap& properties) {
    return generate_from_phonemes(std::vector<std::vector<std::string>>{phoneme_chunks}, speaker_embedding, properties);
}

Text2SpeechDecodedResults Text2SpeechPipeline::generate_from_phonemes(
    const std::vector<std::vector<std::string>>& phoneme_chunks,
    const ov::Tensor& speaker_embedding,
    const ov::AnyMap& properties) {
    SpeechGenerationConfig request_config = m_speech_gen_config;
    request_config.update_generation_config(properties);
    request_config.validate();
    return m_impl->generate_from_phonemes(phoneme_chunks, speaker_embedding, request_config);
}

Text2SpeechDecodedResults Text2SpeechPipeline::generate_from_tokens(const std::vector<SpeechToken>& tokens,
                                                                    const ov::Tensor& speaker_embedding,
                                                                    const ov::AnyMap& properties) {
    return generate_from_tokens(std::vector<std::vector<SpeechToken>>{tokens}, speaker_embedding, properties);
}

Text2SpeechDecodedResults Text2SpeechPipeline::generate_from_tokens(
    const std::vector<std::vector<SpeechToken>>& token_batches,
    const ov::Tensor& speaker_embedding,
    const ov::AnyMap& properties) {
    SpeechGenerationConfig request_config = m_speech_gen_config;
    request_config.update_generation_config(properties);
    request_config.validate();
    return m_impl->generate_from_tokens(token_batches, speaker_embedding, request_config);
}

std::vector<std::string> Text2SpeechPipeline::phonemize(const std::string& text,
                                                        const ov::AnyMap& properties) {
    auto chunks = phonemize(std::vector<std::string>{text}, properties);
    OPENVINO_ASSERT(chunks.size() == 1, "Expected one phonemized item");
    return chunks[0];
}

std::vector<std::vector<std::string>> Text2SpeechPipeline::phonemize(const std::vector<std::string>& texts,
                                                                     const ov::AnyMap& properties) {
    SpeechGenerationConfig request_config = m_speech_gen_config;
    request_config.update_generation_config(properties);
    request_config.validate();
    return m_impl->phonemize(texts, request_config);
}

SpeechGenerationConfig Text2SpeechPipeline::get_generation_config() const {
    return m_speech_gen_config;
}

void Text2SpeechPipeline::set_generation_config(const SpeechGenerationConfig& new_config) {
    m_speech_gen_config = new_config;
}

ov::Shape Text2SpeechPipeline::get_speaker_embedding_shape() const {
    return m_impl->get_speaker_embedding_shape();
}

}  // namespace genai
}  // namespace ov
