// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>

#include "json_utils.hpp"
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

}  // namespace

namespace ov {
namespace genai {

Text2SpeechPipeline::Text2SpeechPipeline(const std::filesystem::path& root_dir,
                                         const std::string& device,
                                         const ov::AnyMap& properties)
    : m_speech_gen_config(utils::from_config_json_if_exists<SpeechGenerationConfig>(root_dir)) {
    const std::string class_name = get_class_name(root_dir);

    auto start_time = std::chrono::steady_clock::now();

    auto tokenizer = ov::genai::Tokenizer(root_dir);
    if (class_name == "SpeechT5ForTextToSpeech") {
        m_impl = std::make_shared<SpeechT5TTSImpl>(root_dir, device, properties, tokenizer);
    } else {
        OPENVINO_THROW("Unsupported text to speech generation pipeline '", class_name, "'");
    }
}

Text2SpeechDecodedResults Text2SpeechPipeline::generate(const std::vector<std::string>& texts,
                                                        const ov::Tensor& speaker_embedding,
                                                        const ov::AnyMap& properties) {
    return m_impl->generate(texts, speaker_embedding, m_speech_gen_config);
}

SpeechGenerationConfig Text2SpeechPipeline::get_generation_config() const {
    return m_speech_gen_config;
}

void Text2SpeechPipeline::set_generation_config(const SpeechGenerationConfig& new_config) {
    m_speech_gen_config = new_config;
}

}  // namespace genai
}  // namespace ov
