// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper_pipeline_static.hpp"

namespace {
ov::genai::WhisperGenerationConfig from_config_json_if_exists(const std::filesystem::path& model_path) {
    auto config_file_path = model_path / "generation_config.json";
    if (std::filesystem::exists(config_file_path)) {
        return ov::genai::WhisperGenerationConfig((config_file_path).string());
    } else {
        return ov::genai::WhisperGenerationConfig{};
    }
}

ov::genai::OptionalWhisperGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count("generation_config")) {
        return config_map.at("generation_config").as<ov::genai::WhisperGenerationConfig>();
    } else {
        return std::nullopt;
    }
}
}  // namespace

namespace ov {
namespace genai {

StaticWhisperPipeline::StaticWhisperPipeline(const std::filesystem::path& model_path,
                                             const ov::genai::Tokenizer& tokenizer,
                                             const ov::AnyMap& plugin_config)
    : WhisperPipelineImplBase(from_config_json_if_exists(model_path),
                              tokenizer,
                              WhisperFeatureExtractor{(model_path / "preprocessor_config.json").string()}) {
}

StaticWhisperPipeline::StaticWhisperPipeline(const std::filesystem::path& model_path,
                                             const ov::AnyMap& plugin_config)
    : StaticWhisperPipeline(model_path, model_path.string(), plugin_config) {
}

DecodedResults StaticWhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                               OptionalWhisperGenerationConfig generation_config,
                                               StreamerVariant streamer) {
}

}  // namespace genai
}  // namespace ov
