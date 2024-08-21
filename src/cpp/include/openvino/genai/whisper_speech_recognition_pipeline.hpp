// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <optional>
#include <variant>

#include "openvino/core/any.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/whisper_generation_config.hpp"

namespace ov::genai {

using OptionalWhisperGenerationConfig = std::optional<WhisperGenerationConfig>;
using PCMf32AudioDataInput = std::vector<float>;

class OPENVINO_GENAI_EXPORTS WhisperSpeechRecognitionPipeline {
    class Impl;
    std::unique_ptr<Impl> m_impl;

public:
    /**
     * @brief Constructs an LLMPipeline from xml/bin files, tokenizers and configuration in the same dir.
     *
     * @param model_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
     * @param device optional device
     * @param plugin_config optional plugin_config
     */
    WhisperSpeechRecognitionPipeline(const std::string& path,
                                     const std::string& device = "CPU",
                                     const ov::AnyMap& plugin_config = {});

    /**
     * @brief Constructs an LLMPipeline from already existing infer InferRequest and Tokenizer
     *
     * @param request infer request of the model
     * @param tokenizer initialized Tokenizer
     * @param generation_config optional generation_config, be default will be initialized for greedy decoding
     */
    WhisperSpeechRecognitionPipeline(const ov::InferRequest& request,
                                     const ov::genai::Tokenizer& tokenizer,
                                     OptionalWhisperGenerationConfig generation_config = std::nullopt);

    /**
     * @brief Constructs a LLMPipeline when ov::genai::Tokenizer is initialized manually using file from the different
     * dirs.
     *
     * @param model_path Path to the dir with model, tokenizer .xml/.bin files, and generation_configs.json
     * @param tokenizer manually initialized ov::genai::Tokenizer
     * @param device optional device
     * @param plugin_config optional plugin_config
     */
    WhisperSpeechRecognitionPipeline(const std::string& model_path,
                                     const ov::genai::Tokenizer& tokenizer,
                                     const std::string& device = "CPU",
                                     const ov::AnyMap& plugin_config = {});

    ~WhisperSpeechRecognitionPipeline();

    /**
     * @brief High level generate that receives prompts as a string or a vector of strings and returns decoded output.
     *
     * @param inputs input prompt or a vector of prompts
     * @param generation_config optional GenerationConfig
     * @param streamer optional streamer
     * @return DecodedResults decoded resulting text
     */
    DecodedResults generate(PCMf32AudioDataInput inputs,
                            OptionalWhisperGenerationConfig generation_config = std::nullopt,
                            StreamerVariant streamer = std::monostate());

    /**
     * @brief High level generate that receives prompts as a string or a vector of strings and returns decoded output.
     * properties can be in any order pipe.generate(..., ov::genai::max_new_tokens(100),
     * ov::genai::streamer(lambda_func)).
     *
     * @param inputs input prompt or a vector of prompts
     * @param properties properties
     * @return DecodedResults decoded resulting text
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> generate(PCMf32AudioDataInput inputs,
                                                                       Properties&&... properties) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }
    DecodedResults generate(PCMf32AudioDataInput inputs, const ov::AnyMap& config_map);

    DecodedResults operator()(PCMf32AudioDataInput inputs,
                              OptionalWhisperGenerationConfig generation_config = std::nullopt,
                              StreamerVariant streamer = std::monostate()) {
        return generate(inputs, generation_config, streamer);
    }

    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> operator()(PCMf32AudioDataInput inputs,
                                                                         Properties&&... properties) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }

    ov::genai::Tokenizer get_tokenizer();
    WhisperGenerationConfig get_generation_config() const;
    void set_generation_config(const WhisperGenerationConfig& config);
};
}  // namespace ov::genai
