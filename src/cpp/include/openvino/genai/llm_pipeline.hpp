// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <openvino/core/any.hpp>
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/streamer_base.hpp"

using namespace std;

namespace ov {

using StreamerVariant = std::variant<std::function<void (std::string)>, std::shared_ptr<StreamerBase>>;
using OptionalGenerationConfig = std::optional<GenerationConfig>;
using OptionalStreamerVariant = std::optional<StreamerVariant>;

/**
* @brief Structure to store resulting batched tokens and scores for each batch sequence
*
* @param tokens sequence of resulting tokens
* @param scores scores for each sequence
*/
class EncodedResults {
public:
    std::vector<std::vector<int64_t>> tokens;
    std::vector<float> scores;
};

/**
* @brief Structure to store resulting batched text outputs and scores for each batch
*
* @param texts vector of resulting sequences
* @param scores scores for each sequence
*/
class DecodedResults {
public:
    std::vector<std::string> texts;
    std::vector<float> scores;
};

/**
* @brief This class is used for generation with LLMs.
 */
class OPENVINO_GENAI_EXPORTS LLMPipeline {
public:
    /**
    * @brief Constructs a LLMPipeline when convert model xml/bin files, tokenizers and configuration and in the same dir.
    *
    * @param model_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    LLMPipeline(std::string& path, std::string device="CPU", const ov::AnyMap& plugin_config={});
    
    /**
    * @brief Constructs a LLMPipeline when ov::Tokenizer is initialized manually using file from the different dirs.
    *
    * @param model_path Path to the dir with model, tokenizer .xml/.bin files, and generation_configs.json
    * @param tokenizer manually initialized ov::Tokenizer 
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    LLMPipeline(
        const std::string model_path,
        const ov::Tokenizer& tokenizer,
        const std::string device="CPU",
        const ov::AnyMap& plugin_config = {}
    );
    
    ~LLMPipeline();

    /**
    * @brief High level generate for the input with a single prompt which encodes inputs and returns decoded output
    *
    * @param text input prompt
    * @param generation_config optional GenerationConfig
    * @param streamer optional streamer
    * @return std::string decoded resulting text
    */
    std::string generate(std::string text, OptionalGenerationConfig generation_config, OptionalStreamerVariant streamer);
    

    template <typename... Properties>
    util::EnableIfAllStringAny<std::string, Properties...> generate(
        std::string text,
        Properties&&... properties) {
        return generate(text, AnyMap{std::forward<Properties>(properties)...});
    }
    std::string generate(std::string text, const ov::AnyMap& config);

    template <typename... Properties>
    util::EnableIfAllStringAny<EncodedResults, Properties...> generate(
        ov::Tensor input_ids,
        Properties&&... properties) {
        return generate(input_ids, AnyMap{std::forward<Properties>(properties)...});
    }
    EncodedResults generate(ov::Tensor input_ids, const ov::AnyMap& config);

    /**
    * @brief High level generate for batched prompts which encodes inputs and returns decoded outputs. 
    * Streamer cannot be used for multibatch inputs.
    *
    * @param text input prompt
    * @param generation_config optional GenerationConfig
    * @return DecodedResults a structure with resulting texts & scores
    */
    DecodedResults generate(std::vector<std::string> texts, OptionalGenerationConfig generation_config);

    /**
    * @brief Low level generate to be called with already encoded input_ids tokens.
    * Streamer cannot be used for multibatch inputs.
    *
    * @param input_ids encoded input prompt tokens
    * @param attention_mask optional attention_mask
    * @param generation_config optional GenerationConfig
    * @param streamer optional streamer
    * @return EncodedResults a structure with resulting tokens and scores
    * @throws Exception if the stremaer is set for inputs_ids with multiple batches
    */
    EncodedResults generate(ov::Tensor input_ids, 
                            std::optional<ov::Tensor> attention_mask, 
                            OptionalGenerationConfig generation_config,
                            OptionalStreamerVariant streamer);
    
    template <typename InputsType, typename... Properties>
    util::EnableIfAllStringAny<std::string, Properties...> operator()(
        InputsType text,
        Properties&&... properties) {
        return generate(text, AnyMap{std::forward<Properties>(properties)...});
    }
    std::string operator()(std::string text, OptionalGenerationConfig generation_config={});
    
    DecodedResults operator()(std::vector<std::string> text, OptionalGenerationConfig generation_config);
    DecodedResults operator()(std::initializer_list<std::string> text, OptionalGenerationConfig generation_config);

    // generate with streamers
    std::string operator()(std::string text, OptionalGenerationConfig generation_config, OptionalStreamerVariant streamer);
    std::string operator()(std::string text, OptionalStreamerVariant streamer);
    
    ov::Tokenizer get_tokenizer();
    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& generation_config);

    void start_chat();
    void finish_chat();
    void reset_state();
    std::string apply_chat_template(std::string prompt, std::string role = "user") const;
private:
    class LLMPipelineImpl;
    std::unique_ptr<LLMPipelineImpl> m_pimpl;
};

/*
 * utils that allow to use generate and operarator() in the folllowing way:
 * pipe.generate(input_ids, ov::max_new_tokens(200), ov::temperature(1.0f),...)
 * pipe(text, ov::max_new_tokens(200), ov::temperature(1.0f),...)
 * All names match to names in cofnig except streamer.
*/
static constexpr ov::Property<size_t> max_new_tokens{"max_new_tokens"};
static constexpr ov::Property<float> temperature{"temperature"};

// It's problematic to store and automaticall convert std::variant in AnyMap
static constexpr ov::Property<std::function<void (std::string)>> streamer_lambda{"streamer_lambda"};
static constexpr ov::Property<std::shared_ptr<StreamerBase>> streamer{"streamer"};

} // namespace ov
