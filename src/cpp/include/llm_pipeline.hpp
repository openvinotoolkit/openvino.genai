// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// #pragma once

#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include "generation_config.hpp"
#include "llm_tokenizer.hpp"
#include "streamer_base.hpp"
#include <filesystem>
#include <optional>

using namespace std;

namespace ov {

using StreamerVariant = std::variant<std::monostate, std::function<void (std::string)>, std::shared_ptr<StreamerBase>>;
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
class LLMPipeline {
public:
    /**
    * @brief Constructs a LLMPipeline when convert model xml/bin files, tokenizers and configuration and in the same dir
    *
    * @param model_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    LLMPipeline(std::string& path, std::string device="CPU", const ov::AnyMap& plugin_config={});
    
    /**
    * @brief Constructs a LLMPipeline when model and tokenizers are in separate dirs
    *
    * @param model_path Path to the dir with model, tokenizer .xml/.bin files, and generation_configs.json
    * @param tokenizer_path path to the tokenizer
    * @param detokenizer_path path to the detokenizer_path
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    LLMPipeline(
        std::string& model_path,
        std::string& tokenizer_path,  // todo: make possible to specify tokenizers with ov::Model, ov::CompiledModel, etc. 
        std::string& detokenizer_path, // todo: do we deen separate detokenizer path?
        std::string device="CPU",
        const ov::AnyMap& plugin_config={}
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

    std::string operator()(std::string text, OptionalGenerationConfig generation_config);
    DecodedResults operator()(std::vector<std::string> text, OptionalGenerationConfig generation_config);
    DecodedResults operator()(std::initializer_list<std::string> text, OptionalGenerationConfig generation_config);

    // generate with streamers
    std::string operator()(std::string text, OptionalGenerationConfig generation_config, StreamerVariant streamer);
    std::string operator()(std::string text, StreamerVariant streamer);
    
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

} // namespace ov
