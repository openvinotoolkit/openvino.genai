// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "openvino/core/any.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/streamer_base.hpp"

namespace ov {
namespace genai {

using StreamerVariant = std::variant<std::function<bool(std::string)>, std::shared_ptr<StreamerBase>, std::monostate>;
using OptionalGenerationConfig = std::optional<GenerationConfig>;
using EncodedInputs = std::variant<ov::Tensor, std::pair<ov::Tensor, ov::Tensor>, TokenizedInputs>;
using StringInputs = std::variant<std::string, std::vector<std::string>>;

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

     // @brief Convert DecodedResults to a vector of strings.
     // @return A std::vector<std::string> containing the texts from the DecodedResults object.
    operator std::string() const { 
        OPENVINO_ASSERT(texts.size() == 1, "DecodedResults can be converted to string only if contains a single prompt");
        return texts.at(0); 
    }

    // @brief Convert DecodedResults to a single string.
    // @return std::string containing the texts from the DecodedResults object.
    operator std::vector<std::string>() const { 
        return texts; 
    }
    
     // @brief Overloads operator<< to enhance output the contents of DecodedResults.
     // @return A reference to the output stream with the concatenated texts.
    friend std::ostream& operator<<(std::ostream& os, const DecodedResults& dr) {
       for (size_t i = 0; i < dr.texts.size(); ++i) {
            os << dr.texts[i];
            if (i != dr.texts.size() - 1) {
                os << std::endl;
            }
        }
        return os;
    }
};

/**
* @brief This class is used for generation with LLMs.
 */
class OPENVINO_GENAI_EXPORTS LLMPipeline {
public:
    /**
    * @brief Constructs an LLMPipeline from xml/bin files, tokenizers and configuration in the same dir.
    *
    * @param model_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    LLMPipeline(
        const std::string& path, 
        const std::string& device="CPU", 
        const ov::AnyMap& plugin_config={}
    );

    /**
    * @brief Constructs an LLMPipeline from already existing infer InferRequest and Tokenizer
    *
    * @param request infer request of the model
    * @param tokenizer initialized Tokenizer 
    * @param generation_config optional generation_config, be default will be initialized for greedy decoding
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    LLMPipeline(
        const ov::InferRequest& request, 
        const ov::genai::Tokenizer& tokenizer, 
        OptionalGenerationConfig generation_config=std::nullopt
    );
    
    /**
    * @brief Constructs a LLMPipeline when ov::Tokenizer is initialized manually using file from the different dirs.
    *
    * @param model_path Path to the dir with model, tokenizer .xml/.bin files, and generation_configs.json
    * @param tokenizer manually initialized ov::Tokenizer 
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    LLMPipeline(
        const std::string& model_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device="CPU",
        const ov::AnyMap& plugin_config = {}
    );
    
    ~LLMPipeline();

    /**
    * @brief High level generate that receives prompts as a string or a vector of strings and returns decoded output.
    *
    * @param inputs input prompt or a vector of prompts
    * @param generation_config optional GenerationConfig
    * @param streamer optional streamer
    * @return DecodedResults decoded resulting text
    */
    DecodedResults generate(
        StringInputs inputs, 
        OptionalGenerationConfig generation_config=std::nullopt, 
        StreamerVariant streamer=std::monostate()
    );

    /**
    * @brief High level generate that receives prompts as a string or a vector of strings and returns decoded output.
    * properties can be in any order pipe.generate(..., ov::genai::max_new_tokens(100), ov::genai::streamer(lambda_func)).
    * 
    * @param inputs input prompt or a vector of prompts
    * @param properties properties 
    * @return DecodedResults decoded resulting text
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> generate(
            StringInputs inputs,
            Properties&&... properties) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }
    DecodedResults generate(StringInputs inputs, const ov::AnyMap& config_map);


    DecodedResults operator()(
        StringInputs inputs, 
        OptionalGenerationConfig generation_config=std::nullopt, 
        StreamerVariant streamer=std::monostate()
    ) {
        return generate(inputs, generation_config, streamer);
    }

    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> operator()(
            StringInputs inputs,
            Properties&&... properties) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
    * @brief Low level generate to be called with already encoded input_ids tokens.
    * Streamer cannot be used for multibatch inputs.
    *
    * @param input_ids or pair of (input_ids, attentino_mask) encoded input prompt tokens
    * @param generation_config optional GenerationConfig
    * @param streamer optional streamer
    * @return EncodedResults a structure with resulting tokens and scores
    * @throws Exception if the stremaer is set for inputs_ids with multiple batches
    */
    EncodedResults generate(
        const EncodedInputs& inputs, 
        OptionalGenerationConfig generation_config=std::nullopt,
        StreamerVariant streamer=std::monostate()
    );

    /**
    * @brief Low level generate to be called with already encoded input_ids tokens.
    * Streamer cannot be used for multibatch inputs.
    *
    * @param input_ids or pair of (input_ids, attentino_mask) encoded input prompt tokens
    * @param generation config params
    * @return EncodedResults a structure with resulting tokens and scores
    * @throws Exception if the stremaer is set for inputs_ids with multiple batches
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<EncodedResults, Properties...> generate(
            const EncodedInputs& inputs,
            Properties&&... properties) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }
    EncodedResults generate(const EncodedInputs& inputs, const ov::AnyMap& config_map);
  
    ov::genai::Tokenizer get_tokenizer();
    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& generation_config);

    void start_chat();
    void finish_chat();
    std::string apply_chat_template(std::string prompt, std::string role = "user") const;
private:
    class LLMPipelineImpl;
    std::unique_ptr<LLMPipelineImpl> m_pimpl;
};

std::pair<std::string, Any> streamer(StreamerVariant func);
std::pair<std::string, Any> generation_config(const GenerationConfig& config);

}  // namespace genai
}  // namespace ov
