// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <initializer_list>
#include <filesystem>

#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"
#include <openvino/runtime/properties.hpp>

namespace ov {
namespace genai {

using ChatHistory = std::vector<std::unordered_map<std::string, std::string>>;

struct TokenizedInputs {
    ov::Tensor input_ids;
    ov::Tensor attention_mask;
};

/**
* @brief class is used to encode prompts and decode resulting tokens
*/
class OPENVINO_GENAI_EXPORTS Tokenizer {
public:
    /**
    * @brief ov::genai::Tokenizer constructor.
    * @param tokenizer_path openvino_tokenizer.xml and openvino_detokenizer.xml should be located in the tokenizer_path
    * @param properties Properties passed to ov::Core::compile_model
    */
    Tokenizer(const std::filesystem::path& tokenizer_path, const ov::AnyMap& properties = {});

    /**
     * @brief ov::genai::Tokenizer constructor with variable number of properties
     * @param tokenizer_path openvino_tokenizer.xml and openvino_detokenizer.xml should be located in the tokenizer_path
     * @param properties optional properties
     */
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Tokenizer(const std::filesystem::path& tokenizer_path,
              Properties&&... properties)
        : Tokenizer(tokenizer_path, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /**
    * @brief encode a single prompt
    * @param prompt std::string with input prompt
    * @param tokenization_params AnyMap with tokenization parameters, e.g. {"add_special_tokens", false}
    * @return pair of [input_ids, attention_mask]
    */
    TokenizedInputs encode(const std::string prompt, const ov::AnyMap& tokenization_params = {});
    
    /**
    * @brief encode batch of prompts. Left padding will be applied by default
    * @param prompts vector storing batch of prompts
    * @param tokenization_params AnyMap with tokenization parameters, e.g. {"add_special_tokens", false}
    * @return pair of [input_ids, attention_mask]
    */
    TokenizedInputs encode(std::vector<std::string>& prompt, const ov::AnyMap& tokenization_params = {});
    TokenizedInputs encode(std::vector<std::string>&& prompts, const ov::AnyMap& tokenization_params = {});
    TokenizedInputs encode(std::initializer_list<std::string>& prompts, const ov::AnyMap& tokenization_params = {});

    /**
    * @brief encode a single prompt
    * @param prompt std::string with input prompt
    * @param properties tokenization properties, e.g. ov::genai::add_special_tokens(false)
    * @return pair of [input_ids, attention_mask]
    */    
    template <typename... Properties>
    util::EnableIfAllStringAny<TokenizedInputs, Properties...> encode(std::string& prompt, Properties&&... properties) {
        return encode(prompt, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
    * @brief encode batch of prompts. Left padding will be applied by default
    * @param prompts vector storing batch of prompts
    * @param properties tokenization properties, e.g. ov::genai::add_special_tokens(false)
    * @return pair of [input_ids, attention_mask]
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<TokenizedInputs, Properties...> encode(std::vector<std::string>& prompts, Properties&&... properties) {
        return encode(prompts, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
    * @brief decode sequence of tokens
    * @param tokens vector storing tokens
    * @param detokenization_params AnyMap with detokenization parameters, e.g. {"skip_special_tokens", false}
    * @return sequence string
    */
    std::string decode(std::vector<int64_t> tokens, const ov::AnyMap& detokenization_params = {});

    /**
    * @brief decode sequence of tokens
    * @param tokens vector storing tokens
    * @param detokenization_params detokenization parameters,  e.g. ov::genai::skip_special_tokens(true)
    * @return sequence string
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<std::string, Properties...> decode(std::vector<int64_t>& tokens, Properties&&... detokenization_params) {
        return decode(tokens, AnyMap{std::forward<Properties>(detokenization_params)...});
    }

    /**
    * @brief decode tokens. 
    * @param tokens ov::Tensor with tokens with shape [batch_size, seq_len]
    * @param detokenization_params AnyMap with detokenization parameters, e.g. {"skip_special_tokens", false}
    * @return vector of std::string, with size = batch_size
    */
    std::vector<std::string> decode(ov::Tensor tokens, const ov::AnyMap& detokenization_params = {});

    /**
    * @brief decode sequence of tokens
    * @param tokens ov::Tensor with tokens with shape [batch_size, seq_len]
    * @param detokenization_params detokenization parameters,  e.g. ov::genai::skip_special_tokens(true)
    * @return vector of std::string, with size = batch_size
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<std::vector<std::string>, Properties...> decode(ov::Tensor tokens, Properties&&... detokenization_params) {
        return decode(tokens, AnyMap{std::forward<Properties>(detokenization_params)...});
    }

    /**
    * @brief batched decoding of tokens. 
    * @param tokens vector of vectors with tokens, tokens.size() is equal to batch_size
    * @param detokenization_params AnyMap with detokenization parameters, e.g. {"skip_special_tokens", false}
    * @return vector of std::string, with size equal to batch_size
    */
    std::vector<std::string> decode(std::vector<std::vector<int64_t>> tokens, const ov::AnyMap& detokenization_params = {});

    /**
    * @brief decode sequence of tokens
    * @param tokens ov::Tensor with tokens with shape [batch_size, seq_len]
    * @param detokenization_params detokenization parameters,  e.g. ov::genai::skip_special_tokens(true)
    * @return vector of std::string, with size = batch_size
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<std::vector<std::string>, Properties...> decode(std::vector<std::vector<int64_t>> tokens, Properties&&... detokenization_params) {
        return decode(tokens, AnyMap{std::forward<Properties>(detokenization_params)...});
    }

    /**
     * @brief Embeds input prompts with special tags for a chat scenario.
     * 
     * For example, for Qwen family models, the prompt "1+1=" would be transformed into 
     * <|im_start|>user\n1+1=<|im_end|>\n<|im_start|>assistant\n.
     *
     * @param history A vector of maps, with chat history, e.g. [{"role": "user", "content": "prompt"}, ...].
     * @param add_generation_prompt Whether to add an ending that indicates the start of generation.
     * @param chat_template An optional chat template string, if not specified will be taken from the tokenizer.
     * @return A string with the transformed and concatenated prompts from the chat history.
     * @throws Exception if the chat template was unable to parse the input history.
     */
    std::string apply_chat_template(ChatHistory history,
                                    bool add_generation_prompt, 
                                    const std::string& chat_template = {}) const;

    /// @brief Override a chat_template read from tokenizer_config.json.
    /// @param chat_template The new template to override with.
    void set_chat_template(const std::string& chat_template);

    // information about <bos>, <eos> tokens should be public,
    // they are used at least in StreamerBase descendants
    int64_t get_bos_token_id() const;
    int64_t get_eos_token_id() const;
    int64_t get_pad_token_id() const;

    std::string get_bos_token() const;
    std::string get_eos_token() const;
    std::string get_pad_token() const;

    Tokenizer() = default;
    ~Tokenizer();
private:
    class TokenizerImpl;
    std::shared_ptr<TokenizerImpl> m_pimpl;
};

static constexpr ov::Property<bool> add_special_tokens{"add_special_tokens"};
static constexpr ov::Property<bool> skip_special_tokens{"skip_special_tokens"};

}  // namespace genai
}  // namespace ov
