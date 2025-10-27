// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <initializer_list>
#include <filesystem>
#include <optional>

#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"
#include <openvino/runtime/properties.hpp>

#include "openvino/genai/chat_history.hpp"

namespace ov {
namespace genai {

using ov::genai::JsonContainer;
using ov::genai::ChatHistory;

using Vocab = std::unordered_map<std::string, int64_t>;  // similar to huggingface .get_vocab() output format

struct TokenizedInputs {
    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    std::optional<ov::Tensor> token_type_ids;
};

/**
 * @brief The class is used to encode prompts and decode resulting tokens
 *
 * Chat template is initialized from sources in the following order
 * overriding the previous value:
 * 1. chat_template entry from tokenizer_config.json
 * 2. chat_template entry from processor_config.json
 * 3. chat_template entry from chat_template.json
 * 4. chat_template entry from rt_info section of ov::Model
 * 5. If the template is known to be not supported by GenAI, it's
 *     replaced with a simplified supported version.
*/
class OPENVINO_GENAI_EXPORTS Tokenizer {
public:
    /**
     * @brief ov::genai::Tokenizer constructor.
     * @param tokenizer_path openvino_tokenizer.xml and openvino_detokenizer.xml should be located in the tokenizer_path
     * @param properties Properties passed to ov::Core::compile_model
     */
    explicit Tokenizer(const std::filesystem::path& tokenizer_path, const ov::AnyMap& properties = {});

    /**
     * @brief ov::genai::Tokenizer constructor to initialize directly from model and weights
     *
     * This constructor is used when tokenizer and detokenizer are separate models already loaded into memory.
     * When this constructor is used bos, eos, pad token ids are expected to be in IR.
     * If an IR is older (< 2024.3) then this tokens are default initialized to be ignored.
     * @param tokenizer_model_str tokenizer model string
     * @param tokenizer_weights_tensor ov::Tensor with tokenizer weights
     * @param detokenizer_model_str detokenizer model string
     * @param detokenizer_weights_tensor ov::Tensor with detokenizer weights
     * @param properties Properties passed to ov::Core::compile_model
     */
    Tokenizer(
        const std::string& tokenizer_model_str,
        const ov::Tensor& tokenizer_weights_tensor,
        const std::string& detokenizer_model_str,
        const ov::Tensor& detokenizer_weights_tensor,
        const ov::AnyMap& properties = {}
    );

    /**
     * @brief ov::genai::Tokenizer constructor to initialize directly from model and weights.
     *
     * This constructor is used when tokenizer (or detokenizer) already loaded into memory. Whether it's
     * tokenizer or detokenizer is defined from model input signature. When this constructor is used bos, eos, pad token ids
     * are expected to be in IR. If an IR is older (< 2024.3) then this tokens are default initialized to be ignored.
     * @param model_str model string
     * @param weights_tensor ov::Tensor with model weights
     * @param properties Properties passed to ov::Core::compile_model
     */
    Tokenizer(const std::string& model_str, ov::Tensor& weights_tensor, const ov::AnyMap& properties = {});

    /**
     * @brief ov::genai::Tokenizer constructor with variable number of properties
     * @param tokenizer_model_str tokenizer model string
     * @param tokenizer_weights_tensor ov::Tensor with tokenizer weights
     * @param detokenizer_model_str detokenizer model string
     * @param detokenizer_weights_tensor ov::Tensor with detokenizer weights
     * @param properties optional properties
     */
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Tokenizer(
        const std::string& tokenizer_model_str,
        ov::Tensor& tokenizer_weights_tensor,
        std::string& detokenizer_model_str,
        ov::Tensor& detokenizer_weights_tensor,
        Properties&&... properties
        ) : Tokenizer(tokenizer_model_str, tokenizer_weights_tensor, detokenizer_model_str, detokenizer_weights_tensor, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /**
     * @brief ov::genai::Tokenizer constructor with variable number of properties
     * @param model_str model string
     * @param weights_tensor ov::Tensor with model weights
     * @param properties optional properties
     */
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Tokenizer(const std::string& model_str, ov::Tensor& weights_tensor,
              Properties&&... properties)
        : Tokenizer(model_str, weights_tensor, ov::AnyMap{std::forward<Properties>(properties)...}) { }

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
    * @param tokenization_params AnyMap with tokenization parameters, e.g. {{"add_special_tokens", false}, {"max_length", 128}}
    * @return pair of [input_ids, attention_mask]
    */
    TokenizedInputs encode(const std::string& prompt, const ov::AnyMap& tokenization_params = {});

    /**
    * @brief encode batch of prompts.
    * @param prompts vector storing batch of prompts
    * @param tokenization_params AnyMap with tokenization parameters, e.g. {{"add_special_tokens", false}, {"max_length", 128}}
    * @return pair of [input_ids, attention_mask]
    */
    TokenizedInputs encode(const std::vector<std::string>& prompt, const ov::AnyMap& tokenization_params = {});
    TokenizedInputs encode(const std::initializer_list<std::string>& prompts, const ov::AnyMap& tokenization_params = {});
   
    /**
    * @brief encode paired prompts.
    * 
    * This overload copies prompts to the the pair of vectors, thus is less efficient than encode(prompts_1, prompts_2).
    * In case if efficiency is important, please use encode(prompts_1, prompts_2).
    * @param prompts vector storing batch of prompts
    * @param tokenization_params AnyMap with tokenization parameters, e.g. {{"add_special_tokens", false}, {"max_length", 128}}
    * @return pair of [input_ids, attention_mask]
    */
   TokenizedInputs encode(const std::vector<std::pair<std::string, std::string>>& prompts, const ov::AnyMap& tokenization_params = {});

   /**
   * @brief encode paired prompts.
   * 
   * Prompts should be of the same length, or one of them should be of length 1. In the latest case, the prompt will be
   * broadcasted to the length of the other prompt.
   * @param prompts vector storing batch of prompts
   * @param tokenization_params AnyMap with tokenization parameters, e.g. {{"add_special_tokens", false}, {"max_length", 128}}
   * @return pair of [input_ids, attention_mask]
   */
    TokenizedInputs encode(const std::vector<std::string>& prompts_1, const std::vector<std::string>& prompts_2, const ov::AnyMap& tokenization_params = {});

    /**
    * @brief encode a single prompt
    * @param prompt std::string with input prompt
    * @param add_special_tokens whether to add special tokens
    * @param max_length optional maximum length to which output will be truncated and/or padded. If not defined, taken from IR (where default value from original HF/GGUF model is stored).
    * @param pad_to_max_length either pad to max_length, or pad to the longest sequence in the batch. Default is false.
    * @param padding_side side to pad, either "left" or "right". If not defined value is taken from IR (where default value from original HF/GGUF model is stored).
    * @return pair of [input_ids, attention_mask]
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<TokenizedInputs, Properties...> encode(const std::string& prompt, Properties&&... properties) {
        return encode(prompt, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
    * @brief encode batch of prompts.
    * @param prompts vector storing batch of prompts
    * @param add_special_tokens whether to add special tokens
    * @param max_length optional maximum length to which output will be truncated and/or padded. If not defined, taken from IR (where default value from original HF/GGUF model is stored).
    * @param pad_to_max_length either pad to max_length, or pad to the longest sequence in the batch. Default is false.
    * @param padding_side side to pad, either "left" or "right". If not defined value is taken from IR (where default value from original HF/GGUF model is stored).
    * @return pair of [input_ids, attention_mask]
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<TokenizedInputs, Properties...> encode(const std::vector<std::string>& prompts, Properties&&... properties) {
        return encode(prompts, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
    * @brief decode sequence of tokens
    * @param tokens vector storing tokens
    * @param detokenization_params AnyMap with detokenization parameters, e.g. {"skip_special_tokens", false}
    * @return sequence string
    */
    std::string decode(const std::vector<int64_t>& tokens, const ov::AnyMap& detokenization_params = {});

    /**
    * @brief decode sequence of tokens
    * @param tokens vector storing tokens
    * @param detokenization_params detokenization parameters,  e.g. ov::genai::skip_special_tokens(true)
    * @return sequence string
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<std::string, Properties...> decode(const std::vector<int64_t>& tokens, Properties&&... detokenization_params) {
        return decode(tokens, AnyMap{std::forward<Properties>(detokenization_params)...});
    }

    /**
    * @brief decode tokens.
    * @param tokens ov::Tensor with tokens with shape [batch_size, seq_len]
    * @param detokenization_params AnyMap with detokenization parameters, e.g. {"skip_special_tokens", false}
    * @return vector of std::string, with size = batch_size
    */
    std::vector<std::string> decode(const ov::Tensor& tokens, const ov::AnyMap& detokenization_params = {});

    /**
    * @brief decode sequence of tokens
    * @param tokens ov::Tensor with tokens with shape [batch_size, seq_len]
    * @param detokenization_params detokenization parameters,  e.g. ov::genai::skip_special_tokens(true)
    * @return vector of std::string, with size = batch_size
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<std::vector<std::string>, Properties...> decode(const ov::Tensor& tokens, Properties&&... detokenization_params) {
        return decode(tokens, AnyMap{std::forward<Properties>(detokenization_params)...});
    }

    /**
    * @brief batched decoding of tokens.
    * @param tokens vector of vectors with tokens, tokens.size() is equal to batch_size
    * @param detokenization_params AnyMap with detokenization parameters, e.g. {"skip_special_tokens", false}
    * @return vector of std::string, with size equal to batch_size
    */
    std::vector<std::string> decode(const std::vector<std::vector<int64_t>>& tokens, const ov::AnyMap& detokenization_params = {});

    /**
    * @brief decode sequence of tokens
    * @param tokens ov::Tensor with tokens with shape [batch_size, seq_len]
    * @param detokenization_params detokenization parameters,  e.g. ov::genai::skip_special_tokens(true)
    * @return vector of std::string, with size = batch_size
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<std::vector<std::string>, Properties...> decode(const std::vector<std::vector<int64_t>>& tokens, Properties&&... detokenization_params) {
        return decode(tokens, AnyMap{std::forward<Properties>(detokenization_params)...});
    }

    /**
     * @brief Applies a chat template to format chat history into a prompt string.
     *
     * For example, for Qwen family models, the prompt "1+1=" would be transformed into
     * <|im_start|>user\n1+1=<|im_end|>\n<|im_start|>assistant\n.
     *
     * @param history Chat history containing the conversation messages and optional tools/extra_context. Each message is a JSON-like object, e.g. [{"role": "user", "content": "prompt"}, ...].
     * @param add_generation_prompt Whether to add an ending that indicates the start of generation.
     * @param chat_template An optional custom chat template string, if not specified will be taken from the tokenizer.
     * @param tools An optional JSON-like array of tool definitions to be used in the chat template. If provided, overrides tools from chat history.
     * @param extra_context An optional JSON-like object with additional variables to be used in the chat template. If provided, overrides extra_context from chat history.
     * @return A string with the formatted and concatenated prompts from the chat history.
     * @throws Exception if the chat template was unable to parse the input history.
     */
    std::string apply_chat_template(
        const ChatHistory& history,
        bool add_generation_prompt,
        const std::string& chat_template = {},
        const std::optional<JsonContainer>& tools = std::nullopt,
        const std::optional<JsonContainer>& extra_context = std::nullopt
    ) const;

    /// @brief Override a chat_template read from tokenizer_config.json.
    /// @param chat_template The new template to override with.
    void set_chat_template(const std::string& chat_template);

    // get information about a chat template to check its status, for example whether it is empty
    std::string get_chat_template() const;

    // get original chat template, before any modifications
    std::string get_original_chat_template() const;

    // information about <bos>, <eos> tokens should be public,
    // they are used at least in StreamerBase descendants
    int64_t get_bos_token_id() const;
    int64_t get_eos_token_id() const;
    int64_t get_pad_token_id() const;

    std::string get_bos_token() const;
    std::string get_eos_token() const;
    std::string get_pad_token() const;

    /**
     * @brief Get the vocabulary of the tokenizer.
     *
     * This function retrieves the vocabulary from the detokenizer, which maps
     * token strings to their corresponding integer IDs. Note that some token strings
     * may not be valid UTF-8 encoded. The resulting vocabulary may differ from the
     * original tokenizer's vocabulary due to optimizations during conversion (space symbol
     * swaps, byte fallback reverse, and other preprocessing changes).
     *
     * @return A map of string tokens to int64_t IDs.
     * @throws Exception if the detokenizer is not available.
     */
    Vocab get_vocab() const;

    /**
     * @brief Get the vocabulary vector of the tokenizer.
     *
     * This function retrieves the vocabulary from the detokenizer, which maps
     * token indices to their corresponding string. Note that some token strings
     * may not be valid UTF-8 encoded. The resulting vocabulary may differ from the
     * original tokenizer's vocabulary due to optimizations during conversion (space symbol
     * swaps, byte fallback reverse, and other preprocessing changes).
     *
     * @return A vector of string tokens.
     * @throws Exception if the detokenizer is not available.
     */
    const std::vector<std::string>& get_vocab_vector() const;
    
     /// @brief Check if the tokenizer supports paired input.
    bool supports_paired_input() const;

    Tokenizer() = default;
    ~Tokenizer();
    bool operator==(const Tokenizer& other) const {
        return m_pimpl == other.m_pimpl;
    }
    class TokenizerImpl;
private:
    friend class StructuredOutputConfig;
    friend class Sampler;
    std::shared_ptr<TokenizerImpl> m_pimpl;
};

static constexpr ov::Property<bool> add_second_input{"add_second_input"};
static constexpr ov::Property<bool> add_special_tokens{"add_special_tokens"};
static constexpr ov::Property<bool> skip_special_tokens{"skip_special_tokens"};
static constexpr ov::Property<bool> pad_to_max_length{"pad_to_max_length"};
static constexpr ov::Property<std::string> padding_side{"padding_side"};

}  // namespace genai
}  // namespace ov
