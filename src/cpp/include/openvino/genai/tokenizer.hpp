// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <initializer_list>
#include <openvino/runtime/tensor.hpp>
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

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
    * @brief ov::Tokenizer constructor.
    * @param tokenizer_path openvino_tokenizer.xml and openvino_detokenizer.xml should be located in the tokenizer_path
    */
    Tokenizer(const std::string& tokenizer_path);

    /**
    * @brief encode a single prompt
    * @return pair of [input_ids, attention_mask]
    */
    TokenizedInputs encode(const std::string prompt);
    
    /**
    * @brief encode batch of prompts. Left padding will be applied by default
    * @param prompts vector storing batch of prompts
    * @return pair of [input_ids, attention_mask]
    */
    TokenizedInputs encode(std::vector<std::string>& prompts);
    TokenizedInputs encode(std::vector<std::string>&& prompts);
    TokenizedInputs encode(std::initializer_list<std::string>& prompts);
    
    /**
    * @brief decode sequence of tokens
    * @param tokens vector storing tokens
    * @return sequence string
    */
    std::string decode(std::vector<int64_t> tokens);
    
    /**
    * @brief decode tokens. 
    * @param tokens ov::Tensor with tokens with shape [batch_size, seq_len]
    * @return vector of std::string, with size = batch_size
    */
    std::vector<std::string> decode(ov::Tensor tokens);

    /**
    * @brief batched decoding of tokens. 
    * @param tokens vector of vectors with tokens, tokens.size() is equal to batch_size
    * @return vector of std::string, with size equal to batch_size
    */
    std::vector<std::string> decode(std::vector<std::vector<int64_t>> tokens);

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

/**
* @brief Returns an absolute path. The path is this library's directory
 * concatenated with openvino_tokenizers OS specific
 * * name (.so, .dll, .dylib, lib prefix). This is part of the interface
 * because it's reused in Python bindings.
 * tokenizers_relative_to_genai() and ScopedVar allow passing a path to
 * openvino_tokenizers through env var removing one argument from
 * Tokenizer's constructor.
*/
OPENVINO_GENAI_EXPORTS std::filesystem::path tokenizers_relative_to_genai();

/**
* @brief Sets ENVIRONMENT_VARIABLE_NAME to environment_variable_value
 * and unsets in destructor. Does nothing if ENVIRONMENT_VARIABLE_NAME
 * was already defined.
*/
class OPENVINO_GENAI_EXPORTS ScopedVar {
public:
    explicit ScopedVar(const std::string& environment_variable_value);
    ~ScopedVar();
    bool was_already_set;
    static constexpr char ENVIRONMENT_VARIABLE_NAME[] = "OPENVINO_TOKENIZERS_PATH_GENAI";
};
}  // namespace genai
}  // namespace ov
