// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include <filesystem>

namespace ov {

/**
* @brief class used to encode prompts and decode resulting tokens
*/
class Tokenizer {
public:
    /**
    * @brief ov::Tokenizer constructor.
    * @param tokenizer_path openvino_tokenizer.xml and openvino_detokenizer.xml should be located in the tokenizer_path
    * @param device device. Currently only 'CPU' is supported
    */
    Tokenizer(const std::string tokenizers_path, const std::string device="CPU");

    /**
    * @brief encode a single prompt
    * @return pair of [input_ids, attention_mask]
    */
    std::pair<ov::Tensor, ov::Tensor> encode(std::string prompt);
    
    /**
    * @brief encode batch of prompts. Left padding will be applied by default
    * @param prompts vector storing batch of prompts
    * @return pair of [input_ids, attention_mask]
    */
    std::pair<ov::Tensor, ov::Tensor> encode(std::vector<std::string> prompts);
    std::pair<ov::Tensor, ov::Tensor> encode(std::initializer_list<std::string> prompts);
    
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

    int64_t m_bos_token_id = 1;  // todo: read from rt_info
    int64_t m_eos_token_id = 2;  // todo: read from rt_info

    Tokenizer() = default;
    ~Tokenizer();
private:
    class TokenizerImpl;
    std::shared_ptr<TokenizerImpl> m_pimpl;
};

} // namespace ov
