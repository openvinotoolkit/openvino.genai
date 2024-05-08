// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include <filesystem>

namespace ov {
    
class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer();
    Tokenizer(std::string& tokenizers_path, std::string device="CPU");
    Tokenizer(std::string& tokenizer_path, std::string& detokenizer_path, std::string device="CPU");
    
    std::pair<ov::Tensor, ov::Tensor> encode(std::string prompt);
    std::pair<ov::Tensor, ov::Tensor> encode(std::vector<std::string> prompts);
    std::pair<ov::Tensor, ov::Tensor> encode(std::initializer_list<std::string> text);
    
    std::string decode(std::vector<int64_t> tokens);
    std::vector<std::string> decode(ov::Tensor tokens);
    std::vector<std::string> decode(std::vector<std::vector<int64_t>> lines);

    int64_t m_eos_token = 2;  // todo: read from rt_info
    int64_t m_bos_token = 1;  // todo: read from rt_info
private:
    class TokenizerImpl;
    std::shared_ptr<TokenizerImpl> m_pimpl;
};

} // namespace ov
