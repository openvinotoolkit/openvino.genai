// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include <filesystem>

using namespace std;

std::pair<ov::Tensor, ov::Tensor> pad_left(ov::Tensor&& input_ids, ov::Tensor&& attention_mask, int64_t pad_token=2);


class Tokenizer {
public:
    int64_t m_eos_token = 2;

    Tokenizer() = default;
    Tokenizer(std::string& tokenizers_path, std::string device="CPU");

    // Tokenizer(std::string& tokenizer_path, std::string& detokenizer_path, std::string device="CPU");
    
    std::pair<ov::Tensor, ov::Tensor> tokenize(std::string prompt);

    std::pair<ov::Tensor, ov::Tensor> tokenize(std::vector<std::string> prompts);
    
    std::pair<ov::Tensor, ov::Tensor> tokenize(std::initializer_list<std::string> text);
    
    std::string detokenize(std::vector<int64_t> tokens);
    
    std::vector<std::string> detokenize(ov::Tensor tokens);
    
    std::vector<std::string> detokenize(std::vector<std::vector<int64_t>> lines);
private:
    ov::InferRequest m_tokenize_request;
    ov::InferRequest m_detokenizer_request;
    std::string m_device;
};


class TextCoutStreamer {
public:
    std::string put(int64_t token);

    std::string end();
    TextCoutStreamer(const Tokenizer& tokenizer, bool m_print_eos_token = false);
    TextCoutStreamer() = default;
    void set_tokenizer(Tokenizer tokenizer);
private:
    bool m_print_eos_token = false;
    Tokenizer m_tokenizer;
    std::vector<int64_t> m_tokens_cache;
    size_t print_len = 0;
    std::function<void (std::string)> m_callback = [](std::string words){ ;};
};
