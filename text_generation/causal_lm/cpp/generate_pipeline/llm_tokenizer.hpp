// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include <filesystem>


using GenerationResult = std::vector<std::pair<float, std::vector<int64_t>>>;
using namespace std;

std::pair<ov::Tensor, ov::Tensor> pad_left(ov::Tensor&& input_ids, ov::Tensor&& attention_mask, int64_t pad_token=2);


class Tokenizer {
    ov::InferRequest m_tokenize_request;
    ov::InferRequest m_detokenizer_request;
    std::string m_device;

public:
    Tokenizer() = default;
    Tokenizer(std::string& tokenizers_path, std::string device="CPU");

    // Tokenizer(std::string& tokenizer_path, std::string& detokenizer_path, std::string device="CPU");
    
    std::pair<ov::Tensor, ov::Tensor> tokenize(std::string prompt);

    std::pair<ov::Tensor, ov::Tensor> tokenize(std::vector<std::string> prompts);
    
    std::pair<ov::Tensor, ov::Tensor> tokenize(std::initializer_list<std::string> text);
    
    std::string detokenize(std::vector<int64_t> tokens);
    
    std::vector<std::string> detokenize(ov::Tensor tokens);
    
    std::vector<std::string> detokenize(GenerationResult lines);
};


// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
// class TextStreamer {
//     LLMPipeline pipe;
//     std::vector<int64_t> token_cache;
//     size_t print_len = 0;

//     // TextStreamer(Tokenizer)

//     void put(int64_t token) {
//         token_cache.push_back(token);
//         std::string text = pipe.detokenize(token_cache);
//         if (!text.empty() && '\n' == text.back()) {
//             // Flush the cache after the new line symbol
//             std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
//             token_cache.clear();
//             print_len = 0;
// 	        return;
//         }
//         if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
//             // Don't print incomplete text
//             return;
//         }
//         std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
//         print_len = text.size();
//     }

//     void end() {
//         std::string text = pipe.detokenize(token_cache);
//         std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
//         token_cache.clear();
//         print_len = 0;
//     }
// };