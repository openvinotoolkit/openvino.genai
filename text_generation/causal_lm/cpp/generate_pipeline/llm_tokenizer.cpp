// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "generate_pipeline/llm_tokenizer.hpp"
#include <filesystem>


Tokenizer::Tokenizer(std::string& tokenizers_path, std::string device): m_device(device) {
    ov::Core core;
    
    auto is_xml = [](std::string path) -> bool { return path.compare(path.length() - 4, 4, ".xml") == 0;};
    
    if (is_xml(tokenizers_path))
        OPENVINO_THROW("tokenizers_path should be a path to a dir not to xml file");
  
    // todo: add loading EOS_TOKEN_ID from IR
    // todo:: OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  
    // tokenizer and detokenizer work on CPU only
    
    m_tokenize_request = core.compile_model(tokenizers_path + "/openvino_tokenizer.xml", "CPU").create_infer_request();
    m_detokenizer_request = core.compile_model(tokenizers_path + "/openvino_detokenizer.xml", "CPU").create_infer_request();
}

// Tokenizer::Tokenizer(std::string& tokenizer_path, std::string& detokenizer_path, std::string device="CPU") {

// }

std::pair<ov::Tensor, ov::Tensor> Tokenizer::tokenize(std::string prompt) {
    size_t batch_size = 1;
    m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {batch_size}, &prompt});
    m_tokenize_request.infer();

    vector<vector<int64_t>> input_ids_vec;
    input_ids_vec.reserve(1);
    auto res_tensor = m_tokenize_request.get_tensor("input_ids");
    auto res_shape = res_tensor.get_shape();
    
    for (int i = 0; i < res_shape[0]; ++i) {
        int64_t* start = res_tensor.data<int64_t>() + i * res_shape[1];
        input_ids_vec.emplace_back(std::vector<int64_t>(start, start + res_shape[1]));
    }

    return {m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask")};
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::tokenize(std::vector<std::string> prompts) {
    m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {prompts.size()}, prompts.data()});
    auto size_ = m_tokenize_request.get_input_tensor().get_shape();
    m_tokenize_request.infer();

    pad_left(m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask"));
    // todo: fix mask filled with '2' instead of '0'
    ov::Tensor attention_mask = m_tokenize_request.get_tensor("attention_mask");
    int64_t* attention_mask_data = attention_mask.data<int64_t>();
    std::replace(attention_mask_data, attention_mask_data + attention_mask.get_size(), 2, 0);
    
    vector<vector<int64_t>> input_ids_vec;
    vector<vector<int64_t>> atten_mask_vec;
    
    input_ids_vec.reserve(prompts.size());
    atten_mask_vec.reserve(prompts.size());
    auto res_tensor = m_tokenize_request.get_tensor("input_ids");
    auto atten_tensor = m_tokenize_request.get_tensor("attention_mask");
    auto res_shape = res_tensor.get_shape();
    
    for (int i = 0; i < res_shape[0]; ++i) {
        int64_t* start = res_tensor.data<int64_t>() + i * res_shape[1];
        input_ids_vec.emplace_back(std::vector<int64_t>(start, start + res_shape[1]));
        
        int64_t* atten_start = atten_tensor.data<int64_t>() + i * res_shape[1];
        atten_mask_vec.emplace_back(std::vector<int64_t>(atten_start, atten_start + res_shape[1]));
    }

    return {m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask")};
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::tokenize(std::initializer_list<std::string> text) {
    return tokenize(std::vector<std::string>(text.begin(), text.end()));
}


std::string Tokenizer::detokenize(std::vector<int64_t> tokens) {
    size_t batch_size = 1;
    m_detokenizer_request.set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
    m_detokenizer_request.infer();
    return m_detokenizer_request.get_output_tensor().data<std::string>()[0];
}

std::vector<std::string> Tokenizer::detokenize(ov::Tensor tokens) {
    m_detokenizer_request.set_input_tensor(tokens);
    auto shape = tokens.get_shape();
    auto data = tokens.data<int64_t>();
    m_detokenizer_request.infer();
    auto res = m_detokenizer_request.get_output_tensor();
    
    std::vector<std::string> strings;
    for (int i = 0; i < res.get_shape()[0]; ++i) {
        strings.emplace_back(res.data<std::string>()[i]);
    }
    return strings;
}

std::vector<std::string> Tokenizer::detokenize(GenerationResult lines) {
    // todo: implement calling detokenizer in a single batch

    std::vector<std::string> strings;
    for (auto& [score, line]: lines){
        ov::Tensor tokens = ov::Tensor{ov::element::i64, {1, line.size()}, line.data()};
        m_detokenizer_request.set_input_tensor(tokens);
        m_detokenizer_request.infer();
        auto res = m_detokenizer_request.get_output_tensor();
        auto res_str = res.data<std::string>()[0];
        strings.emplace_back(res_str);
    }
    
    return strings;
}

TextCoutStreamer::TextCoutStreamer(const Tokenizer& tokenizer, bool print_eos_token) {
    m_tokenizer = tokenizer;
    m_print_eos_token = print_eos_token;
}

std::string TextCoutStreamer::put(int64_t token) {
    std::stringstream res;

    // do not print anything and flush cache if EOS token is met
    if (token == m_tokenizer.m_eos_token) {
        return end();
    }

    m_tokens_cache.push_back(token);
    std::string text = m_tokenizer.detokenize(m_tokens_cache);
    if (!text.empty() && '\n' == text.back()) {
        // Flush the cache after the new line symbol
        res << std::string_view{text.data() + print_len, text.size() - print_len};
        m_tokens_cache.clear();
        print_len = 0;
        return res.str();
    }
    if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
        // Don't print incomplete text
        return res.str();
    }
    res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
    print_len = text.size();
    return res.str();
}

std::string TextCoutStreamer::end() {
    std::stringstream res;
    std::string text = m_tokenizer.detokenize(m_tokens_cache);
    res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
    m_tokens_cache.clear();
    print_len = 0;
    return res.str();
}

void TextCoutStreamer::set_tokenizer(Tokenizer tokenizer) {
    this->m_tokenizer = tokenizer;
}
