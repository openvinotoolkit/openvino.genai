// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "llm_tokenizer.hpp"
#include <filesystem>

std::pair<ov::Tensor, ov::Tensor> pad_left(ov::Tensor&& input_ids, ov::Tensor&& attention_mask, int64_t pad_token=2);

namespace ov {

class Tokenizer::TokenizerImpl {
public:
    ov::InferRequest m_tokenize_request;
    ov::InferRequest m_detokenizer_request;
    std::string m_device;

    TokenizerImpl() = default;
    TokenizerImpl(std::string& tokenizers_path, std::string device);
    TokenizerImpl(std::string& tokenizer_path, std::string& detokenizer_path, std::string device);
    std::pair<ov::Tensor, ov::Tensor> encode(std::string prompt);

    std::pair<ov::Tensor, ov::Tensor> encode(std::vector<std::string> prompts);
      
    std::string decode(std::vector<int64_t> tokens);
    
    std::vector<std::string> decode(ov::Tensor tokens);
    
    std::vector<std::string> decode(std::vector<std::vector<int64_t>> lines);
};

Tokenizer::TokenizerImpl::TokenizerImpl(std::string& tokenizers_path, std::string device): m_device(device) {
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
    // todo: read eos, bos here
}

Tokenizer::TokenizerImpl::TokenizerImpl(std::string& tokenizer_path, std::string& detokenizer_path, std::string device): m_device(device) {
    ov::Core core;
    
    auto is_xml = [](std::string path) -> bool { return path.compare(path.length() - 4, 4, ".xml") == 0;};
    if (!is_xml(tokenizer_path))
        OPENVINO_THROW("tokenizers_path should be a path to a xml file");
    if (!is_xml(detokenizer_path))
        OPENVINO_THROW("detokenizer_path should be a path to a xml file");
  
    // todo: add loading EOS_TOKEN_ID from IR
    // todo:: OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  
    // tokenizer and detokenizer work on CPU only
    
    m_tokenize_request = core.compile_model(tokenizer_path, "CPU").create_infer_request();
    m_detokenizer_request = core.compile_model(detokenizer_path, "CPU").create_infer_request();
}

Tokenizer::Tokenizer(std::string& tokenizers_path, std::string device) {
    m_pimpl = std::make_shared<TokenizerImpl>(tokenizers_path, device);
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::encode(std::string prompt) {
    return m_pimpl->encode(prompt);
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::TokenizerImpl::encode(std::string prompt) {
    size_t batch_size = 1;
    m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {batch_size}, &prompt});
    m_tokenize_request.infer();

    std::vector<std::vector<int64_t>> input_ids_vec;
    input_ids_vec.reserve(1);
    auto res_tensor = m_tokenize_request.get_tensor("input_ids");
    auto res_shape = res_tensor.get_shape();
    
    for (int i = 0; i < res_shape[0]; ++i) {
        int64_t* start = res_tensor.data<int64_t>() + i * res_shape[1];
        input_ids_vec.emplace_back(std::vector<int64_t>(start, start + res_shape[1]));
    }

    return {m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask")};
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::encode(std::vector<std::string> prompts) {
    return m_pimpl->encode(prompts);
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::TokenizerImpl::encode(std::vector<std::string> prompts) {
    m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {prompts.size()}, prompts.data()});
    auto size_ = m_tokenize_request.get_input_tensor().get_shape();
    m_tokenize_request.infer();

    pad_left(m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask"));
    // todo: fix mask filled with '2' instead of '0'
    ov::Tensor attention_mask = m_tokenize_request.get_tensor("attention_mask");
    int64_t* attention_mask_data = attention_mask.data<int64_t>();
    std::replace(attention_mask_data, attention_mask_data + attention_mask.get_size(), 2, 0);
    
    return {m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask")};
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::encode(std::initializer_list<std::string> text) {
    return encode(std::vector<std::string>(text.begin(), text.end()));
}


std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_pimpl->decode(tokens);
}

std::string Tokenizer::TokenizerImpl::decode(std::vector<int64_t> tokens) {
    size_t batch_size = 1;
    m_detokenizer_request.set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
    m_detokenizer_request.infer();
    return m_detokenizer_request.get_output_tensor().data<std::string>()[0];
}

std::vector<std::string> Tokenizer::decode(ov::Tensor tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::TokenizerImpl::decode(ov::Tensor tokens) {
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

std::vector<std::string> Tokenizer::decode(std::vector<std::vector<int64_t>> lines) {
    return m_pimpl->decode(lines);
}

std::vector<std::string> Tokenizer::TokenizerImpl::decode(std::vector<std::vector<int64_t>> lines) {
    // todo: implement calling detokenizer in a single batch

    std::vector<std::string> results;
    for (auto& line: lines){
        ov::Tensor tokens = ov::Tensor{ov::element::i64, {1, line.size()}, line.data()};
        m_detokenizer_request.set_input_tensor(tokens);
        m_detokenizer_request.infer();
        auto res = m_detokenizer_request.get_output_tensor();
        auto res_str = res.data<std::string>()[0];
        results.emplace_back(res_str);
    }
    
    return results;
}

Tokenizer::~Tokenizer() = default;

} // namespace ov
