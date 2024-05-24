// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "openvino/genai/tokenizer.hpp"
#include "utils.hpp"

namespace {

// todo: remove when openvino-tokenizers will support left padding
std::pair<ov::Tensor, ov::Tensor> pad_left(ov::Tensor&& input_ids, ov::Tensor&& attention_mask, int64_t pad_token) {
    const size_t batch_size = input_ids.get_shape()[0];
    const size_t sequence_length = input_ids.get_shape()[1];
    int64_t* inputs_data = input_ids.data<int64_t>();
    int64_t* attention_mask_data = attention_mask.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;

        // last token in the sequence is not a PAD_TOKEN, skipping
        if (inputs_data[batch_offset + sequence_length - 1] != pad_token)
            continue;

        size_t pad_tokens_number = 0;
        for (int i = sequence_length - 1; i >= 0; i--) {
            const size_t token_offset = batch_offset + i;

            if (inputs_data[token_offset] == pad_token)
                continue;

            if (pad_tokens_number == 0)
                pad_tokens_number = sequence_length - i - 1;

            std::swap(inputs_data[token_offset], inputs_data[token_offset + pad_tokens_number]);
            std::swap(attention_mask_data[token_offset], attention_mask_data[token_offset + pad_tokens_number]);
        }
    }

    return {input_ids, attention_mask};
}

}

namespace ov {
namespace genai {

class Tokenizer::TokenizerImpl {
public:
    ov::InferRequest m_tokenize_request;
    ov::InferRequest m_detokenizer_request;
    int64_t m_pad_token_id = 0;
    int64_t m_bos_token_id = 1;
    int64_t m_eos_token_id = 2;

    TokenizerImpl() = default;
    TokenizerImpl(std::string tokenizers_path, const std::string device, const std::string& ov_tokenizers_path) {
        ov::Core core;
        
        if (ov::genai::utils::is_xml(tokenizers_path))
            OPENVINO_THROW("tokenizers_path should be a path to a dir not a xml file");
    
        if (ov_tokenizers_path.empty()) {
            // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
            core.add_extension(OPENVINO_TOKENIZERS_PATH);
        } else {
            core.add_extension(ov_tokenizers_path + "/libopenvino_tokenizers.so");
        }
        std::shared_ptr<ov::Model> tokenizer_model, detokenizer_model;
        try {
            tokenizer_model = core.read_model(tokenizers_path + "/openvino_tokenizer.xml");
            detokenizer_model = core.read_model(tokenizers_path + "/openvino_detokenizer.xml");
        } catch (...) {
            OPENVINO_THROW("Cannot compile tokenizer and/or detokenizer. Please check that "
                        "openvino_tokenizer.xml and openvino_detokenizer.xml exist in \"" + tokenizers_path + "\"");
        }
        m_tokenize_request = core.compile_model(tokenizer_model, device).create_infer_request();
        m_detokenizer_request = core.compile_model(detokenizer_model, device).create_infer_request();

        auto rt_info = tokenizer_model->get_rt_info();
        if (rt_info.count("eos_token_id") > 0)
            m_eos_token_id = rt_info["eos_token_id"].as<int64_t>();
        if (rt_info.count("bos_token_id") > 0)
            m_bos_token_id = rt_info["bos_token_id"].as<int64_t>();
        if (rt_info.count("pad_token_id") > 0)
            m_pad_token_id = rt_info["pad_token_id"].as<int64_t>();
        }

    std::pair<ov::Tensor, ov::Tensor> encode(std::string prompt) {
        size_t batch_size = 1;
        m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {batch_size}, &prompt});
        m_tokenize_request.infer();
        return {m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask")};
    }

    std::pair<ov::Tensor, ov::Tensor> encode(std::vector<std::string>& prompts) {
        m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {prompts.size()}, prompts.data()});
        auto size_ = m_tokenize_request.get_input_tensor().get_shape();
        m_tokenize_request.infer();
        pad_left(m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask"), m_pad_token_id);
        
        // todo: fix mask filled with '2' instead of '0' 
        // https://github.com/openvinotoolkit/openvino_tokenizers/pull/90 should've fixed this
        ov::Tensor attention_mask = m_tokenize_request.get_tensor("attention_mask");
        int64_t* attention_mask_data = attention_mask.data<int64_t>();
        std::replace(attention_mask_data, attention_mask_data + attention_mask.get_size(), 2, 0);
        
        return {m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask")};
    }

    std::string decode(std::vector<int64_t> tokens) {
        size_t batch_size = 1;
        m_detokenizer_request.set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
        m_detokenizer_request.infer();
        return m_detokenizer_request.get_output_tensor().data<std::string>()[0];
    }

    std::vector<std::string> decode(ov::Tensor tokens) {
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

    std::vector<std::string> decode(std::vector<std::vector<int64_t>> lines) {
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
};

Tokenizer::Tokenizer(const std::string& tokenizers_path, const std::string& device, const std::string& ov_tokenizers_path) {
    m_pimpl = std::make_shared<TokenizerImpl>(tokenizers_path, device, ov_tokenizers_path);
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::encode(const std::string prompt) {
    return m_pimpl->encode(std::move(prompt));
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::encode(std::vector<std::string>& prompts) {
    return m_pimpl->encode(prompts);
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::encode(std::vector<std::string>&& prompts) {
    return m_pimpl->encode(prompts);
}

std::pair<ov::Tensor, ov::Tensor> Tokenizer::encode(std::initializer_list<std::string>& text) {
    return encode(std::vector<std::string>(text.begin(), text.end()));
}

std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(ov::Tensor tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(std::vector<std::vector<int64_t>> lines) {
    return m_pimpl->decode(lines);
}

int64_t Tokenizer::get_bos_token_id() const {
    return m_pimpl->m_bos_token_id;
}

int64_t Tokenizer::get_eos_token_id() const {
    return m_pimpl->m_eos_token_id;
}

int64_t Tokenizer::get_pad_token_id() const {
    return m_pimpl->m_pad_token_id;
}

void Tokenizer::set_pad_token_id(int64_t pad_token_id) {
    m_pimpl->m_pad_token_id = pad_token_id;
}

void Tokenizer::set_bos_token_id(int64_t bos_token_id) {
    m_pimpl->m_bos_token_id = bos_token_id;
}

void Tokenizer::set_eos_token_id(int64_t eos_token_id) {
    m_pimpl->m_eos_token_id = eos_token_id;
}

Tokenizer::~Tokenizer() = default;

}  // namespace genai
}  // namespace ov
