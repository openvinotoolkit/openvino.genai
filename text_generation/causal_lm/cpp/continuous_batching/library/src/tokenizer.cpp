
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/runtime/core.hpp"

#include "tokenizer.hpp"

class Tokenizer::Impl {
    const size_t TOKENIZER_BATCH_SIZE = 1;
    ov::InferRequest m_tokenizer, m_detokenizer;
    std::size_t m_eos_token_id;

public:
    explicit Impl(const std::string& models_path) {
        ov::Core core;
        core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt

        std::shared_ptr<ov::Model> tokenizer_model = core.read_model(models_path + "/openvino_tokenizer.xml");
        const ov::AnyMap& rt_info = tokenizer_model->get_rt_info();
        OPENVINO_ASSERT(rt_info.find("eos_token_id") != rt_info.end(), "Failed to detect \"eos_token_id\" in openvino_tokenizer.xml runtime information");
        m_eos_token_id = rt_info.at("eos_token_id").as<int64_t>();

        // tokenizer and detokenizer work on CPU only
        m_tokenizer = core.compile_model(
            tokenizer_model, "CPU").create_infer_request();
        m_detokenizer = core.compile_model(
            models_path + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    }

    ov::Tensor encode(std::string prompt) {
        m_tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {TOKENIZER_BATCH_SIZE}, &prompt});
        m_tokenizer.infer();
        return m_tokenizer.get_tensor("input_ids");
    }

    std::string decode(std::vector<int64_t> tokens) {
        m_detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {TOKENIZER_BATCH_SIZE, tokens.size()}, tokens.data()});
        m_detokenizer.infer();
        return m_detokenizer.get_output_tensor().data<std::string>()[0];
    }

    size_t get_eos_token_id() const {
        return m_eos_token_id;
    }
};

Tokenizer::Tokenizer(const std::string& models_path) {
    m_impl = std::make_shared<Impl>(models_path);
}

ov::Tensor Tokenizer::encode(std::string prompt) {
    return m_impl->encode(prompt);
}

std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_impl->decode(tokens);
}

size_t Tokenizer::get_eos_token_id() const {
    return m_impl->get_eos_token_id();
}
