
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <mutex>
#include <queue>
#include "openvino/runtime/core.hpp"

#include "tokenizer.hpp"

/*
class InferenceRunner {

public:
    std::queue<std::size_t> m_free_ir_indexes;
    std::vector<ov::InferRequest> m_infer_requests;
    std::shared_ptr<ov::Model> m_model;
    std::mutex m_mutex; // protecting the free ir indexes queue

    explicit InferenceRunner(std::string model_path) {
        ov::Core core;
        core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
        m_model = core.read_model(model_path);
        ov::CompiledModel compiled_model = core.compile_model(m_model, "CPU");
        for (int i = 0; i < 128; i++) {
            m_infer_requests.push_back(compiled_model.create_infer_request());
            m_free_ir_indexes.push(i);
        }
    }
};

class TokenizerRunner : public InferenceRunner {
public:
    using InferenceRunner::InferenceRunner;

    ov::Tensor run(std::string& prompt) {
        int free_ir_index;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            free_ir_index = m_free_ir_indexes.front();
            m_free_ir_indexes.pop();
        }
        m_infer_requests[free_ir_index].set_input_tensor(ov::Tensor{ov::element::string, {1}, &prompt});
        m_infer_requests[free_ir_index].infer();
        ov::Tensor output = m_infer_requests[free_ir_index].get_tensor("input_ids");

        std::unique_lock<std::mutex> lock(m_mutex);
        m_free_ir_indexes.push(free_ir_index);
        return output;
    }

    std::size_t get_eos_token_id() {
        const ov::AnyMap& rt_info = m_model->get_rt_info();
        OPENVINO_ASSERT(rt_info.find("eos_token_id") != rt_info.end(), "Failed to detect \"eos_token_id\" in openvino_tokenizer.xml runtime information");
        return rt_info.at("eos_token_id").as<int64_t>();
    }
};


class DetokenizerRunner : public InferenceRunner {
public:
    using InferenceRunner::InferenceRunner;

    std::string run(std::vector<int64_t>& tokens) {
        int free_ir_index;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            free_ir_index = m_free_ir_indexes.front();
            m_free_ir_indexes.pop();
        }
        m_infer_requests[free_ir_index].set_input_tensor(ov::Tensor{ov::element::i64, {1, tokens.size()}, tokens.data()});
        m_infer_requests[free_ir_index].infer();
        std::string output = m_infer_requests[free_ir_index].get_output_tensor().data<std::string>()[0];

        std::unique_lock<std::mutex> lock(m_mutex);
        m_free_ir_indexes.push(free_ir_index);
        return output;
    }
};
*/

class Tokenizer::Impl {
    const size_t TOKENIZER_BATCH_SIZE = 1;
    //TokenizerRunner m_tokenizer;
    //DetokenizerRunner m_detokenizer;
    ov::InferRequest m_tokenizer;
    ov::InferRequest m_detokenizer;
    std::size_t m_eos_token_id;
    // TODO: Using multiple infer requests hang. For now we synchronize entire execution.
    std::mutex m_tokenizer_mutex;
    std::mutex m_detokenizer_mutex;

public:
    explicit Impl(const std::string& models_path) /*: 
        m_tokenizer(models_path + "/openvino_tokenizer.xml"),
        m_detokenizer(models_path + "/openvino_detokenizer.xml") */
    {
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
        std::unique_lock<std::mutex> lock(m_tokenizer_mutex);
        m_tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {TOKENIZER_BATCH_SIZE}, &prompt});
        m_tokenizer.infer();
        return m_tokenizer.get_tensor("input_ids");
    }

    std::string decode(std::vector<int64_t> tokens) {
        std::unique_lock<std::mutex> lock(m_detokenizer_mutex);
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
