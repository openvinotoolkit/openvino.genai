// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include "sampling_parameters.hpp"


using GenerationResult = std::vector<int64_t>;

class LLMEngine {
    ov::InferRequest m_model_runner;

    GenerationResult greedy_search(ov::Tensor prompts, SamplingParameters sampling_params) {
        ov::Shape prompts_shape = prompts.get_shape();
        size_t batch_size = prompts_shape[0];
        OPENVINO_ASSERT(batch_size == 1);
        
        GenerationResult results;
        results.reserve(sampling_params.max_new_tokens);
        auto attention_mask = ov::Tensor{ov::element::i64, prompts.get_shape()};
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);
        auto position_ids = ov::Tensor{ov::element::i64, prompts.get_shape()};
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
        auto initial_seq_len = prompts.get_shape()[1];

        m_model_runner.set_tensor("input_ids", prompts);
        m_model_runner.set_tensor("attention_mask", attention_mask);
        m_model_runner.set_tensor("position_ids", position_ids);
    
        // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
        m_model_runner.get_tensor("beam_idx").set_shape({batch_size});
        m_model_runner.get_tensor("beam_idx").data<int32_t>()[0] = 0;

        for (size_t i = 0; i < sampling_params.max_new_tokens; ++i) {
            m_model_runner.infer();
            auto logits = m_model_runner.get_tensor("logits");
            ov::Shape logits_shape = logits.get_shape();

            size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2];
            OPENVINO_ASSERT(batch_size == 1);
            // todo: implement for batch > 1

            const float * logits_data = logits.data<const float>() + (seq_len - 1) * vocab_size;
            int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;

            m_model_runner.get_tensor("input_ids").set_shape({batch_size, 1});
            m_model_runner.get_tensor("input_ids").data<int64_t>()[0] = out_token;
            
            m_model_runner.get_tensor("attention_mask").set_shape({batch_size, m_model_runner.get_tensor("attention_mask").get_shape()[1] + 1});
            std::fill_n(m_model_runner.get_tensor("attention_mask").data<int64_t>(), m_model_runner.get_tensor("attention_mask").get_size(), 1);
            
            m_model_runner.get_tensor("position_ids").set_shape({batch_size, 1});
            m_model_runner.get_tensor("position_ids").data<int64_t>()[0] = int64_t(initial_seq_len + i);
            results.emplace_back(out_token);
        }
        return results;
    }

    GenerationResult beam_search(ov::Tensor prompts, SamplingParameters sampling_params) {
        // todo: implement
        GenerationResult results;
        results.reserve(10);
        return results;
    }

    GenerationResult multinomial_sampling(ov::Tensor prompts, SamplingParameters sampling_params) {
        // todo: implement
        GenerationResult results;
        results.reserve(10);
        return results;
    }

public:
    LLMEngine(ov::InferRequest& request) :
          m_model_runner(request) {
            // todo
    }

    // more high level interface
    GenerationResult generate(ov::Tensor prompts, SamplingParameters sampling_params) {
        if (sampling_params.is_gready_sampling()) {
            return greedy_search(prompts, sampling_params);
        } else if (sampling_params.is_beam_search()) {
            return beam_search(prompts, sampling_params);
        } else {  // if (sampling_params.is_multimomial()) {
            return multinomial_sampling(prompts, sampling_params);
        }
    }
};

class LLMPipeline {
    ov::InferRequest m_model_runner;
    std::string m_path;
    SamplingParameters sampling_parameters;

public:
    LLMPipeline(std::string& path) : m_path(path) {
        // load generation config from the file
        // todo
    }

    GenerationResult call() {
        // will call generate inside itself
        GenerationResult results;
        results.reserve(10);
        return results;
    }

};
