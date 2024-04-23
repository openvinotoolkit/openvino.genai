// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include "generate_pipeline/generation_config.hpp"
#include "generate_pipeline/llm_tokenizer.hpp"
#include <filesystem>


using GenerationResult = std::vector<std::pair<float, std::vector<int64_t>>>;
using namespace std;

void update_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask);
void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask);
ov::Tensor init_attention_mask(ov::Tensor& position_ids);
ov::Tensor extend_attention(ov::Tensor attention_mask);
ov::Tensor trimm_tensor(ov::Tensor& tensor, uint64_t seq_len_axis, uint64_t new_seq_len);
void update_kv_cache(ov::InferRequest request, uint64_t seq_len_axis, uint64_t new_seq_len);

class Tokenizer; // forward declaration

class LLMPipeline {
public:
    ov::InferRequest m_model_runner;
    Tokenizer m_tokenizer;
    GenerationConfig m_sampling_parameters;
    std::string m_device;
    ov::AnyMap m_config;
    size_t kv_cache_len = 0;
    ov::Tensor m_attentions_mask_cache;
    std::function<void (std::string)> m_callback = [](std::string ){ ;};
    TextCoutStreamer m_streamer;
    bool is_streamer_set = false;
    
    // TODO: add constructor for specifying manually tokenizer path
    // dir path
    // xml file path
    // compiled model
    // infer request
    // ov::Model

    LLMPipeline(
        std::string& model_path,
        std::string& tokenizer_path,
        std::string& detokenizer_path,
        std::string device="CPU",
        const ov::AnyMap& config={}
    );

    LLMPipeline(std::string& path, std::string device="CPU", const ov::AnyMap& config={});
    GenerationConfig generation_config() const;

    GenerationResult greedy_search(ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig sampling_params);

    GenerationResult beam_search(ov::Tensor prompts, ov::Tensor attention_mask, GenerationConfig sampling_params);

    GenerationResult speculative_sampling(ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig sampling_params);

    GenerationResult multinomial_sampling(ov::Tensor prompts, GenerationConfig sampling_params);

    std::string call(std::string text);

    std::string call(std::string text, GenerationConfig generation_config, bool first_time = false);

    std::vector<std::string> call(std::vector<std::string> text, GenerationConfig sampling_parameters);

    std::string operator()(std::string text);

    std::string operator()(std::string text, GenerationConfig sampling_parameters);

    std::vector<std::string> operator()(std::vector<std::string> text, GenerationConfig sampling_parameters);

    std::vector<std::string> operator()(std::initializer_list<std::string> text, GenerationConfig sampling_parameters);

    GenerationResult generate(ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig sampling_params);

    GenerationResult generate(ov::Tensor input_ids, ov::Tensor attention_mask);

    GenerationResult generate(ov::Tensor input_ids, GenerationConfig sampling_params);

    GenerationResult generate(ov::Tensor input_ids);

    Tokenizer get_tokenizer();

    void set_streamer_callback(std::function<void (std::string)> callback);
};
