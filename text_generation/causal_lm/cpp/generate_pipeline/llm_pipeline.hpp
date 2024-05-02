// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include "generate_pipeline/generation_config.hpp"
#include "generate_pipeline/llm_tokenizer.hpp"
#include <filesystem>

using namespace std;

void update_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask);
void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask);
ov::Tensor init_attention_mask(ov::Tensor& position_ids);
ov::Tensor extend_attention(ov::Tensor attention_mask);
ov::Tensor trimm_tensor(ov::Tensor& tensor, uint64_t seq_len_axis, uint64_t new_seq_len);
void update_kv_cache(ov::InferRequest request, uint64_t seq_len_axis, uint64_t new_seq_len);

std::pair<int64_t, float> softmax(const ov::Tensor& logits, const size_t batch_idx);

class Tokenizer; // forward declaration

namespace ov {

template <class T, class ItemType>
class ResultsIterator {
    public:
        ResultsIterator(const T& results, size_t index) : results(results), index(index) {}

        bool operator!=(const ResultsIterator& other) const;

        ItemType operator*() const;

        ResultsIterator& operator++();

    private:
        const T& results;
        size_t index;
};

class TextScorePair {
public:
    std::string text;
    float score;
};

class TokensScorePair {
public:
    std::vector<int64_t> tokens;
    float score;
};

class GenerationResults {
public:
    std::vector<std::vector<int64_t>> tokens;
    std::vector<float> scores;

    TokensScorePair operator[](size_t index) const;

    ResultsIterator<GenerationResults, TokensScorePair> begin() const;

    ResultsIterator<GenerationResults, TokensScorePair> end() const;
};

class PipelineResults {
public:
    std::vector<std::string> texts;
    std::vector<float> scores;
    
    TextScorePair operator[](size_t index) const;

    ResultsIterator<PipelineResults, TextScorePair> begin() const;

    ResultsIterator<PipelineResults, TextScorePair> end() const;
};



class LLMPipeline {
public:
    ov::InferRequest m_model_runner;
    Tokenizer m_tokenizer;
    GenerationConfig m_sampling_parameters;
    std::string m_device;
    ov::AnyMap m_config;
    ov::Tensor m_attentions_mask_cache;
    bool is_streamer_set = false;
    std::string m_chat_template = "";
    
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

    GenerationResults greedy_search(ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig sampling_params);

    GenerationResults beam_search(ov::Tensor prompts, ov::Tensor attention_mask, GenerationConfig sampling_params);

    GenerationResults speculative_sampling(ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig sampling_params);

    GenerationResults multinomial_sampling(ov::Tensor prompts, GenerationConfig sampling_params);

    std::string call(std::string text);

    std::string call(std::string text, GenerationConfig generation_config);

    PipelineResults call(std::vector<std::string> text, GenerationConfig sampling_parameters);

    std::string operator()(std::string text);

    std::string operator()(std::string text, GenerationConfig sampling_parameters);

    PipelineResults operator()(std::vector<std::string> text, GenerationConfig sampling_parameters);

    PipelineResults operator()(std::initializer_list<std::string> text, GenerationConfig sampling_parameters);

    GenerationResults generate(ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig sampling_params);

    GenerationResults generate(ov::Tensor input_ids, ov::Tensor attention_mask);

    GenerationResults generate(ov::Tensor input_ids, GenerationConfig sampling_params);

    GenerationResults generate(ov::Tensor input_ids);

    Tokenizer get_tokenizer();

    std::string apply_chat_template(std::string prompt, std::string role = "user") const;

    void set_streamer(std::function<void (std::string)> callback);
    void start_conversation();
    void stop_conversation();
    void reset_state();
private:
    TextCoutStreamer m_streamer;
    std::function<void (std::string)> m_streamer_callback = [](std::string ){ ;};
    bool is_chat_conversation = false;
};



} // namespace ov