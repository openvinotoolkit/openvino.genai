// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>

#include "generation_config.hpp"
#include "timer.hpp"

struct Token {
    float m_log_prob = 0.;
    int64_t m_index = 0;

    Token(float log_prob, int64_t index) : m_log_prob(log_prob), m_index(index) {}
    Token() = default;
};

struct LogitTransformWrapper {
    // Orignal logits buffer
    float * m_logit_data;
    // Tokens considered 
    size_t m_effective_size;
    // Vector with post transforation positioning information.
    // Initialized only when needed. If there's no need for manipulating logits array we use
    // original buffer to avoid unnecessary copying and sorting.
    std::unique_ptr<std::vector<Token>> m_tokens = nullptr;

    LogitTransformWrapper(float* logit_data, size_t effective_size): m_logit_data(logit_data), m_effective_size(effective_size) {}

    size_t get_effective_size() { return m_effective_size; }

    void set_effective_size(size_t size) {
        m_effective_size = size;
        if (m_tokens)
            m_tokens->resize(size);
    }

    Token get_token_at_position(size_t position) const {
        if (m_tokens)
            return m_tokens->at(position);
        else
            return Token(m_logit_data[position], position);
    }

    float get_logit_at_position(size_t position) const {
        if (m_tokens)
            return m_tokens->at(position).m_log_prob;
        else
            return m_logit_data[position];
    }

    bool tokens_vector_initialized() const { return m_tokens != nullptr; }

    // Creates vector of Tokens. Only first call is effective.
    void _initialize_tokens_vector() {
        if (m_tokens == nullptr) {
            m_tokens = std::make_unique<std::vector<Token>>();
            m_tokens->reserve(get_effective_size());
            for (int i = 0; i < get_effective_size(); i++) {
                m_tokens->push_back(Token(m_logit_data[i], i));
            }
        }
    }

    // Currently used in top_p transform
    void initialize_sorted_token_vector() {
        _initialize_tokens_vector();
        std::sort(m_tokens->begin(), m_tokens->end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
    }

    // Currently used in top_k transform
    void initialize_partially_sorted_token_vector(size_t top_k) {
        _initialize_tokens_vector();
        std::partial_sort(m_tokens->begin(), m_tokens->begin() + top_k, m_tokens->end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
    }

    // Note that with above implemetation only one initialization is effective so order of transformations matter. 
};

namespace LogitTransformers {
using TokenIds = std::vector<int64_t>;

class ILogitTransformer {
public:
    virtual void apply(LogitTransformWrapper& logits) = 0;

    virtual bool is_applicable(size_t generated_tokens_cnt = 0) {
        return true;
    }
};

class TopPFilter : public ILogitTransformer {
public:
    TopPFilter(double top_p) : m_top_p(top_p) {}

    void apply(LogitTransformWrapper& logits) override {
        if (!logits.tokens_vector_initialized())
            logits.initialize_sorted_token_vector();
        float probability_sum = 0.0f;
        size_t nucleus_size = 0;
        for (int i = 0; i < logits.get_effective_size(); i++) {
            probability_sum += logits.get_logit_at_position(i);
            nucleus_size += 1;
            if (probability_sum > m_top_p) break;
        }
        logits.set_effective_size(nucleus_size);
    }

protected:
    double m_top_p = 0.f;
};

class TopKFilter : public ILogitTransformer {
public:
    TopKFilter(size_t top_k) : m_top_k(top_k) {}

    void apply(LogitTransformWrapper& logits) override {
        if (!logits.tokens_vector_initialized())
            logits.initialize_partially_sorted_token_vector(m_top_k);
        size_t top_k = logits.get_effective_size() >= m_top_k ? m_top_k : logits.get_effective_size();
        logits.set_effective_size(top_k);
    }

protected:
    size_t m_top_k = 0;
};

class TemperatureLogitTransform : public ILogitTransformer {
public:
    TemperatureLogitTransform(double temperature) : m_temperature(temperature) {};

    void apply(LogitTransformWrapper& logits) override {
        float max_logit = 0.0;
        for (int i = 0; i < logits.get_effective_size(); i++) {
            if(logits.m_logit_data[i] > max_logit) {
                max_logit = logits.m_logit_data[i];
            }
        }

        float norm_sum = 0.0;
        for (int i = 0; i < logits.get_effective_size(); i++) {
            logits.m_logit_data[i] = expf((logits.m_logit_data[i] - max_logit) / this->m_temperature);
            norm_sum += logits.m_logit_data[i];
        }

        for (int i = 0; i < logits.get_effective_size(); i++) {
            logits.m_logit_data[i] /= norm_sum;
        }
    }

protected:
    float m_temperature = 0.f;
};


class IPenaltyTransformer : public ILogitTransformer {
public:
    void set_unique_generated_token_ids(const std::shared_ptr<std::map<int64_t, size_t>>& unique_generated_token_ids) {
        if (unique_generated_token_ids != nullptr) {
            m_unique_generated_token_ids = unique_generated_token_ids;
        } else {
            m_unique_generated_token_ids = std::shared_ptr<std::map<int64_t, size_t>>(new std::map<int64_t, size_t>);
        }
    }

    void extract_generated_tokens(const TokenIds& input_ids) {
        set_unique_generated_token_ids(m_unique_generated_token_ids);

        for (const auto& input_id : input_ids) {
            if (m_unique_generated_token_ids->count(input_id)) {
                m_unique_generated_token_ids->at(input_id)++;
            } else {
                m_unique_generated_token_ids->insert({input_id, 1});
            }
        }
    }

protected:
    std::shared_ptr<std::map<int64_t, size_t>> m_unique_generated_token_ids = nullptr;
    double m_penalty = 0.f;
};

class RepetitionPenaltyTransform : public IPenaltyTransformer {
public:
    RepetitionPenaltyTransform(double repetition_penalty) {
        m_penalty = repetition_penalty;
    };

    void apply(LogitTransformWrapper& logits) override {
        //std::vector<Token> output(logits.m_logit_data.begin(), logits.m_logit_data.end());
        size_t vocab_size = logits.get_effective_size();
        for (const auto& prompt_id : *m_unique_prompt_token_ids) {
            OPENVINO_ASSERT((prompt_id >= 0) && (prompt_id < vocab_size), "input_ids token out of bounds");
            //OPENVINO_ASSERT(logits.m_logit_data[prompt_id].m_index == prompt_id, "logits.m_logit_data must have original index order");
            if (logits.m_logit_data[prompt_id] >= 0) {
                logits.m_logit_data[prompt_id] /= m_penalty;
            } else {
                logits.m_logit_data[prompt_id] *= m_penalty;
            };
        }
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            //OPENVINO_ASSERT(logits.m_logit_data[input_id].m_index == input_id, "logits.m_logit_data must have original index order");
            if (logits.m_logit_data[input_id] >= 0) {
                logits.m_logit_data[input_id] /= m_penalty;
            } else {
                logits.m_logit_data[input_id] *= m_penalty;
            };
        }
        // return output;
    }

    void apply(LogitTransformWrapper& logits, const TokenIds& input_ids) {
        set_unique_prompt_token_ids(nullptr);
        extract_generated_tokens(input_ids);
        apply(logits);
    }

    void set_unique_prompt_token_ids(const std::shared_ptr<std::set<int64_t>>& unique_prompt_token_ids) {
        if (unique_prompt_token_ids != nullptr) {
            m_unique_prompt_token_ids = unique_prompt_token_ids;
        } else {
            m_unique_prompt_token_ids = std::shared_ptr<std::set<int64_t>>(new std::set<int64_t>);
        }
    }

protected:
    std::shared_ptr<std::set<int64_t>> m_unique_prompt_token_ids = nullptr;
};

class EOSPenaltyTransform : public ILogitTransformer {
public:
    EOSPenaltyTransform(size_t eos_token_id, size_t min_generated_tokens) : 
        m_eos_token_id(eos_token_id), m_applicable_tensor_len(min_generated_tokens) {}

    void apply(LogitTransformWrapper& logits) {
        for (int i = 0; i < logits.get_effective_size(); i++) {
            // This does not look right
            if (i == m_eos_token_id) {
                logits.m_logit_data[i] = 0.f;
            }
        }
    }
    

    bool is_applicable(size_t generated_tokens_cnt = 0) override {
        return generated_tokens_cnt < m_applicable_tensor_len;
    }

protected:
    size_t m_applicable_tensor_len = std::numeric_limits<size_t>::max();
    size_t m_eos_token_id;
};

class FrequencyPenaltyTransform : public IPenaltyTransformer {
public:
    FrequencyPenaltyTransform(double value) {
        m_penalty = value;
    };

    void apply(LogitTransformWrapper& logits) override {
        size_t vocab_size = logits.get_effective_size();
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            //OPENVINO_ASSERT(logits.m_logit_data[input_id].m_index == input_id, "logits.m_logit_data must have original index order");
            if (logits.m_logit_data[input_id] >= 0) {
                logits.m_logit_data[input_id] -= m_penalty * input_id_pair.second;
            } else {
                logits.m_logit_data[input_id] += m_penalty * input_id_pair.second;
            };
        }
    }

    void apply(LogitTransformWrapper& logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        apply(logits);
    }
};

class PresencePenaltyTransform : public IPenaltyTransformer {
public:
    PresencePenaltyTransform(double value) {
        m_penalty = value;
    };

    void apply(LogitTransformWrapper& logits) override {
        size_t vocab_size = logits.get_effective_size();
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            // OPENVINO_ASSERT(logits.m_logit_data[input_id].m_index == input_id, "logits.m_logit_data must have original index order");
            if (logits.m_logit_data[input_id] >= 0) {
                logits.m_logit_data[input_id] -= m_penalty;
            } else {
                logits.m_logit_data[input_id] += m_penalty;
            };
        }
    }

    void apply(LogitTransformWrapper& logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        apply(logits);
    }
};


class ProbabilityNormalizeTransform : public ILogitTransformer {
public:
    ProbabilityNormalizeTransform() = default;

    void apply(LogitTransformWrapper& logits) override {
        float norm_sum = 0.0;
        for (int i = 0; i < logits.get_effective_size(); i++) {
            norm_sum += logits.m_logit_data[i];
        }

        for (int i = 0; i < logits.get_effective_size(); i++) {
            logits.m_logit_data[i] /= norm_sum;
        }
    }
};

} // namespace LogitTransformers

class LogitProcessor {
protected:
    std::vector<std::shared_ptr<LogitTransformers::ILogitTransformer>> m_logit_transformers;
    
    std::shared_ptr<std::map<int64_t, size_t>> m_unique_generated_token_ids = std::shared_ptr<std::map<int64_t, size_t>>(new std::map<int64_t, size_t>);
    std::shared_ptr<std::set<int64_t>> m_unique_prompt_token_ids = std::shared_ptr<std::set<int64_t>>(new std::set<int64_t>);
    size_t m_generated_tokens = 0;

public:
    LogitProcessor(const GenerationConfig& sampling_params,
                   const LogitTransformers::TokenIds& input_ids) {
        for (const auto& input_id : input_ids) {
            m_unique_prompt_token_ids->insert(input_id);
        }

        if (sampling_params.min_new_tokens > 0) {
            m_logit_transformers.emplace_back(
                new LogitTransformers::EOSPenaltyTransform(sampling_params.eos_token_id, sampling_params.min_new_tokens)
            );
        }

        if (sampling_params.is_multinomial() || sampling_params.is_greedy_sampling()) {
            if (sampling_params.repetition_penalty != 1.0f) {
                std::shared_ptr<LogitTransformers::RepetitionPenaltyTransform> transformer = 
                    std::shared_ptr<LogitTransformers::RepetitionPenaltyTransform>(new LogitTransformers::RepetitionPenaltyTransform(sampling_params.repetition_penalty));
                transformer->set_unique_prompt_token_ids(m_unique_prompt_token_ids);
                transformer->set_unique_generated_token_ids(m_unique_generated_token_ids);
                m_logit_transformers.push_back(transformer);
            }
            if (sampling_params.presence_penalty != 0.0f) {
                std::shared_ptr<LogitTransformers::PresencePenaltyTransform> transformer = 
                    std::shared_ptr<LogitTransformers::PresencePenaltyTransform>(new LogitTransformers::PresencePenaltyTransform(sampling_params.presence_penalty)); 
                transformer->set_unique_generated_token_ids(m_unique_generated_token_ids);
                m_logit_transformers.push_back(transformer);
                
            }
            if (sampling_params.frequence_penalty != 0.0f) {
                std::shared_ptr<LogitTransformers::FrequencyPenaltyTransform> transformer = 
                    std::shared_ptr<LogitTransformers::FrequencyPenaltyTransform>(new LogitTransformers::FrequencyPenaltyTransform(sampling_params.frequence_penalty)); 
                transformer->set_unique_generated_token_ids(m_unique_generated_token_ids);
                m_logit_transformers.push_back(transformer);
            }

            if (sampling_params.is_multinomial()) {
                m_logit_transformers.emplace_back(new LogitTransformers::TemperatureLogitTransform(sampling_params.temperature));
                
                if (sampling_params.top_p != 0.0f) {
                    m_logit_transformers.emplace_back(new LogitTransformers::TopPFilter(sampling_params.top_p));
                }
                if (sampling_params.top_k > 0) {
                    m_logit_transformers.emplace_back(new LogitTransformers::TopKFilter(sampling_params.top_k));
                }
                // Normalization happens in temperature transformer and top_p/top_k do not require it
                // m_logit_transformers.emplace_back(new LogitTransformers::ProbabilityNormalizeTransform());
            }
        }
    }

    void apply(LogitTransformWrapper& logits) {
        static ManualTimer timer("logit_processor::apply (no copy)");
        timer.start();
        for (const auto& transformer : m_logit_transformers) {
            if (transformer->is_applicable(m_generated_tokens)) {
                transformer->apply(logits);
            }
        }
        timer.end();
    }

    void increment_gen_tokens() {
        ++m_generated_tokens;
    }

    void register_new_generated_token(int64_t new_token_id) {
        auto it = m_unique_generated_token_ids->find(new_token_id);
        if (it == m_unique_generated_token_ids->end()) {
            m_unique_generated_token_ids->insert({new_token_id, 1});
        } else {
            it->second++;
        }
    }
};
