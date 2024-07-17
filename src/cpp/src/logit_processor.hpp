// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>

#include "openvino/genai/generation_config.hpp"

struct Token {
    float m_log_prob = 0.;
    int64_t m_index = 0;

    Token(float log_prob, int64_t index) : m_log_prob(log_prob), m_index(index) {}
    Token() = default;
};

namespace LogitTransformers {
using TokenIds = std::vector<int64_t>;

class ILogitTransformer {
public:
    virtual void apply(std::vector<Token>& logits) = 0;

    virtual bool is_applicable(size_t generated_tokens_cnt = 0) {
        return true;
    }
};

class TopPFilter : public ILogitTransformer {
public:
    TopPFilter(double top_p) : m_top_p(top_p) {}

    void apply(std::vector<Token>& logits) override {
        std::sort(logits.begin(), logits.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
        float probability_sum = 0.0f;
        size_t nucleus_size = 0;
        for (const auto& probability : logits) {
            probability_sum += probability.m_log_prob;
            nucleus_size += 1;
            if (probability_sum > m_top_p) break;
        }
        logits.resize(nucleus_size);
    }

protected:
    double m_top_p = 0.f;
};

class TopKFilter : public ILogitTransformer {
public:
    TopKFilter(size_t top_k) : m_top_k(top_k) {}

    void apply(std::vector<Token>& logits) override {
        std::sort(logits.begin(), logits.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
        size_t top_k = logits.size() >= m_top_k ? m_top_k : logits.size();
        logits.resize(top_k);
    }

protected:
    size_t m_top_k = 0;
};

class TemperatureLogitTransform : public ILogitTransformer {
public:
    TemperatureLogitTransform(double temperature) : m_temperature(temperature) {};

    void apply(std::vector<Token>& logits) override {
        auto max_prob_token = std::max_element(logits.begin(), logits.end(), [](const Token& lhs, const Token& rhs) { return lhs.m_log_prob < rhs.m_log_prob; });
        float max_logit = max_prob_token->m_log_prob;

        std::for_each(logits.begin(), logits.end(), [max_logit, this](Token& val) {val.m_log_prob = expf((val.m_log_prob - max_logit) / this->m_temperature);});

        float norm_sum = 0.0;
        for (const auto& val : logits) {
            norm_sum += val.m_log_prob;
        }

        std::for_each(logits.begin(), logits.end(), [norm_sum](Token& val) {val.m_log_prob /= norm_sum;});
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

    void apply(std::vector<Token>& logits) override {
        size_t vocab_size = logits.size();
        for (const auto& prompt_id : *m_unique_prompt_token_ids) {
            OPENVINO_ASSERT((prompt_id >= 0) && (prompt_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(logits[prompt_id].m_index == prompt_id, "input_logits must have original index order");
            auto logit_value = logits[prompt_id].m_log_prob;
            if (logit_value >= 0) {
                logits[prompt_id].m_log_prob /= m_penalty;
            } else {
                logits[prompt_id].m_log_prob *= m_penalty;
            };
        }
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(logits[input_id].m_index == input_id, "input_logits must have original index order");
            auto logit_value = logits[input_id].m_log_prob;
            if (logit_value >= 0) {
                logits[input_id].m_log_prob /= m_penalty;
            } else {
                logits[input_id].m_log_prob *= m_penalty;
            };
        }
    }

    void apply(std::vector<Token>& logits, const TokenIds& input_ids) {
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

    void apply(std::vector<Token>& logits) override {
        // Since EOS penalty is applied early, the token vector is not sorted
        // and we can assume element order match token ids.
        logits[m_eos_token_id].m_log_prob = 0.f;
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

    void apply(std::vector<Token>& logits) override {
        size_t vocab_size = logits.size();
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(logits[input_id].m_index == input_id, "input_logits must have original index order");
            auto logit_value = logits[input_id].m_log_prob;
            if (logit_value >= 0) {
                logits[input_id].m_log_prob -= m_penalty * input_id_pair.second;
            } else {
                logits[input_id].m_log_prob += m_penalty * input_id_pair.second;
            };
        }
    }

    void apply(std::vector<Token>& logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        apply(logits);
    }
};

class PresencePenaltyTransform : public IPenaltyTransformer {
public:
    PresencePenaltyTransform(double value) {
        m_penalty = value;
    };

    void apply(std::vector<Token>& logits) override {
        size_t vocab_size = logits.size();
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(logits[input_id].m_index == input_id, "input_logits must have original index order");
            auto logit_value = logits[input_id].m_log_prob;
            if (logit_value >= 0) {
                logits[input_id].m_log_prob -= m_penalty;
            } else {
                logits[input_id].m_log_prob += m_penalty;
            };
        }
    }

    void apply(std::vector<Token>& logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        apply(logits);
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
    LogitProcessor(const ov::genai::GenerationConfig& sampling_params,
                   const LogitTransformers::TokenIds& input_ids) {
        for (const auto& input_id : input_ids) {
            m_unique_prompt_token_ids->insert(input_id);
        }

        if (sampling_params.min_new_tokens > 0) {
            m_logit_transformers.emplace_back(
                new LogitTransformers::EOSPenaltyTransform(sampling_params.eos_token_id, sampling_params.min_new_tokens)
            );
        }

        if (sampling_params.is_multinomial() || sampling_params.is_greedy_decoding()) {
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
            if (sampling_params.frequency_penalty != 0.0f) {
                std::shared_ptr<LogitTransformers::FrequencyPenaltyTransform> transformer = 
                    std::shared_ptr<LogitTransformers::FrequencyPenaltyTransform>(new LogitTransformers::FrequencyPenaltyTransform(sampling_params.frequency_penalty));
                transformer->set_unique_generated_token_ids(m_unique_generated_token_ids);
                m_logit_transformers.push_back(transformer);
            }

            if (sampling_params.is_multinomial()) {
                m_logit_transformers.emplace_back(new LogitTransformers::TemperatureLogitTransform(sampling_params.temperature));
                if (sampling_params.top_p != 1.0f) {
                    m_logit_transformers.emplace_back(new LogitTransformers::TopPFilter(sampling_params.top_p));
                }
                if (sampling_params.top_k > 0) {
                    m_logit_transformers.emplace_back(new LogitTransformers::TopKFilter(sampling_params.top_k));
                }
            }
        }
    }

    void apply(std::vector<Token>& logits) {
        for (const auto& transformer : m_logit_transformers) {
            if (transformer->is_applicable(m_generated_tokens)) {
                transformer->apply(logits);
            }
        }
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
