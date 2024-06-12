// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>

#include "generation_config.hpp"

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
    virtual std::vector<Token> apply(const std::vector<Token>& input_logits) = 0;

    void set_unique_generated_token_ids(const std::shared_ptr<std::map<int64_t, size_t>>& unique_generated_token_ids) {
        if (unique_generated_token_ids != nullptr) {
            m_unique_generated_token_ids = unique_generated_token_ids;
        } else {
            m_unique_generated_token_ids = std::shared_ptr<std::map<int64_t, size_t>>(new std::map<int64_t, size_t>);
        }
    }

    void set_unique_prompt_token_ids(const std::shared_ptr<std::set<int64_t>>& unique_prompt_token_ids) {
        if (unique_prompt_token_ids != nullptr) {
            m_unique_prompt_token_ids = unique_prompt_token_ids;
        } else {
            m_unique_prompt_token_ids = std::shared_ptr<std::set<int64_t>>(new std::set<int64_t>);
        }
    }

protected:
    double m_float_value = 0.f;
    double m_uint_value = 0;
    std::shared_ptr<std::map<int64_t, size_t>> m_unique_generated_token_ids = nullptr;
    std::shared_ptr<std::set<int64_t>> m_unique_prompt_token_ids = nullptr;

    void extract_generated_tokens(const TokenIds& input_ids) {
        set_unique_generated_token_ids(m_unique_generated_token_ids);
        set_unique_prompt_token_ids(m_unique_prompt_token_ids);

        for (const auto& input_id : input_ids) {
            if (m_unique_generated_token_ids->count(input_id)) {
                m_unique_generated_token_ids->at(input_id)++;
            } else {
                m_unique_generated_token_ids->insert({input_id, 1});
            }
        }
    }

    ILogitTransformer() = default;
};

class TopPFilter : public ILogitTransformer {
public:
    TopPFilter(double top_p) {
        m_float_value = top_p;
    }

    std::vector<Token> apply(const std::vector<Token>& input_probs) override {
        std::vector<Token> tmp(input_probs);
        std::sort(tmp.begin(), tmp.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
        float probability_sum = 0.0f;
        size_t nucleus_size = 0;
        for (const auto& probability : tmp) {
            probability_sum += probability.m_log_prob;
            nucleus_size += 1;
            if (probability_sum > m_float_value) break;
        }
        tmp.resize(nucleus_size);
        return tmp;
    }
};

class TopKFilter : public ILogitTransformer {
public:
    TopKFilter(size_t top_k) {
        m_uint_value = top_k;
    }

    std::vector<Token> apply(const std::vector<Token>& input_probs) override {
        std::vector<Token> tmp(input_probs);
        std::sort(tmp.begin(), tmp.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
        size_t top_k = input_probs.size() >= m_uint_value ? m_uint_value : input_probs.size();
        tmp.resize(top_k);
        return tmp;
    }
};

class TemperatureLogitTransform : public ILogitTransformer {
public:
    TemperatureLogitTransform(double temperature) {
        m_float_value = temperature;
    };

    std::vector<Token> apply(const std::vector<Token>& input_logits) override {
        std::vector<Token> output(input_logits.begin(), input_logits.end());
        std::sort(output.begin(), output.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
        float max_logit = output[0].m_log_prob;

        std::for_each(output.begin(), output.end(), [max_logit, this](Token& val) {val.m_log_prob = expf((val.m_log_prob - max_logit) / this->m_float_value);});

        float norm_sum = 0.0;
        for (const auto& val : output) {
            norm_sum += val.m_log_prob;
        }

        std::for_each(output.begin(), output.end(), [norm_sum](Token& val) {val.m_log_prob /= norm_sum;});
        return output;
    }
};

class RepetitionPenaltyTransform : public ILogitTransformer {
public:
    RepetitionPenaltyTransform(double value) {
        m_float_value = value;
    };

    std::vector<Token> apply(const std::vector<Token>& input_logits) override {
        std::vector<Token> output(input_logits.begin(), input_logits.end());
        size_t vocab_size = input_logits.size();
        for (const auto& prompt_id : *m_unique_prompt_token_ids) {
            OPENVINO_ASSERT((prompt_id >= 0) && (prompt_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(input_logits[prompt_id].m_index == prompt_id, "input_logits must have original index order");
            auto logit_value = output[prompt_id].m_log_prob;
            if (logit_value >= 0) {
                output[prompt_id].m_log_prob /= m_float_value;
            } else {
                output[prompt_id].m_log_prob *= m_float_value;
            };
        }
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(input_logits[input_id].m_index == input_id, "input_logits must have original index order");
            auto logit_value = output[input_id].m_log_prob;
            if (logit_value >= 0) {
                output[input_id].m_log_prob /= m_float_value;
            } else {
                output[input_id].m_log_prob *= m_float_value;
            };
        }
        return output;
    }

    std::vector<Token> apply(const std::vector<Token>& input_logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        return this->apply(input_logits);
    }
};

class EOSPenaltyTransform : public ILogitTransformer {
public:
    EOSPenaltyTransform(size_t eos_token_id) {
        m_uint_value = eos_token_id; 
    }

    std::vector<Token> apply(const std::vector<Token>& input_logits) {
        std::vector<Token> output(input_logits.begin(), input_logits.end());
        for (auto& token_id : output) {
            if (token_id.m_index == m_uint_value) {
                token_id.m_log_prob = 0.f;
            }
        }
        return output;
    }
};

class FrequencyPenaltyTransform : public ILogitTransformer {
public:
    FrequencyPenaltyTransform(double value) {
        m_float_value = value;
    };

    std::vector<Token> apply(const std::vector<Token>& input_logits) override {
        std::vector<Token> output(input_logits.begin(), input_logits.end());
        size_t vocab_size = input_logits.size();
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(input_logits[input_id].m_index == input_id, "input_logits must have original index order");
            auto logit_value = output[input_id].m_log_prob;
            if (logit_value >= 0) {
                output[input_id].m_log_prob -= m_float_value * input_id_pair.second;
            } else {
                output[input_id].m_log_prob += m_float_value * input_id_pair.second;
            };
        }
        return output;
    }

    std::vector<Token> apply(const std::vector<Token>& input_logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        return this->apply(input_logits);
    }
};

class PresencePenaltyTransform : public ILogitTransformer {
public:
    PresencePenaltyTransform(double value) {
        m_float_value = value;
    };

    std::vector<Token> apply(const std::vector<Token>& input_logits) override {
        std::vector<Token> output(input_logits.begin(), input_logits.end());
        size_t vocab_size = input_logits.size();
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(input_logits[input_id].m_index == input_id, "input_logits must have original index order");
            auto logit_value = output[input_id].m_log_prob;
            if (logit_value >= 0) {
                output[input_id].m_log_prob -= m_float_value;
            } else {
                output[input_id].m_log_prob += m_float_value;
            };
        }
        return output;
    }

    std::vector<Token> apply(const std::vector<Token>& input_logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        return this->apply(input_logits);
    }
};


class ProbabilityNormalizeTransform : public ILogitTransformer {
public:
    ProbabilityNormalizeTransform() = default;

    std::vector<Token> apply(const std::vector<Token>& input_probs) override {
        std::vector<Token> output(input_probs);
        float norm_sum = 0.0;
        for (const auto& val : output) norm_sum += val.m_log_prob;
        for (auto& val : output) val.m_log_prob /= norm_sum;
        return output;
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

        if (sampling_params.min_new_tokens > 0 && m_generated_tokens < sampling_params.min_new_tokens) {
            m_logit_transformers.emplace_back(
                new LogitTransformers::EOSPenaltyTransform(sampling_params.eos_token_id)
            );
        }

        if (sampling_params.is_multinomial() || sampling_params.is_greedy_sampling()) {
            if (sampling_params.repetition_penalty != 1.0f) {
                m_logit_transformers.emplace_back(
                    new LogitTransformers::RepetitionPenaltyTransform(sampling_params.repetition_penalty)
                );
            }
            if (sampling_params.presence_penalty != 0.0f) {
                m_logit_transformers.emplace_back(new LogitTransformers::PresencePenaltyTransform(sampling_params.presence_penalty));
                
            }
            if (sampling_params.frequence_penalty != 0.0f) {
                m_logit_transformers.emplace_back(
                    new LogitTransformers::FrequencyPenaltyTransform(sampling_params.frequence_penalty)
                );
            }

            if (sampling_params.is_multinomial()) {
                m_logit_transformers.emplace_back(new LogitTransformers::TemperatureLogitTransform(sampling_params.temperature));
                if (sampling_params.top_p != 0.0f) {
                    m_logit_transformers.emplace_back(new LogitTransformers::TopPFilter(sampling_params.top_p));
                }
                if (sampling_params.top_k > 0) {
                    m_logit_transformers.emplace_back(new LogitTransformers::TopKFilter(sampling_params.top_k));
                }
                m_logit_transformers.emplace_back(new LogitTransformers::ProbabilityNormalizeTransform());
            }
        }
        for (const auto& transformer : m_logit_transformers) {
            transformer->set_unique_prompt_token_ids(m_unique_prompt_token_ids);
            transformer->set_unique_generated_token_ids(m_unique_generated_token_ids);
        }
    }

    std::vector<Token> apply(const std::vector<Token>& logits) {
        std::vector<Token> outputs(logits.begin(), logits.end());
        for (const auto& transformer : m_logit_transformers) {
            outputs = transformer->apply(outputs);
        }
        return outputs;
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
