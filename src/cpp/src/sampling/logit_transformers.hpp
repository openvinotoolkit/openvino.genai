// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>

#include "openvino/genai/generation_config.hpp"

namespace ov::genai {

struct Token {
    float m_log_prob = 0.;
    int64_t m_index = 0;

    Token(float log_prob, int64_t index) : m_log_prob(log_prob), m_index(index) {}
    Token() = default;
};

struct Logits {
    float * m_data = nullptr;
    size_t m_size;
    // Late initialized for top_p or top_k transforms
    std::vector<Token> m_vector;

    Logits(float* data, size_t size): m_data(data), m_size(size) {}


    void initialize_vector() {
        OPENVINO_ASSERT(m_vector.size() == 0, "Logits vector already initialized");
        m_vector.reserve(m_size);
        for (size_t i = 0; i < m_size; i++)
            m_vector.emplace_back(m_data[i], i);
    }

    bool is_vector_initialized() const {
        return m_vector.size() > 0;
    }

    void resize(size_t new_size) {
        m_size = new_size;
        m_vector.resize(new_size);
    }
};

namespace LogitTransformers {

using TokenIds = std::vector<int64_t>;

class ILogitTransformer {
public:
    virtual void apply(Logits& logits) = 0;

    virtual bool is_applicable(size_t generated_tokens_cnt = 0) {
        return true;
    }
};

/**
 * @brief Interface for logit transformers that maintain state across token generations.
 * 
 * ILogitTransformer interface is used for logit transformers that do not maintain state across token generations.
 * accept_tokens method is used to accept a sequence of token ids, which can be used to update the internal state of the transformer.
 */
class IStatefulLogitTransformer: public ILogitTransformer {
public:
    virtual void accept_tokens(const TokenIds& input_ids) = 0;
};


class TopPFilter : public ILogitTransformer {
public:
    TopPFilter(double top_p) : m_top_p(top_p) {}

    bool partial_sort_and_resize(Logits& logits) {
        // Since most of the time huge part of logits vector contains minimal values
        // expensive sorting of entire vector might be unnecessary, especially for low values of top_p.
        // This method partially sorts vector finding M top elements and stops when top_p condition is met.
        // It iterates a few times starting with M = 16 and multiplying it by 2 each iteration until M = 1024.
        // If top_p is found in considered scope it resizes logits vector and returns true. Otherwise it returns false.
        // Note that it can we less performant than standard approach if logits value are more evenly distributed across the vector.
        for (size_t step = 16; step <= 1024; step *= 2) {
            if (logits.m_vector.size() <= step)
                break;
            std::partial_sort(logits.m_vector.begin(), logits.m_vector.begin() + step, logits.m_vector.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
            float sum = 0.0;
            for (int i = 0; i < step; i++) {
                sum += logits.m_vector[i].m_log_prob;
                if (sum > m_top_p) {
                    logits.resize(i+1);
                    return true;
                }
            }
        }
        return false;
    }

    void full_sort_and_resize(Logits& logits) {
        std::sort(logits.m_vector.begin(), logits.m_vector.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
        float probability_sum = 0.0f;
        size_t nucleus_size = 0;
        for (const auto& logit : logits.m_vector) {
            probability_sum += logit.m_log_prob;
            nucleus_size += 1;
            if (probability_sum > m_top_p) break;
        }
        logits.resize(nucleus_size);
    }

    void apply(Logits& logits) override {
        // Initialize and sort vector. Try partial sorting first and if it's not enough, sort entire vector.
        logits.initialize_vector();
        if(!partial_sort_and_resize(logits))
            full_sort_and_resize(logits);
    }

protected:
    double m_top_p = 0.f;
};

class TopKFilter : public ILogitTransformer {
public:
    TopKFilter(size_t top_k) : m_top_k(top_k) {}

    // If this transform is used along with top_p, it should be applied after it since top_p sorts entire vector and top_k does it only partially
    void apply(Logits& logits) override {

        if (m_top_k >= logits.m_size)
            return;

        // If top_p is also used vector is already initialized and sorted
        if (!logits.is_vector_initialized()) {
            // Initialize and partially sort vector
            logits.initialize_vector();
            std::partial_sort(logits.m_vector.begin(), logits.m_vector.begin() + m_top_k, logits.m_vector.end(), [](const Token& lhs, const Token& rhs) {return lhs.m_log_prob > rhs.m_log_prob; });
        }
        logits.resize(m_top_k);
    }

protected:
    size_t m_top_k = 0;
};

class TemperatureLogitTransform : public ILogitTransformer {
public:
    TemperatureLogitTransform(double temperature) : m_temperature(temperature) {};

    void apply(Logits& logits) override {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < logits.m_size; i++) {
            if (logits.m_data[i] > max_logit) {
                max_logit = logits.m_data[i];
            }
        }

        float norm_sum = 0.0;
        for (size_t i = 0; i < logits.m_size; i++) {
            logits.m_data[i] = expf((logits.m_data[i] - max_logit) / this->m_temperature);
            norm_sum += logits.m_data[i];
        }

        for (size_t i = 0; i < logits.m_size; i++) {
            logits.m_data[i] /= norm_sum;
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

    void apply(Logits& logits) override {
        size_t vocab_size = logits.m_size;
        for (const auto& prompt_id : *m_unique_prompt_token_ids) {
            OPENVINO_ASSERT((prompt_id >= 0) && (prompt_id < vocab_size), "input_ids token out of bounds");
            if (logits.m_data[prompt_id] >= 0) {
                logits.m_data[prompt_id] /= m_penalty;
            } else {
                logits.m_data[prompt_id] *= m_penalty;
            };
        }
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            if (1 == m_unique_prompt_token_ids->count(input_id)) {
                // repetition_penalty was already accounted by the for
                // loop above.
                continue;
            }
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            if (logits.m_data[input_id] >= 0) {
                logits.m_data[input_id] /= m_penalty;
            } else {
                logits.m_data[input_id] *= m_penalty;
            };
        }
    }

    void apply(Logits& logits, const TokenIds& input_ids) {
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
    EOSPenaltyTransform(const std::set<int64_t>& stop_token_ids, size_t min_generated_tokens) :
        m_stop_token_ids(stop_token_ids), m_applicable_tensor_len(min_generated_tokens) {}

    void apply(Logits& logits) override {
        // Since EOS penalty is applied early, the token vector is not initialized yet
        // and we can assume element order match token ids.
        for (auto stop_token_id: m_stop_token_ids)
            logits.m_data[stop_token_id] = 0.f;
    }


    bool is_applicable(size_t generated_tokens_cnt = 0) override {
        return generated_tokens_cnt < m_applicable_tensor_len;
    }

protected:
    size_t m_applicable_tensor_len = std::numeric_limits<size_t>::max();
    std::set<int64_t> m_stop_token_ids;
};

class FrequencyPenaltyTransform : public IPenaltyTransformer {
public:
    FrequencyPenaltyTransform(double value) {
        m_penalty = value;
    };

    void apply(Logits& logits) override {
        size_t vocab_size = logits.m_size;
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            if (logits.m_data[input_id] >= 0) {
                logits.m_data[input_id] -= m_penalty * input_id_pair.second;
            } else {
                logits.m_data[input_id] += m_penalty * input_id_pair.second;
            };
        }
    }

    void apply(Logits& logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        apply(logits);
    }
};

class PresencePenaltyTransform : public IPenaltyTransformer {
public:
    PresencePenaltyTransform(double value) {
        m_penalty = value;
    };

    void apply(Logits& logits) override {
        size_t vocab_size = logits.m_size;
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            if (logits.m_data[input_id] >= 0) {
                logits.m_data[input_id] -= m_penalty;
            } else {
                logits.m_data[input_id] += m_penalty;
            };
        }
    }

    void apply(Logits& logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        apply(logits);
    }
};

} // namespace LogitTransformers
} // namespace ov::genai
