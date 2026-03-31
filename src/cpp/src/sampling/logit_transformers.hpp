// Copyright (C) 2025-2026 Intel Corporation
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
    // Set by TemperatureLogitTransform when top_k > 0 AND top_p == 1.0.
    // When true, m_vector contains K elements (heap order) with either:
    //   - raw logits unchanged         (T == 1.0 — TemperatureLogitTransform is a no-op)
    //   - logits scaled by 1/T         (T != 1.0 — pure multiply, no max subtraction)
    // _multinomial_sample owns the max scan and fuses expf() with the CDF scan.
    // NOT set for full-vocab (top_k == 0): CDF scan in vocab-index order has no early-exit
    // benefit and costs ~1.5N expf vs N expf for the standard normalised path.
    bool m_defer_expf = false;

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
        if (!logits.is_vector_initialized()) {
            // Temperature ran on m_data (full-vocab path): copy normalised probs into m_vector.
            logits.initialize_vector();
        }
        // m_vector already populated by TopKFilter+Temperature (K-element prob path) or
        // just initialized above. Either way it holds normalised probabilities.
        if (!partial_sort_and_resize(logits))
            full_sort_and_resize(logits);
    }

protected:
    double m_top_p = 0.f;
};

class TopKFilter : public ILogitTransformer {
public:
    TopKFilter(size_t top_k) : m_top_k(top_k) {}

    void apply(Logits& logits) override {

        if (m_top_k >= logits.m_size)
            return;

        // Build a K-element min-heap directly on m_data in one sequential O(N log K) pass.
        //
        // Why not initialize_vector() + partial_sort:
        //   initialize_vector() allocates and writes N × 12 bytes (1.8 MB for vocab=151936)
        //   every token step, most of which is immediately discarded.  The heap never exceeds
        //   K × 12 bytes (480 bytes for K=40) — it stays in L1 cache the entire scan.
        //
        // Algorithm:
        //   1. Seed m_vector with the first K elements; build a min-heap (smallest at root).
        //   2. For each remaining element: if it beats the current min replace the root and
        //      restore heap order (pop_heap + push_heap = two O(log K) sifts).
        //   3. m_vector ends up with the top-K values in arbitrary heap order.
        //
        // Downstream code (TemperatureLogitTransform, TopPFilter, _multinomial_sample)
        // must NOT assume m_vector[0] is the maximum — they all do their own O(K) max scan.
        const size_t K = m_top_k;
        static const auto min_cmp = [](const Token& a, const Token& b) {
            return a.m_log_prob > b.m_log_prob;  // inverted: makes std::*_heap a min-heap
        };

        logits.m_vector.resize(K);
        for (size_t i = 0; i < K; i++)
            logits.m_vector[i] = Token(logits.m_data[i], static_cast<int64_t>(i));
        std::make_heap(logits.m_vector.begin(), logits.m_vector.end(), min_cmp);

        for (size_t i = K; i < logits.m_size; i++) {
            if (logits.m_data[i] > logits.m_vector[0].m_log_prob) {
                std::pop_heap(logits.m_vector.begin(), logits.m_vector.end(), min_cmp);
                logits.m_vector.back() = Token(logits.m_data[i], static_cast<int64_t>(i));
                std::push_heap(logits.m_vector.begin(), logits.m_vector.end(), min_cmp);
            }
        }
        logits.m_size = K;  // m_vector is already size K
    }

protected:
    size_t m_top_k = 0;
};

class TemperatureLogitTransform : public ILogitTransformer {
public:
    // defer_expf=true: TopPFilter will NOT run — we only scale logits here and let
    // _multinomial_sample fuse expf() with the CDF scan (llama.cpp-style).
    // defer_expf=false: TopPFilter follows and needs normalised probabilities, so we
    // compute expf() + normalise as before.
    TemperatureLogitTransform(double temperature, bool defer_expf = false)
        : m_temperature(temperature), m_defer_expf(defer_expf) {}

    void apply(Logits& logits) override {
        if (m_defer_expf) {
            // Deferred path (mirrors llama.cpp llama_sampler_temp_impl):
            // For T=1 do nothing — _multinomial_sample receives raw logits.
            // For T!=1 scale each logit by 1/T (pure multiply, no max subtraction).
            // The max-stabilisation scan is done inside _multinomial_sample where it is
            // fused with the expf + CDF scan.  A plain multiply loop is trivially
            // auto-vectorised with AVX2; the previous subtract+multiply was too, but
            // this also makes T=1 a genuine zero-work fast path.
            if (m_temperature != 1.0f) {
                const float inv_T = 1.0f / m_temperature;
                if (logits.is_vector_initialized()) {
                    for (size_t i = 0; i < logits.m_size; i++)
                        logits.m_vector[i].m_log_prob *= inv_T;
                } else {
                    for (size_t i = 0; i < logits.m_size; i++)
                        logits.m_data[i] *= inv_T;
                }
            }
            logits.m_defer_expf = true;
        } else {
            // Standard path: compute softmax(logits / T) in place.
            // TopPFilter follows and requires normalised probabilities.
            if (logits.is_vector_initialized()) {
                // TopKFilter ran: m_vector holds K raw logits in arbitrary order
                // TopKFilter leaves m_vector in arbitrary heap order. Scan K elements for max — O(K), trivial.
                float max_logit = logits.m_vector[0].m_log_prob;
                for (size_t i = 1; i < logits.m_size; i++)
                    if (logits.m_vector[i].m_log_prob > max_logit)
                        max_logit = logits.m_vector[i].m_log_prob;
                float norm_sum = 0.0f;
                if (m_temperature == 1.0f) {
                    for (size_t i = 0; i < logits.m_size; i++) {
                        logits.m_vector[i].m_log_prob = expf(logits.m_vector[i].m_log_prob - max_logit);
                        norm_sum += logits.m_vector[i].m_log_prob;
                    }
                } else {
                    const float inv_T = 1.0f / m_temperature;
                    for (size_t i = 0; i < logits.m_size; i++) {
                        logits.m_vector[i].m_log_prob = expf((logits.m_vector[i].m_log_prob - max_logit) * inv_T);
                        norm_sum += logits.m_vector[i].m_log_prob;
                    }
                }
                for (size_t i = 0; i < logits.m_size; i++)
                    logits.m_vector[i].m_log_prob /= norm_sum;
            } else {
                // Full-vocab path: expf on all m_data elements.
                float max_logit = *std::max_element(logits.m_data, logits.m_data + logits.m_size);
                float norm_sum = 0.0f;
                if (m_temperature == 1.0f) {
                    for (size_t i = 0; i < logits.m_size; i++) {
                        logits.m_data[i] = expf(logits.m_data[i] - max_logit);
                        norm_sum += logits.m_data[i];
                    }
                } else {
                    const float inv_T = 1.0f / m_temperature;
                    for (size_t i = 0; i < logits.m_size; i++) {
                        logits.m_data[i] = expf((logits.m_data[i] - max_logit) * inv_T);
                        norm_sum += logits.m_data[i];
                    }
                }
                // Normalization required for TopPFilter correctness (cumulative prob comparison).
                for (size_t i = 0; i < logits.m_size; i++)
                    logits.m_data[i] /= norm_sum;
            }
        }
    }

protected:
    float m_temperature = 0.f;
    bool  m_defer_expf  = false;
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
