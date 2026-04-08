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
    // When true, m_vector contains K elements in arbitrary order (heap order on the fast path,
    // nth_element order on the logprobs path) with either:
    //   - raw logits unchanged         (T == 1.0 — TemperatureLogitTransform is a no-op)
    //   - logits scaled by 1/T         (T != 1.0 — pure multiply, no max subtraction)
    // _multinomial_sample owns the max scan and fuses expf() with the CDF scan.
    bool m_defer_expf = false;

    // Set by FullVocabLogSumExpTransform when logprobs > 0 for multinomial sampling.
    // Holds log(Σ exp(raw_logit_i)) over ALL N vocabulary tokens computed from the original
    // model output, before any penalty, filter, or temperature transform.
    //
    // After sampling selects token index `idx`, _multinomial_sample reads the unchanged
    // raw logit from m_data[idx] and computes:
    //   log p_i = m_data[idx] − m_full_vocab_log_sum_exp
    //
    // NaN (default) means logprobs were not requested; sampler uses the post-filter logprob.
    float m_full_vocab_log_sum_exp = std::numeric_limits<float>::quiet_NaN();

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
            // Fast path (logprobs == 0): Temperature ran directly on m_data; copy probs to m_vector now.
            logits.initialize_vector();
        }
        // m_vector holds normalised probabilities (from Temperature via m_vector or m_data path).
        // Note: TopPFilter truncates but does NOT renormalize — the retained probabilities sum to at most top_p < 1.
        // _multinomial_sample accounts for this by scaling the random draw by total_weight (not assuming sum == 1).
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

        // Build a K-element min-heap in one sequential O(N log K) pass, regardless of whether
        // m_vector is already initialised (logprobs path) or not (fast path).
        //
        // Algorithm:
        //   1. Seed m_vector with the first K elements; build a min-heap (smallest at root).
        //   2. For each remaining element: if it beats the current min, replace the root and
        //      restore heap order (pop_heap + push_heap = two O(log K) sifts).
        //   3. m_vector ends up with the top_k values in arbitrary heap order.
        //
        // Downstream code (TemperatureLogitTransform, TopPFilter, _multinomial_sample)
        // must NOT assume m_vector[0] is the maximum — they all do their own O(K) max scan.
        static const auto min_cmp = [](const Token& a, const Token& b) {
            return a.m_log_prob > b.m_log_prob;  // inverted: makes std::*_heap a min-heap
        };

        if (logits.is_vector_initialized()) {
            // logprobs path: m_vector has N elements; run heap selection in-place.
            // Seed heap from the first K elements already in m_vector.
            std::make_heap(logits.m_vector.begin(), logits.m_vector.begin() + m_top_k, min_cmp);
            for (size_t i = m_top_k; i < logits.m_size; i++) {
                if (logits.m_vector[i].m_log_prob > logits.m_vector[0].m_log_prob) {
                    std::pop_heap(logits.m_vector.begin(), logits.m_vector.begin() + m_top_k, min_cmp);
                    logits.m_vector[m_top_k - 1] = logits.m_vector[i];
                    std::push_heap(logits.m_vector.begin(), logits.m_vector.begin() + m_top_k, min_cmp);
                }
            }
            logits.m_vector.resize(m_top_k);
        } else {
            // Fast path: m_vector not yet allocated; build heap directly from m_data to avoid
            // a full initialize_vector() alloc+write of N elements that would be mostly discarded.
            logits.m_vector.resize(m_top_k);
            for (size_t i = 0; i < m_top_k; i++)
                logits.m_vector[i] = Token(logits.m_data[i], static_cast<int64_t>(i));
            std::make_heap(logits.m_vector.begin(), logits.m_vector.end(), min_cmp);

            for (size_t i = m_top_k; i < logits.m_size; i++) {
                if (logits.m_data[i] > logits.m_vector[0].m_log_prob) {
                    std::pop_heap(logits.m_vector.begin(), logits.m_vector.end(), min_cmp);
                    logits.m_vector.back() = Token(logits.m_data[i], static_cast<int64_t>(i));
                    std::push_heap(logits.m_vector.begin(), logits.m_vector.end(), min_cmp);
                }
            }
        }
        logits.m_size = m_top_k;
    }

protected:
    size_t m_top_k = 0;
};

// Computes log(Σ exp(raw_logit_i)) over the full vocabulary and stores it in
// Logits::m_full_vocab_log_sum_exp.  Must run first, before m_data is modified or m_vector
// is created, so it captures the original model logits.
//
// After sampling selects a token index `idx`, _multinomial_sample reads m_data[idx] (always the
// original raw logit as long as downstream transforms operate on m_vector, not m_data) and computes:
//   log p_i = m_data[idx] − m_full_vocab_log_sum_exp
//
// Only inserted into the pipeline when logprobs > 0.
class FullVocabLogSumExpTransform : public ILogitTransformer {
public:
    FullVocabLogSumExpTransform() = default;

    void apply(Logits& logits) override {
        OPENVINO_ASSERT(!logits.is_vector_initialized(),
            "FullVocabLogSumExpTransform must run before any transform that modifies m_data or creates m_vector");
        const float max_logit = *std::max_element(logits.m_data, logits.m_data + logits.m_size);
        float sum = 0.0f;
        for (size_t i = 0; i < logits.m_size; ++i)
            sum += expf(logits.m_data[i] - max_logit);
        logits.m_full_vocab_log_sum_exp = logf(sum) + max_logit;
    }
};

// Copies all logits from m_data into m_vector, leaving m_data untouched.
//
// When logprobs > 0, this runs as the second transform (after FullVocabLogSumExpTransform)
// before any penalty or filtering transform.  All downstream transforms use m_vector once it
// is initialised, so m_data remains the original model logits throughout — available at
// sampling time for correct raw log-probability reporting.
class CopyLogitsToVectorTransform : public ILogitTransformer {
public:
    CopyLogitsToVectorTransform() = default;

    void apply(Logits& logits) override {
        logits.initialize_vector();
    }
};

class TemperatureLogitTransform : public ILogitTransformer {
public:
    TemperatureLogitTransform(double temperature, bool defer_expf = false)
        : m_temperature(temperature), m_defer_expf(defer_expf) {}

    void apply(Logits& logits) override {
        if (m_defer_expf) {
            // Deferred expf, only apply logits scaling at this moment
            if (m_temperature != 1.0f) {
                const float scaling_factor = 1.0f / m_temperature;
                if (logits.is_vector_initialized()) {
                    for (size_t i = 0; i < logits.m_size; i++)
                        logits.m_vector[i].m_log_prob *= scaling_factor;
                } else {
                    for (size_t i = 0; i < logits.m_size; i++)
                        logits.m_data[i] *= scaling_factor;
                }
            }
            logits.m_defer_expf = true;
        } else {
            // Standard path: compute softmax(logits / T) in place.
            // TopPFilter follows and requires normalised probabilities.
            if (logits.is_vector_initialized()) {
                // m_vector holds K raw logits in arbitrary order — heap order (fast path, logprobs == 0)
                // or nth_element order (logprobs > 0 path). Temperature does its own max scan so order
                // does not matter.
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
                    const float scaling_factor = 1.0f / m_temperature;
                    for (size_t i = 0; i < logits.m_size; i++) {
                        logits.m_vector[i].m_log_prob = expf((logits.m_vector[i].m_log_prob - max_logit) * scaling_factor);
                        norm_sum += logits.m_vector[i].m_log_prob;
                    }
                }
                for (size_t i = 0; i < logits.m_size; i++)
                    logits.m_vector[i].m_log_prob /= norm_sum;
            } else {
                // No effective top_k filtering: expf on all m_data elements directly.
                float max_logit = *std::max_element(logits.m_data, logits.m_data + logits.m_size);
                float norm_sum = 0.0f;
                if (m_temperature == 1.0f) {
                    for (size_t i = 0; i < logits.m_size; i++) {
                        logits.m_data[i] = expf(logits.m_data[i] - max_logit);
                        norm_sum += logits.m_data[i];
                    }
                } else {
                    const float scaling_factor = 1.0f / m_temperature;
                    for (size_t i = 0; i < logits.m_size; i++) {
                        logits.m_data[i] = expf((logits.m_data[i] - max_logit) * scaling_factor);
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
        // When m_vector is initialised (logprobs > 0), operate on m_vector so m_data
        // stays as original model logits.  m_vector[i].m_index == i holds for the
        // full-vocab copy created by CopyLogitsToVectorTransform, so direct index is valid.
        auto logit_ref = [&](int64_t id) -> float& {
            return logits.is_vector_initialized() ? logits.m_vector[id].m_log_prob : logits.m_data[id];
        };
        for (const auto& prompt_id : *m_unique_prompt_token_ids) {
            OPENVINO_ASSERT((prompt_id >= 0) && (prompt_id < vocab_size), "input_ids token out of bounds");
            auto& val = logit_ref(prompt_id);
            if (val >= 0) val /= m_penalty; else val *= m_penalty;
        }
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            if (1 == m_unique_prompt_token_ids->count(input_id)) {
                // repetition_penalty was already accounted by the for
                // loop above.
                continue;
            }
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            auto& val = logit_ref(input_id);
            if (val >= 0) val /= m_penalty; else val *= m_penalty;
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
        // EOS penalty runs before CopyLogitsToVectorTransform in the pipeline, so m_vector
        // is not yet initialised.  Use the is_vector_initialized() check defensively.
        // Set -inf, so stop token gets zero probability mass.
        for (auto stop_token_id: m_stop_token_ids) {
            if (logits.is_vector_initialized())
                logits.m_vector[stop_token_id].m_log_prob = -std::numeric_limits<float>::infinity();
            else
                logits.m_data[stop_token_id] = -std::numeric_limits<float>::infinity();
        }
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
        auto logit_ref = [&](int64_t id) -> float& {
            return logits.is_vector_initialized() ? logits.m_vector[id].m_log_prob : logits.m_data[id];
        };
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            auto& val = logit_ref(input_id);
            if (val >= 0) val -= m_penalty * input_id_pair.second;
            else          val += m_penalty * input_id_pair.second;
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
        auto logit_ref = [&](int64_t id) -> float& {
            return logits.is_vector_initialized() ? logits.m_vector[id].m_log_prob : logits.m_data[id];
        };
        for (const auto& input_id_pair : *m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            auto& val = logit_ref(input_id);
            if (val >= 0) val -= m_penalty;
            else          val += m_penalty;
        }
    }

    void apply(Logits& logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        apply(logits);
    }
};

} // namespace LogitTransformers
} // namespace ov::genai
