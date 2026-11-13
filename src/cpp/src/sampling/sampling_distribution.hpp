// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "sampling/logit_transformers.hpp"

namespace ov::genai {

// The distribution multinomial sampling draws a token from, as the logit processor leaves the logits: over the
// candidates in m_vector when a transform selected them, otherwise over the whole row in m_data. A candidate's weight
// is exp(logit - max logit) when expf is deferred and its probability otherwise, and it is drawn with probability
// weight / total. The total is summed in double, since top-p or min-p without top-k can keep most of the vocabulary
// and a float sum drops weights smaller than half an ulp of the sum so far. A row without positive finite mass, e.g.
// with every candidate masked, is drawn uniformly.
//
// Speculative sampling is exact only if the q(.) a draft records is the distribution it drew from, so drawing,
// recording and verifying all read the distribution from this class.
class SamplingDistribution {
public:
    explicit SamplingDistribution(const Logits& logits);
    // The distribution reads the candidates of the logits, which must outlive it.
    explicit SamplingDistribution(Logits&&) = delete;

    size_t size() const {
        return m_size;
    }

    // Whether the candidates are the tokens the logit processor selected, rather than the whole row.
    bool is_sparse() const {
        return m_candidates != nullptr;
    }

    int64_t token_id(size_t index) const {
        return is_sparse() ? m_candidates[index].m_index : static_cast<int64_t>(index);
    }

    float probability(size_t index) const {
        return static_cast<float>(weight(index) / m_total);
    }

    // Index of token_id among the candidates; none for a token the distribution cannot draw, including ids beyond
    // the row.
    std::optional<size_t> find(int64_t token_id) const;

    // 0 for a token the distribution cannot draw.
    float probability_of(int64_t token_id) const;

    // The log-probability the sampler reports for a candidate: of the raw logits over the whole vocabulary when
    // logprobs are requested, otherwise of this distribution.
    float log_probability(size_t index) const;

    // Index of a candidate drawn with probability weight / total.
    size_t draw(std::mt19937& rng_engine) const;

    Token sample(std::mt19937& rng_engine) const {
        const size_t index = draw(rng_engine);
        return Token(log_probability(index), token_id(index));
    }

private:
    float weight(size_t index) const;

    // Calls function with the weight of a candidate as a function of its index, chosen once for the whole row.
    template <typename Function>
    decltype(auto) with_weights(Function&& function) const;

    const float* m_data;
    const Token* m_candidates = nullptr;
    size_t m_size;
    // exp(logit - max logit) of the candidates when expf is deferred, otherwise empty.
    std::vector<float> m_exp_weights;
    double m_total = 0.0;
    // Subtracted from a candidate's stored logit (deferred expf) or from the log of its weight to report its
    // log-probability.
    float m_log_normalizer = 0.0f;
    float m_full_vocab_log_sum_exp;
    bool m_is_uniform = false;
};

}  // namespace ov::genai
