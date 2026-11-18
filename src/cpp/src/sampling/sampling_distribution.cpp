// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sampling/sampling_distribution.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>

#include "logger.hpp"
#include "openvino/core/except.hpp"

namespace ov::genai {

template <typename Function>
decltype(auto) SamplingDistribution::with_weights(Function&& function) const {
    if (m_is_uniform) {
        return function([](size_t) {
            return 1.0f;
        });
    }
    if (!m_exp_weights.empty()) {
        return function([this](size_t index) {
            return m_exp_weights[index];
        });
    }
    if (is_sparse()) {
        return function([this](size_t index) {
            return m_candidates[index].m_log_prob;
        });
    }
    return function([this](size_t index) {
        return m_data[index];
    });
}

SamplingDistribution::SamplingDistribution(const Logits& logits)
    : m_data(logits.m_data),
      m_candidates(logits.is_vector_initialized() ? logits.m_vector.data() : nullptr),
      m_size(logits.m_size),
      m_full_vocab_log_sum_exp(logits.m_full_vocab_log_sum_exp) {
    OPENVINO_ASSERT(m_size > 0, "Sampling logits have no candidate tokens.");
    float max_logit = 0.0f;
    if (logits.m_defer_expf) {
        OPENVINO_ASSERT(is_sparse(),
                        "Internal error: m_defer_expf=true but m_vector not initialized. "
                        "defer_expf requires top_k > 0 which always populates m_vector via TopKFilter.");
        // m_vector holds K logits in arbitrary heap order, so scan for the max.
        max_logit = m_candidates[0].m_log_prob;
        for (size_t i = 1; i < m_size; ++i) {
            max_logit = std::max(max_logit, m_candidates[i].m_log_prob);
        }
        m_exp_weights.resize(m_size);
        for (size_t i = 0; i < m_size; ++i) {
            m_exp_weights[i] = expf(m_candidates[i].m_log_prob - max_logit);
        }
    }
    m_total = with_weights([this](auto weight_of) {
        double total = 0.0;
        for (size_t i = 0; i < m_size; ++i) {
            total += weight_of(i);
        }
        return total;
    });
    // Should not occur, since the logit processor always leaves a token with mass, but NaN logits, e.g. after an
    // overflow, can still remove it. Generation then goes on with uniform draws rather than aborting.
    m_is_uniform = !(m_total > 0.0 && std::isfinite(m_total));
    if (m_is_uniform) {
        m_total = static_cast<double>(m_size);
        static std::once_flag warned;
        std::call_once(warned, [] {
            GENAI_WARN("Sampling logits without positive finite probability mass, e.g. NaN logits, are sampled "
                       "uniformly. A speculative decoding draft with such logits gets almost no candidates accepted. "
                       "This warning is shown once.");
        });
    }
    // log(sum of exp(logit)) is log(total) + max logit when expf is deferred.
    m_log_normalizer = static_cast<float>(std::log(m_total)) + (m_is_uniform ? 0.0f : max_logit);
}

float SamplingDistribution::weight(size_t index) const {
    return with_weights([index](auto weight_of) {
        return weight_of(index);
    });
}

std::optional<size_t> SamplingDistribution::find(int64_t token_id) const {
    if (!is_sparse()) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= m_size) {
            return std::nullopt;
        }
        return static_cast<size_t>(token_id);
    }
    for (size_t i = 0; i < m_size; ++i) {
        if (m_candidates[i].m_index == token_id) {
            return i;
        }
    }
    return std::nullopt;
}

float SamplingDistribution::probability_of(int64_t token_id) const {
    const std::optional<size_t> index = find(token_id);
    return index ? probability(*index) : 0.0f;
}

float SamplingDistribution::log_probability(size_t index) const {
    if (m_is_uniform) {
        return std::log(1.0f / static_cast<float>(m_size));
    }
    if (!std::isnan(m_full_vocab_log_sum_exp)) {
        // The transforms left m_data as the raw model logits and wrote to m_vector.
        const int64_t token = token_id(index);
        OPENVINO_ASSERT(token >= 0, "Sampled token ID must be non-negative.");
        return m_data[token] - m_full_vocab_log_sum_exp;
    }
    if (!m_exp_weights.empty()) {
        return m_candidates[index].m_log_prob - m_log_normalizer;
    }
    return std::log(weight(index)) - m_log_normalizer;
}

size_t SamplingDistribution::draw(std::mt19937& rng_engine) const {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double threshold = m_total * uniform(rng_engine);
    return with_weights([this, threshold](auto weight_of) {
        // The draw can round up to the total, so the scan ends on the last candidate with mass.
        double cumulative = 0.0;
        size_t selected = 0;
        for (size_t i = 0; i < m_size; ++i) {
            const float weight = weight_of(i);
            if (!(weight > 0.0f)) {
                continue;
            }
            selected = i;
            cumulative += weight;
            if (cumulative > threshold) {
                break;
            }
        }
        return selected;
    });
}

}  // namespace ov::genai
