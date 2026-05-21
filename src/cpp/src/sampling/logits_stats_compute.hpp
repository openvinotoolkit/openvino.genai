// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "openvino/genai/logits_stats.hpp"
#include "sampling/logit_transformers.hpp"

namespace ov::genai {

static constexpr size_t LOGITS_STATS_TOP_K = 10;

/**
 * @brief Compute per-step logit distribution metrics from raw (pre-transform) logits.
 *
 * All metrics are computed in two O(V) passes:
 *   Pass 1 – find max_logit, second_logit (for margin), logit mean/variance.
 *   Pass 2 – compute log_sum_exp, then p_i, entropy, varentropy in one loop.
 *             Reuses the prob vector for the top-K partial sort.
 */
inline LogitsStepStats compute_logits_step_stats(const Logits& logits) {
    LogitsStepStats stats;

    const float* data = logits.m_data;
    const size_t V = logits.m_size;

    if (V == 0)
        return stats;

    // --- Pass 1: max, second-max, logit mean and variance ----------------------
    float max_logit = data[0];
    float second_logit = -std::numeric_limits<float>::infinity();
    float logit_sum = data[0];

    for (size_t i = 1; i < V; ++i) {
        const float l = data[i];
        if (l > max_logit) {
            second_logit = max_logit;
            max_logit = l;
        } else if (l > second_logit) {
            second_logit = l;
        }
        logit_sum += l;
    }

    const float logit_mean = logit_sum / static_cast<float>(V);
    float logit_sq_dev = 0.0f;
    for (size_t i = 0; i < V; ++i) {
        const float d = data[i] - logit_mean;
        logit_sq_dev += d * d;
    }
    stats.logit_std = std::sqrt(logit_sq_dev / static_cast<float>(V));

    // top1 vs top2 margin is the raw logit difference (softmax-invariant)
    stats.top1_top2_log_margin = (V >= 2) ? (max_logit - second_logit) : 0.0f;

    // --- Pass 2: log_sum_exp, probabilities, entropy, varentropy ---------------
    // Subtract max for numerical stability.
    float sum_exp = 0.0f;
    for (size_t i = 0; i < V; ++i)
        sum_exp += std::exp(data[i] - max_logit);

    const float log_sum_exp = max_logit + std::log(sum_exp);

    // Compute probs, entropy, and collect probs for top-K in one loop.
    std::vector<float> probs(V);
    float entropy = 0.0f;
    for (size_t i = 0; i < V; ++i) {
        const float log_p = data[i] - log_sum_exp;  // log p_i (negative)
        probs[i] = std::exp(log_p);                  // p_i
        entropy -= probs[i] * log_p;                 // += p_i * (-log p_i)
    }

    // Varentropy: sum_i p_i * (surprisal_i - H)^2
    float varentropy = 0.0f;
    for (size_t i = 0; i < V; ++i) {
        const float surprisal = log_sum_exp - data[i];  // -log p_i
        const float diff = surprisal - entropy;
        varentropy += probs[i] * diff * diff;
    }

    stats.entropy = entropy;
    stats.varentropy = varentropy;
    stats.effective_vocab_size = std::exp(entropy);

    stats.top1_log_prob = max_logit - log_sum_exp;
    stats.top2_log_prob = (V >= 2) ? (second_logit - log_sum_exp) : stats.top1_log_prob;
    stats.top1_prob = std::exp(stats.top1_log_prob);

    // --- Top-K mass: partial sort via nth_element (O(V) amortised) ------------
    const size_t K = std::min(LOGITS_STATS_TOP_K, V);
    // After nth_element the K largest probs reside in probs[V-K .. V-1].
    std::nth_element(probs.begin(), probs.begin() + static_cast<std::ptrdiff_t>(V - K), probs.end());
    float top_k_mass = 0.0f;
    for (size_t i = V - K; i < V; ++i)
        top_k_mass += probs[i];
    stats.top10_mass = top_k_mass;

    return stats;
}

}  // namespace ov::genai
