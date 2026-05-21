// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>

#include "openvino/genai/visibility.hpp"

namespace ov::genai {

/**
 * @brief Per-step snapshot of raw logit distribution metrics.
 *
 * Computed on the original model logits BEFORE any transforms (temperature,
 * penalties, top-k/p filtering). One instance is recorded per generated token.
 */
struct OPENVINO_GENAI_EXPORTS LogitsStepStats {
    /// Shannon entropy H = -sum_i p_i * ln(p_i)  (nats). High value means the
    /// model is uncertain; low value means a peaked, confident distribution.
    float entropy = 0.0f;

    /// Probability-weighted variance of per-token surprisal:
    /// varentropy = sum_i p_i * (-ln p_i - H)^2.
    /// Complements entropy: low H + high varentropy = one clear winner (peaked);
    /// high H + low varentropy = near-uniform confusion.
    float varentropy = 0.0f;

    /// Softmax probability of the highest-logit token.
    float top1_prob = 0.0f;

    /// Log-probability of the top-1 token  (= max_logit - log_sum_exp).
    float top1_log_prob = 0.0f;

    /// Log-probability of the second-best token.
    float top2_log_prob = 0.0f;

    /// top1_log_prob - top2_log_prob. Equals the raw logit difference between
    /// rank-1 and rank-2, so no full softmax is required to compute it.
    /// Large value = decisive model; near zero = toss-up between two tokens.
    float top1_top2_log_margin = 0.0f;

    /// exp(entropy): the size of a uniform distribution with the same entropy.
    /// Easier to interpret than raw entropy: value 1 means certainty, value V
    /// means fully uniform over the vocabulary.
    float effective_vocab_size = 0.0f;

    /// Standard deviation of the raw (pre-softmax) logit values.
    /// Low value = flat logit landscape (low temperature effect);
    /// high value = sharp logit landscape.
    float logit_std = 0.0f;

    /// Sum of softmax probabilities of the top-10 tokens (nucleus concentration).
    float top10_mass = 0.0f;

    std::string to_string() const {
        std::ostringstream o;
        o << std::fixed << std::setprecision(4);
        o << "LogitsStepStats\n";
        o << "  entropy         : " << entropy              << " nats\n";
        o << "  varentropy      : " << varentropy           << "\n";
        o << "  top-1 prob      : " << top1_prob            << "\n";
        o << "  top-1 log-prob  : " << top1_log_prob        << "\n";
        o << "  top-2 log-prob  : " << top2_log_prob        << "\n";
        o << "  top1/top2 margin: " << top1_top2_log_margin << " (log)\n";
        o << "  eff vocab size  : " << std::setprecision(2) << effective_vocab_size << "\n";
        o << std::setprecision(4);
        o << "  logit std       : " << logit_std            << "\n";
        o << "  top-10 mass     : " << top10_mass           << "\n";
        return o.str();
    }
};

/**
 * @brief Running averages of LogitsStepStats over a window of generated tokens.
 *
 * Obtained from GenerationHandleImpl::get_logits_stats(). The window size is
 * controlled by GenerationConfig::logits_stats_window (0 = average all tokens).
 */
struct OPENVINO_GENAI_EXPORTS LogitsStats {
    float mean_entropy = 0.0f;
    float mean_varentropy = 0.0f;
    float mean_top1_prob = 0.0f;
    float mean_top1_top2_log_margin = 0.0f;
    float mean_effective_vocab_size = 0.0f;
    float mean_logit_std = 0.0f;
    float mean_top10_mass = 0.0f;

    /// Number of token steps included in these averages.
    size_t num_steps = 0;

    std::string to_string() const {
        std::ostringstream o;
        o << std::fixed << std::setprecision(4);
        o << "LogitsStats [" << num_steps << " step" << (num_steps != 1 ? "s" : "") << "]\n";
        o << "  entropy         : " << mean_entropy              << " nats";
        o << "  (eff vocab: " << std::setprecision(2) << mean_effective_vocab_size << ")\n";
        o << std::setprecision(4);
        o << "  varentropy      : " << mean_varentropy           << "\n";
        o << "  top-1 prob      : " << mean_top1_prob            << "\n";
        o << "  top1/top2 margin: " << mean_top1_top2_log_margin << " (log)\n";
        o << "  logit std       : " << mean_logit_std            << "\n";
        o << "  top-10 mass     : " << mean_top10_mass           << "\n";
        return o.str();
    }
};

}  // namespace ov::genai
