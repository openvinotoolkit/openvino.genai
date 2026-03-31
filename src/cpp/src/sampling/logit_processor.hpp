// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>

#include "openvino/genai/generation_config.hpp"
#include "sampling/logit_transformers.hpp"
#include "sampling/structured_output/structured_output_controller.hpp"

namespace ov::genai {
// Per-token timing breakdown for LogitProcessor::apply_timed().
struct LogitProcessorTimings {
    float misc_us       = 0.f;  // EOS penalty + structured output transforms
    float penalties_us  = 0.f;  // repetition / presence / frequency penalties
    float temperature_us = 0.f; // TemperatureLogitTransform  (expf loop, O(vocab))
    float top_p_us      = 0.f;  // TopPFilter  (sort + prefix scan)
    float top_k_us      = 0.f;  // TopKFilter  (partial sort)
};

class LogitProcessor {
protected:
    // Transforms stored in separate groups so apply_timed() can time each independently
    // without dynamic_cast.  The flat m_logit_transformers vector is kept for the
    // untimed apply() path (and for is_applicable gating on EOS/structured output).
    std::vector<std::shared_ptr<LogitTransformers::ILogitTransformer>> m_logit_transformers;
    std::vector<std::shared_ptr<LogitTransformers::IStatefulLogitTransformer>> m_stateful_logit_transformers;

    // Named sub-groups for timed apply — pointers into m_logit_transformers elements
    std::vector<std::shared_ptr<LogitTransformers::ILogitTransformer>> m_misc_transformers;      // EOS, structured-output
    std::vector<std::shared_ptr<LogitTransformers::ILogitTransformer>> m_penalty_transformers;   // rep/freq/presence
    std::shared_ptr<LogitTransformers::ILogitTransformer> m_temperature_transformer;             // nullptr if greedy
    std::shared_ptr<LogitTransformers::ILogitTransformer> m_top_p_transformer;                  // nullptr if top_p==1
    std::shared_ptr<LogitTransformers::ILogitTransformer> m_top_k_transformer;                  // nullptr if top_k==0

    std::shared_ptr<std::map<int64_t, size_t>> m_unique_generated_token_ids = std::shared_ptr<std::map<int64_t, size_t>>(new std::map<int64_t, size_t>);
    std::shared_ptr<std::set<int64_t>> m_unique_prompt_token_ids = std::shared_ptr<std::set<int64_t>>(new std::set<int64_t>);
    size_t m_generated_tokens = 0;

    // speculative decoding parameters
    float m_assistant_confidence_threshold = 0.f;


public:
    LogitProcessor(const ov::genai::GenerationConfig& sampling_params,
                   const LogitTransformers::TokenIds& input_ids,
                   std::shared_ptr<ov::genai::StructuredOutputController> structured_output_controller = nullptr
    ) {
        for (const auto& input_id : input_ids) {
            m_unique_prompt_token_ids->insert(input_id);
        }

        if (sampling_params.min_new_tokens > 0) {
            auto t = std::shared_ptr<LogitTransformers::ILogitTransformer>(
                new LogitTransformers::EOSPenaltyTransform(sampling_params.stop_token_ids, sampling_params.min_new_tokens));
            m_logit_transformers.push_back(t);
            m_misc_transformers.push_back(t);
        }

        OPENVINO_ASSERT(structured_output_controller != nullptr || !sampling_params.is_structured_output_generation(), "Structured output controller is not set for structured output generation");
        if (sampling_params.is_structured_output_generation() && structured_output_controller != nullptr) {
            auto transformer = structured_output_controller->get_logits_transformer(sampling_params);
            m_logit_transformers.push_back(transformer);
            m_misc_transformers.push_back(transformer);
            m_stateful_logit_transformers.emplace_back(std::dynamic_pointer_cast<LogitTransformers::IStatefulLogitTransformer>(transformer));
        }

        if (sampling_params.is_multinomial() || sampling_params.is_greedy_decoding()) {
            if (sampling_params.repetition_penalty != 1.0f) {
                std::shared_ptr<LogitTransformers::RepetitionPenaltyTransform> transformer =
                    std::shared_ptr<LogitTransformers::RepetitionPenaltyTransform>(new LogitTransformers::RepetitionPenaltyTransform(sampling_params.repetition_penalty));
                transformer->set_unique_prompt_token_ids(m_unique_prompt_token_ids);
                transformer->set_unique_generated_token_ids(m_unique_generated_token_ids);
                m_logit_transformers.push_back(transformer);
                m_penalty_transformers.push_back(transformer);
            }
            if (sampling_params.presence_penalty != 0.0f) {
                std::shared_ptr<LogitTransformers::PresencePenaltyTransform> transformer = 
                    std::shared_ptr<LogitTransformers::PresencePenaltyTransform>(new LogitTransformers::PresencePenaltyTransform(sampling_params.presence_penalty)); 
                transformer->set_unique_generated_token_ids(m_unique_generated_token_ids);
                m_logit_transformers.push_back(transformer);
                m_penalty_transformers.push_back(transformer);
            }
            if (sampling_params.frequency_penalty != 0.0f) {
                std::shared_ptr<LogitTransformers::FrequencyPenaltyTransform> transformer = 
                    std::shared_ptr<LogitTransformers::FrequencyPenaltyTransform>(new LogitTransformers::FrequencyPenaltyTransform(sampling_params.frequency_penalty));
                transformer->set_unique_generated_token_ids(m_unique_generated_token_ids);
                m_logit_transformers.push_back(transformer);
                m_penalty_transformers.push_back(transformer);
            }

            if (sampling_params.is_multinomial()) {
                // Order: TopK → Temperature → TopP
                //
                // TopK first: exp() is monotone, so ranking on raw logits == ranking on
                // softmax(logits/T). Filtering to K candidates *before* temperature means
                // Temperature only calls expf() on K elements, not the full vocab (e.g. 50
                // vs 151 936 for top_k=50 — a ~3000× reduction in expf calls).
                //
                // Temperature: if TopK ran, operates on K-element m_vector; otherwise full m_data.
                //
                // TopP: by the time it runs, m_vector holds normalised probabilities (either
                // K-element from TopK+Temp, or full-vocab from Temp alone), so it just
                // sorts and finds the nucleus without needing to call initialize_vector().
                if (sampling_params.top_k > 0 && sampling_params.top_k < std::numeric_limits<size_t>::max()) {
                    m_top_k_transformer = std::shared_ptr<LogitTransformers::ILogitTransformer>(
                        new LogitTransformers::TopKFilter(sampling_params.top_k));
                    m_logit_transformers.push_back(m_top_k_transformer);
                }
                // Defer expf to the draw step (fused CDF scan) only when BOTH:
                //   1. top_k > 0: TopKFilter will produce a K-element m_vector sorted
                //      in heap order — the CDF early-exit scan is effective because
                //      the K candidates are compact and adjacent in memory.
                //   2. top_p == 1.0: TopPFilter will NOT run (needs normalised probs).
                //
                // Full-vocab deferred path (top_k == 0, top_p == 1.0) is NOT used because
                // the CDF scan iterates m_data in vocabulary-index order (unsorted), so the
                // early-exit provides no benefit — on average N/2 expf calls, making total
                // cost ~1.5N expf vs N expf for the standard normalised path.
                const bool top_k_active = (sampling_params.top_k > 0 &&
                                           sampling_params.top_k < std::numeric_limits<size_t>::max());
                const bool defer_expf = top_k_active && (sampling_params.top_p == 1.0f);
                m_temperature_transformer = std::shared_ptr<LogitTransformers::ILogitTransformer>(
                    new LogitTransformers::TemperatureLogitTransform(sampling_params.temperature, defer_expf));
                m_logit_transformers.push_back(m_temperature_transformer);
                if (sampling_params.top_p != 1.0f) {
                    m_top_p_transformer = std::shared_ptr<LogitTransformers::ILogitTransformer>(
                        new LogitTransformers::TopPFilter(sampling_params.top_p));
                    m_logit_transformers.push_back(m_top_p_transformer);
                }
            }
            if (sampling_params.assistant_confidence_threshold > 0) {
                m_assistant_confidence_threshold = sampling_params.assistant_confidence_threshold;
            }
        }
    }

    float get_assistant_confidence_threshold() {
        return m_assistant_confidence_threshold;
    }

    void apply(Logits& logits) {
        for (const auto& transformer : m_logit_transformers) {
            if (transformer->is_applicable(m_generated_tokens)) {
                transformer->apply(logits);
            }
        }
    }

    // Timed variant — fills t with wall-clock microseconds per transform group.
    void apply_timed(Logits& logits, LogitProcessorTimings& t) {
        auto time_group = [&](std::vector<std::shared_ptr<LogitTransformers::ILogitTransformer>>& group, float& acc) {
            if (group.empty()) return;
            const auto t0 = std::chrono::steady_clock::now();
            for (const auto& tr : group)
                if (tr->is_applicable(m_generated_tokens))
                    tr->apply(logits);
            acc += static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count());
        };
        auto time_single = [&](std::shared_ptr<LogitTransformers::ILogitTransformer>& tr, float& acc) {
            if (!tr) return;
            const auto t0 = std::chrono::steady_clock::now();
            tr->apply(logits);
            acc += static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count());
        };
        time_group(m_misc_transformers,       t.misc_us);
        time_group(m_penalty_transformers,    t.penalties_us);
        time_single(m_top_k_transformer,      t.top_k_us);      // before temperature
        time_single(m_temperature_transformer, t.temperature_us); // K elements if TopK ran
        time_single(m_top_p_transformer,       t.top_p_us);      // after temperature
    }

    void update_generated_len(size_t updated_len) {
        m_generated_tokens = updated_len;
    }

    size_t get_generated_len() {
        return m_generated_tokens;
    }

    void register_new_generated_token(int64_t new_token_id) {
        auto it = m_unique_generated_token_ids->find(new_token_id);
        if (it == m_unique_generated_token_ids->end()) {
            m_unique_generated_token_ids->insert({new_token_id, 1});
        } else {
            it->second++;
        }
        for (const auto& transformer : m_stateful_logit_transformers) {
            if (transformer->is_applicable(m_generated_tokens)) {
                transformer->accept_tokens({new_token_id});
            }
        }
    }

    void decrease_generated_token_occurance(int64_t token_id) {
        OPENVINO_ASSERT(m_unique_generated_token_ids->count(token_id) > 0);
        m_unique_generated_token_ids->at(token_id)--;
    }

};

} // namespace ov::genai
