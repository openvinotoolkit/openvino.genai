// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>

#include "openvino/genai/generation_config.hpp"
#include "sampling/logit_transformers.hpp"

#ifdef ENABLE_XGRAMMAR
#include "sampling/structured_output/structured_output_controller.hpp"
#endif

namespace ov::genai {
class LogitProcessor {
protected:
    std::vector<std::shared_ptr<LogitTransformers::ILogitTransformer>> m_logit_transformers;
    
    std::shared_ptr<std::map<int64_t, size_t>> m_unique_generated_token_ids = std::shared_ptr<std::map<int64_t, size_t>>(new std::map<int64_t, size_t>);
    std::shared_ptr<std::set<int64_t>> m_unique_prompt_token_ids = std::shared_ptr<std::set<int64_t>>(new std::set<int64_t>);
    size_t m_generated_tokens = 0;

    // speculative decoding parameters
    float m_assistant_confidence_threshold = 0.f;


public:
    LogitProcessor(const ov::genai::GenerationConfig& sampling_params,
                   const LogitTransformers::TokenIds& input_ids
                   #ifdef ENABLE_XGRAMMAR
                   ,std::shared_ptr<ov::genai::StructuredOutputController> structured_output_controller = nullptr
                   #endif
                   ) {
        for (const auto& input_id : input_ids) {
            m_unique_prompt_token_ids->insert(input_id);
        }

        if (sampling_params.min_new_tokens > 0) {
            m_logit_transformers.emplace_back(
                new LogitTransformers::EOSPenaltyTransform(sampling_params.stop_token_ids, sampling_params.min_new_tokens)
            );
        }

        #ifdef ENABLE_XGRAMMAR
        OPENVINO_ASSERT(structured_output_controller != nullptr, "Structured output controller is not initialized");
        if (sampling_params.is_structured_output()) {
            auto transformer = structured_output_controller->get_logits_transformer(sampling_params);
            m_logit_transformers.push_back(transformer);
        }
        #endif

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
                if (sampling_params.top_k > 0 && sampling_params.top_k < std::numeric_limits<size_t>::max()) {
                    m_logit_transformers.emplace_back(new LogitTransformers::TopKFilter(sampling_params.top_k));
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
    }

    void decrease_generated_token_occurance(int64_t token_id) {
        OPENVINO_ASSERT(m_unique_generated_token_ids->count(token_id) > 0);
        m_unique_generated_token_ids->at(token_id)--;
    }

};

} // namespace ov::genai