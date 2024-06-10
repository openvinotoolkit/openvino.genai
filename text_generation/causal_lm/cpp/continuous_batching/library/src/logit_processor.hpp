// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>

#include "generation_config.hpp"

namespace LogitTransformers {

using TokenIds = std::vector<int64_t>;
using LogitWithIdx = std::pair<float, size_t>;
using ProbabilityWithIdx = std::pair<float, size_t>;

class ILogitTransformer {
public:
    virtual std::vector<ProbabilityWithIdx> apply(const std::vector<ProbabilityWithIdx>& input_logits) = 0;

protected:
    double m_value = 0.f;
    std::map<int64_t, size_t> m_unique_generated_token_ids;
    std::set<int64_t> m_unique_prompt_token_ids;

    void extract_generated_tokens(const TokenIds& input_ids) {
        std::map<int64_t, size_t> unique_input_ids;
        for (const auto& input_id : input_ids) {
            if (unique_input_ids.count(input_id)) {
                unique_input_ids[input_id]++;
            } else {
                unique_input_ids.insert({input_id, 1});
            }
        }
        this->m_unique_generated_token_ids = unique_input_ids;
    }

    ILogitTransformer() = default;
};

class TopPFilter : public ILogitTransformer {
public:
    TopPFilter(double top_p) {
        m_value = top_p;
    }

    std::vector<ProbabilityWithIdx> apply(const std::vector<ProbabilityWithIdx>& input_probs) override {
        std::vector<ProbabilityWithIdx> tmp(input_probs);
        std::sort(tmp.begin(), tmp.end(), [](const ProbabilityWithIdx& lhs, const ProbabilityWithIdx& rhs) {return lhs.first > rhs.first; });
        float probability_sum = 0.0f;
        size_t nucleus_size = 0;
        for (const auto& probability : tmp) {
            probability_sum += probability.first;
            nucleus_size += 1;
            if (probability_sum > m_value) break;
        }
        tmp.resize(nucleus_size);
        return tmp;
    }
};

class TopKFilter : public ILogitTransformer {
public:
    TopKFilter(size_t top_k) {
        m_value = top_k;
    }

    std::vector<ProbabilityWithIdx> apply(const std::vector<ProbabilityWithIdx>& input_probs) override {
        std::vector<ProbabilityWithIdx> tmp(input_probs);
        std::sort(tmp.begin(), tmp.end(), [](const ProbabilityWithIdx& lhs, const ProbabilityWithIdx& rhs) {return lhs.first > rhs.first; });
        size_t top_k = input_probs.size() >= m_value ? m_value : input_probs.size();
        tmp.resize(top_k);
        return tmp;
    }
};

class TemperatureLogitTransform : public ILogitTransformer {
public:
    TemperatureLogitTransform(double temperature) {
        m_value = temperature;
    };

    std::vector<ProbabilityWithIdx> apply(const std::vector<LogitWithIdx>& input_logits) override {
        std::vector<ProbabilityWithIdx> output(input_logits.begin(), input_logits.end());
        std::sort(output.begin(), output.end(), [](const ProbabilityWithIdx& lhs, const ProbabilityWithIdx& rhs) {return lhs.first > rhs.first; });
        float max_logit = output[0].first;

        std::for_each(output.begin(), output.end(), [max_logit, this](ProbabilityWithIdx& val) {val.first = expf((val.first - max_logit) / this->m_value);});

        float norm_sum = 0.0;
        for (const auto& val : output) {
            norm_sum += val.first;
        }

        std::for_each(output.begin(), output.end(), [norm_sum](ProbabilityWithIdx& val) {val.first /= norm_sum;});
        return output;
    }
};

class RepetitionPenaltyTransform : public ILogitTransformer {
public:
    RepetitionPenaltyTransform(
        double value,
        const std::map<int64_t, size_t>& unique_generated_token_ids = {},
        const std::set<int64_t>& unique_prompt_token_ids = {}) {
        m_value = value;
        m_unique_generated_token_ids = unique_generated_token_ids;
        m_unique_prompt_token_ids = unique_prompt_token_ids;
    };

    std::vector<LogitWithIdx> apply(const std::vector<LogitWithIdx>& input_logits) override {
        std::vector<LogitWithIdx> output(input_logits.begin(), input_logits.end());
        size_t vocab_size = input_logits.size();
        for (const auto& prompt_id : m_unique_prompt_token_ids) {
            OPENVINO_ASSERT((prompt_id >= 0) && (prompt_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(input_logits[prompt_id].second == prompt_id, "input_logits must have original index order");
            auto logit_value = output[prompt_id].first;
            if (logit_value >= 0) {
                output[prompt_id].first /= m_value;
            } else {
                output[prompt_id].first *= m_value;
            };
        }
        for (const auto& input_id_pair : m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(input_logits[input_id].second == input_id, "input_logits must have original index order");
            auto logit_value = output[input_id].first;
            if (logit_value >= 0) {
                output[input_id].first /= m_value;
            } else {
                output[input_id].first *= m_value;
            };
        }
        return output;
    }

    std::vector<LogitWithIdx> apply(const std::vector<LogitWithIdx>& input_logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        return this->apply(input_logits);
    }
};

class FrequencyPenaltyTransform : public ILogitTransformer {
public:
    FrequencyPenaltyTransform(
        double value,
        const std::map<int64_t, size_t>& unique_generated_token_ids = {}) {
        m_value = value;
        m_unique_generated_token_ids = unique_generated_token_ids;
    };

    std::vector<LogitWithIdx> apply(const std::vector<LogitWithIdx>& input_logits) override {
        std::vector<LogitWithIdx> output(input_logits.begin(), input_logits.end());
        size_t vocab_size = input_logits.size();
        for (const auto& input_id_pair : m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(input_logits[input_id].second == input_id, "input_logits must have original index order");
            auto logit_value = output[input_id].first;
            if (logit_value >= 0) {
                output[input_id].first -= m_value * input_id_pair.second;
            } else {
                output[input_id].first += m_value * input_id_pair.second;
            };
        }
        return output;
    }

    std::vector<LogitWithIdx> apply(const std::vector<LogitWithIdx>& input_logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        return this->apply(input_logits);
    }
};

class PresencePenaltyTransform : public ILogitTransformer {
public:
    PresencePenaltyTransform(
        double value,
        const std::map<int64_t, size_t>& unique_generated_token_ids = {}) {
        m_value = value;
        m_unique_generated_token_ids = unique_generated_token_ids;
    };

    std::vector<LogitWithIdx> apply(const std::vector<LogitWithIdx>& input_logits) override {
        std::vector<LogitWithIdx> output(input_logits.begin(), input_logits.end());
        size_t vocab_size = input_logits.size();
        for (const auto& input_id_pair : m_unique_generated_token_ids) {
            const auto& input_id = input_id_pair.first;
            OPENVINO_ASSERT((input_id >= 0) && (input_id < vocab_size), "input_ids token out of bounds");
            OPENVINO_ASSERT(input_logits[input_id].second == input_id, "input_logits must have original index order");
            auto logit_value = output[input_id].first;
            if (logit_value >= 0) {
                output[input_id].first -= m_value;
            } else {
                output[input_id].first += m_value;
            };
        }
        return output;
    }

    std::vector<LogitWithIdx> apply(const std::vector<LogitWithIdx>& input_logits, const TokenIds& input_ids) {
        extract_generated_tokens(input_ids);
        return this->apply(input_logits);
    }
};


class ProbabilityNormalizeTransform : public ILogitTransformer {
public:
    ProbabilityNormalizeTransform() = default;

    std::vector<ProbabilityWithIdx> apply(const std::vector<ProbabilityWithIdx>& input_probs) override {
        std::vector<ProbabilityWithIdx> output(input_probs);
        float norm_sum = 0.0;
        for (const auto& val : output) norm_sum += val.first;
        for (auto& val : output) val.first /= norm_sum;
        return output;
    }
};

} // namespace LogitTransformers

class LogitProcessor {
protected:
    std::vector<std::shared_ptr<LogitTransformers::ILogitTransformer>> m_logit_transformers;

public:
    LogitProcessor(const GenerationConfig& sampling_params,
                   const std::map<int64_t, size_t>& unique_generated_token_ids,
                   const std::set<int64_t>& unique_prompt_token_ids) {
        if (sampling_params.is_multinomial() || sampling_params.is_greedy_sampling()) {
            if (sampling_params.repetition_penalty != 1.0f) {
                m_logit_transformers.emplace_back(
                    new LogitTransformers::RepetitionPenaltyTransform(sampling_params.repetition_penalty, unique_generated_token_ids, unique_prompt_token_ids)
                );
            }
            if (sampling_params.presence_penalty != 0.0f) {
                m_logit_transformers.emplace_back(new LogitTransformers::PresencePenaltyTransform(sampling_params.presence_penalty, unique_generated_token_ids));
                
            }
            if (sampling_params.frequence_penalty != 0.0f) {
                m_logit_transformers.emplace_back(
                    new LogitTransformers::FrequencyPenaltyTransform(sampling_params.frequence_penalty, unique_generated_token_ids)
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
    }

    std::vector<LogitTransformers::ProbabilityWithIdx> apply(const std::vector<LogitTransformers::ProbabilityWithIdx>& logits) {
        std::vector<LogitTransformers::ProbabilityWithIdx> outputs(logits.begin(), logits.end());
        for (const auto& transformer : m_logit_transformers) {
            outputs = transformer->apply(outputs);
        }
        return outputs;
    }
};
