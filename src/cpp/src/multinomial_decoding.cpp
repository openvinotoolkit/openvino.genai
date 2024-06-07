// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <vector>

#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"


namespace {

struct TokenIdScore {
    int64_t id;
    float score;

    bool operator<(const TokenIdScore& other) const {
        return score < other.score;
    }

    bool operator>(const TokenIdScore& other) const {
        return score > other.score;
    }
};

void apply_softmax_inplace(std::vector<TokenIdScore>& tokens) {
    float max_score = std::max_element(tokens.begin(), tokens.end())->score;
    float sum = 0.f;

    for (auto& token : tokens) {
        float s = std::exp(token.score - max_score);
        token.score = s;
        sum += s;
    }

    float inv_sum = 1.f / sum;

    for (auto& token : tokens) {
        token.score *= inv_sum;
    }
}

TokenIdScore* sample_top_p(TokenIdScore* first, TokenIdScore* last, float top_p) {
    // sort score
    std::sort(first, last, std::greater<TokenIdScore>());

    int tokens_size = last - first;
    std::vector<TokenIdScore> token_scores(tokens_size);
    for (size_t i = 0; i < tokens_size; i++) {
        token_scores[i] = first[i];
    }

    // calculate softmax
    apply_softmax_inplace(token_scores);

    float prefix_sum = 0.0f;

    // top_p
    for (size_t i = 0; i < tokens_size; i++) {
        prefix_sum += token_scores[i].score;
        if (prefix_sum >= top_p) {
            return first + (i + 1);
        }
    }

    return last;
}

void apply_repetition_penalty(float* first, float* last, const std::vector<int64_t>& input_ids, float penalty) {
    const float inv_penalty = 1.f / penalty;
    const int vocab_size = last - first;
    std::vector<bool> occurrence(vocab_size, false);
    for (const int64_t id : input_ids) {
        if (!occurrence[id]) {
            first[id] *= (first[id] > 0) ? inv_penalty : penalty;
        }
        occurrence[id] = true;
    }
}

void apply_inv_temperature(float* first, float* last, float inv_temperature) {
    for (float* it = first; it != last; it++) {
        *it *= inv_temperature;
    }
}

struct RandomSampling {
    const size_t top_k;
    const float top_p;
    const float inv_temperature;
    const float repetition_penalty;

    std::mt19937 gen{std::random_device{}()};

    RandomSampling(ov::genai::GenerationConfig generation_config)
        : top_k{generation_config.top_k},
          top_p{generation_config.top_p},
          inv_temperature{1.f / generation_config.temperature},
          repetition_penalty{generation_config.repetition_penalty} {
    }

    TokenIdScore get_out_token(float* logits, size_t vocab_size, const std::vector<int64_t>& tokens) {
        // logits pre-process
        if (repetition_penalty != 1.0f) {
            apply_repetition_penalty(logits, logits + vocab_size, tokens, repetition_penalty);
        }

        if (inv_temperature != 1.0f) {
            apply_inv_temperature(logits, logits + vocab_size, inv_temperature);
        }

        std::vector<TokenIdScore> token_scores(vocab_size);
        for (size_t i = 0; i < vocab_size; i++) {
            token_scores[i] = TokenIdScore{int64_t(i), logits[i]};
        }

        // top_k sampling
        if (0 < top_k && top_k < token_scores.size()) {
            std::nth_element(token_scores.data(),
                             token_scores.data() + top_k,
                             token_scores.data() + token_scores.size(),
                             std::greater<TokenIdScore>());
            token_scores.resize(top_k);
        }

        // top_p sampling
        if (0.f < top_p && top_p < 1.0f) {
            auto pos = sample_top_p(token_scores.data(), token_scores.data() + token_scores.size(), top_p);
            token_scores.resize(pos - token_scores.data());
        }

        // sample next token
        apply_softmax_inplace(token_scores);
        for (size_t i = 0; i < token_scores.size(); i++) {
            logits[i] = token_scores[i].score;
        }

        std::discrete_distribution<> dist(logits, logits + token_scores.size());
        return token_scores[dist(gen)];
    }
};
}  // namespace

namespace ov {
namespace genai {

ov::genai::EncodedResults multinominal_decoding(ov::InferRequest& m_model_runner,
                                                ov::Tensor input_ids,
                                                ov::Tensor attention_mask,
                                                ov::genai::GenerationConfig config,
                                                std::shared_ptr<ov::genai::StreamerBase> streamer) {
    ov::Shape prompts_shape = input_ids.get_shape();
    size_t batch_size = prompts_shape[0];

    OPENVINO_ASSERT(batch_size == 1, "Only batch size = 1 supported for multinomial decoding");

    size_t prompt_len = prompts_shape[1];

    ov::genai::EncodedResults results;
    results.scores.resize(batch_size, 0);
    results.tokens.resize(batch_size);

    // Initialize inputs
    m_model_runner.set_tensor("input_ids", input_ids);
    m_model_runner.set_tensor("attention_mask", attention_mask);

    auto num_inputs = m_model_runner.get_compiled_model().inputs().size();
    bool position_ids_available = num_inputs == 4;
    if (position_ids_available) {
        ov::Tensor position_ids = m_model_runner.get_tensor("position_ids");
        position_ids.set_shape(input_ids.get_shape());
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    }
    
    // Input values are persistent between inference calls.
    // That allows to set values, which aren't going to change, only once
    m_model_runner.get_tensor("beam_idx").set_shape({batch_size});
    m_model_runner.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    m_model_runner.infer();

    auto logits_tensor = m_model_runner.get_tensor("logits");

    int64_t sequence_offset = logits_tensor.get_shape().at(1) - 1;
    size_t vocab_size = logits_tensor.get_shape().back();

    float* logits = logits_tensor.data<float>() + sequence_offset * vocab_size;

    const int64_t* input_ids_data = input_ids.data<const int64_t>();

    std::vector<int64_t> tokens{input_ids_data, input_ids_data + input_ids.get_size()};

    RandomSampling sampling{config};

    TokenIdScore out_token = sampling.get_out_token(logits, vocab_size, tokens);

    tokens.push_back(out_token.id);
    results.tokens[0].push_back(out_token.id);
    results.scores[0] += out_token.score;

    if (streamer && streamer->put(out_token.id)) {
        return results;
    }

    if (!config.ignore_eos && out_token.id == config.eos_token_id) {
        return results;
    }

    m_model_runner.get_tensor("input_ids").set_shape({batch_size, 1});
    if (position_ids_available)
        m_model_runner.get_tensor("position_ids").set_shape({batch_size, 1});

    size_t max_new_tokens = config.get_max_new_tokens(prompt_len);

    for (size_t i = 0; i < max_new_tokens - 1; i++) {
        if (position_ids_available) {
            ov::genai::utils::update_position_ids(m_model_runner.get_tensor("position_ids"),
                                                  m_model_runner.get_tensor("attention_mask"));
        }
        m_model_runner.set_tensor("attention_mask",
                                  ov::genai::utils::extend_attention(m_model_runner.get_tensor("attention_mask")));

        m_model_runner.get_tensor("input_ids").data<int64_t>()[0] = out_token.id;

        m_model_runner.infer();

        logits = m_model_runner.get_tensor("logits").data<float>();
        out_token = sampling.get_out_token(logits, vocab_size, tokens);

        tokens.push_back(out_token.id);
        results.tokens[0].push_back(out_token.id);
        results.scores[0] += out_token.score;

        if (streamer && streamer->put(out_token.id)) {
            return results;
        }
    
        if (!config.ignore_eos && out_token.id == config.eos_token_id) {
            break;
        }
    }

    if (streamer) {
        streamer->end();
    }

    return results;
}
}  // namespace genai
}  // namespace ov
