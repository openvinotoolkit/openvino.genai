// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <regex>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

struct TokenIdScore {
    int id;
    float score;

    TokenIdScore() = default;
    TokenIdScore(int id, float score) : id(id), score(score) {}

    bool operator<(const TokenIdScore& other) const { return score < other.score; }
    bool operator>(const TokenIdScore& other) const { return score > other.score; }

    friend std::ostream& operator<<(std::ostream& os, const TokenIdScore& self) {
        return os << "TokenIdScore(id=" << self.id << ", score=" << self.score << ")";
    }
};

void sampling_softmax_inplace(TokenIdScore* first, TokenIdScore* last) {
    float max_score = std::max_element(first, last)->score;
    float sum = 0.f;
    for (TokenIdScore* p = first; p != last; p++) {
        float s = std::exp(p->score - max_score);
        p->score = s;
        sum += s;
    }
    float inv_sum = 1.f / sum;
    for (TokenIdScore* p = first; p != last; p++) {
        p->score *= inv_sum;
    }
}

void sampling_top_k(TokenIdScore* first, TokenIdScore* kth, TokenIdScore* last) {
    std::nth_element(first, kth, last, std::greater<TokenIdScore>());
}

TokenIdScore* sampling_top_p(TokenIdScore* first, TokenIdScore* last, float top_p) {
    // fast top_p in expected O(n) time complexity
    sampling_softmax_inplace(first, last);

    while (first + 1 < last) {
        const float pivot_score = (last - 1)->score; // use mid score?
        TokenIdScore* mid =
            std::partition(first, last - 1, [pivot_score](const TokenIdScore& x) { return x.score > pivot_score; });
        std::swap(*mid, *(last - 1));

        const float prefix_sum =
            std::accumulate(first, mid, 0.f, [](float sum, const TokenIdScore& x) { return sum + x.score; });
        if (prefix_sum >= top_p) {
            last = mid;
        }
        else if (prefix_sum + mid->score < top_p) {
            first = mid + 1;
            top_p -= prefix_sum + mid->score;
        }
        else {
            return mid + 1;
        }
    }
    return last;
}

void sampling_repetition_penalty(float* first, float* last, const std::vector<int64_t>& input_ids,
    float penalty) {
    if (penalty < 0) {
        std::cout << "penalty must be a positive float, but got " << penalty;
        return;
    }
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

void sampling_temperature(float* first, float* last, float temp) {
    const float inv_temp = 1.f / temp;
    for (float* it = first; it != last; it++) {
        *it *= inv_temp;
    }
}

struct SamplingParameters {
    std::vector<int64_t> prompt;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.95;
    float repeat_penalty = 1.1;
    bool do_sample = true;
};

// GreedySampling processes logits prduced by a language model and chooses the token with
// the highest probablity as the next token in the sequence. get_out_token() returns token 
// ids selected by the algorithm. The value is used for next inference. 
struct GreedySampling {
    SamplingParameters parameters;
    GreedySampling(SamplingParameters parameters) : parameters{ std::move(parameters) } {        
    }

    int64_t get_out_token(float* logits, size_t vocab_size) {
        int64_t out_token;
        std::vector<int64_t> prompt{ parameters.prompt };

        // logits pre-process
        if (parameters.repeat_penalty != 1.f) {
            sampling_repetition_penalty(logits, logits + vocab_size, prompt, parameters.repeat_penalty);
        }

        if (parameters.do_sample)
        {
            if (parameters.temp > 0) {
                sampling_temperature(logits, logits + vocab_size, parameters.temp);
            }

            std::vector<TokenIdScore> token_scores(vocab_size);
            for (int i = 0; i < vocab_size; i++) {
                token_scores[i] = TokenIdScore(i, logits[i]);
            }

            // top_k sampling
            if (0 < parameters.top_k && parameters.top_k < (int)token_scores.size()) {
                sampling_top_k(token_scores.data(), token_scores.data() + parameters.top_k,
                    token_scores.data() + token_scores.size());
                token_scores.resize(parameters.top_k);
            }

            // top_p sampling
            if (0.f < parameters.top_p && parameters.top_p < 1.f) {
                auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), parameters.top_p);
                token_scores.resize(pos - token_scores.data());
            }

            // sample next token
            sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
            for (size_t i = 0; i < token_scores.size(); i++) {
                logits[i] = token_scores[i].score;
            }

            thread_local std::random_device rd;
            thread_local std::mt19937 gen(rd());

            std::discrete_distribution<> dist(logits, logits + token_scores.size());
            out_token = token_scores[dist(gen)].id;
        }
        else {
            out_token = std::max_element(logits, logits + vocab_size) - logits;
        }

        prompt.push_back(out_token);

        return { out_token };
    }
};