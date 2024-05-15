// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/runtime/tensor.hpp>
#include "group_beam_searcher.hpp"
#include "generation_config_helper.hpp"
#include "utils.hpp"

namespace {

// Modifyed Knuth–Morris–Pratt algorithm which returns tokens following after every needle occurance in haystack
std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, const std::vector<int64_t>& needle) {
    if (needle.empty()) {  // no_repeat_ngram_size == 1, ban every token
        return {haystack.begin(), haystack.end()};
    }
    std::vector<int> partial_match_table(needle.size() + 1, -1);
    int cnd = 0;
    for (size_t pos = 1; pos < needle.size(); ++pos) {
        if (needle.at(pos) == needle.at(size_t(cnd))) {
            partial_match_table.at(pos) = partial_match_table.at(size_t(cnd));
        } else {
            partial_match_table.at(pos) = cnd;
            while (cnd >= 0 && needle.at(pos) != needle.at(size_t(cnd))) {
                cnd = partial_match_table.at(size_t(cnd));
            }
        }
        ++cnd;
    }
    partial_match_table.back() = cnd;
    std::vector<int64_t> res;
    size_t haystack_id = 0;
    int needle_id = 0;
    while (haystack_id < haystack.size() - 1) {
        if (needle.at(size_t(needle_id)) == haystack.at(haystack_id)) {
            ++haystack_id;
            ++needle_id;
            if (needle_id == int(needle.size())) {
                res.push_back(haystack.at(haystack_id));
                needle_id = partial_match_table.at(size_t(needle_id));
            }
        } else {
            needle_id = partial_match_table.at(size_t(needle_id));
            if (needle_id < 0) {
                ++haystack_id;
                ++needle_id;
            }
        }
    }
    return res;
}

struct Token {
    float log_prob;
    int64_t idx;
};

std::vector<Token> log_softmax(const ov::Tensor& logits, const size_t batch_idx) {
    if (logits.get_shape().at(0) <= batch_idx) {
        throw std::runtime_error("logits batch size doesn't match the number of beams");
    }
    size_t vocab_size = logits.get_shape().back();
    size_t batch_offset = batch_idx * logits.get_shape().at(1) * vocab_size;
    size_t sequence_offset = (logits.get_shape().at(1) - 1) * vocab_size;
    const float* beam_logits = logits.data<const float>() + batch_offset + sequence_offset;
    float max_logit = *std::max_element(beam_logits, beam_logits + vocab_size);
    float log_sum = std::log(
        std::accumulate(beam_logits, beam_logits + vocab_size, 0.0f, [max_logit](float accumulated, float to_add) {
            return accumulated + std::exp(to_add - max_logit);
        }));
    std::vector<Token> tokens;
    tokens.reserve(vocab_size);
    for (size_t idx = 0; idx < vocab_size; ++idx) {
        tokens.push_back({beam_logits[idx] - max_logit - log_sum, int64_t(idx)});
    }
    return tokens;
}

struct Beam {
    float score = -std::numeric_limits<float>::infinity();  // The bigger, the better
    std::vector<int64_t> tokens;
    size_t global_beam_idx = 0;
};

bool greater(const Beam& left, const Beam& right) {
    return left.score > right.score;
}

struct Parameters {
    std::vector<std::vector<int64_t>> prompts;
    int64_t eos_token;
    size_t n_groups = 3;
    size_t group_size = 5;
    float diversity_penalty = 1.0;
    size_t max_new_tokens = 20;
    ov::StopCriteria stop_criteria = ov::StopCriteria::heuristic;
    float length_penalty = 1.0;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();

    std::function<bool(const Beam&)> early_finish = [](const Beam&) {
        return false;
    };
};

struct Group {
    std::vector<Beam> ongoing;   // Best beams in front
    std::vector<Beam> min_heap;  // The worst of the best completed beams is the first
    bool done = false;

    void finish(Beam&& beam, const Parameters& parameters) {
        beam.score /= std::pow(float(beam.tokens.size()), parameters.length_penalty);

        // HF implementation counts eos_token for length penalty calculation
        if (beam.tokens.back() == parameters.eos_token) {
            beam.tokens.pop_back();
        }

        min_heap.push_back(std::move(beam));
        std::push_heap(min_heap.begin(), min_heap.end(), greater);
        if (min_heap.size() > parameters.group_size) {
            std::pop_heap(min_heap.begin(), min_heap.end(), greater);
            min_heap.pop_back();
        }
    }
    void is_done(const Parameters& parameters) {
        if (min_heap.size() < parameters.group_size) {
            return;
        }
        size_t cur_len = ongoing.front().tokens.size();
        float best_sum_logprobs = ongoing.front().score;
        float worst_score = min_heap.front().score;
        switch (parameters.stop_criteria) {
        case ov::StopCriteria::early:
            done = true;
            return;
        case ov::StopCriteria::heuristic: {
            float highest_attainable_score = best_sum_logprobs / std::pow(float(cur_len), parameters.length_penalty);
            done = worst_score >= highest_attainable_score;
            return;
        }
        case ov::StopCriteria::never: {
            size_t length = parameters.length_penalty > 0.0 ? parameters.max_new_tokens : cur_len;
            float highest_attainable_score = best_sum_logprobs / std::pow(float(length), parameters.length_penalty);
            done = worst_score >= highest_attainable_score;
            return;
        }
        default:
            throw std::runtime_error("Never reached");
        }
    }
};

// GroupBeamSearcher processes logits prduced by a language model and accumulates beams using group beam search
// algorithm. select_next_tokens() returns token ids selected by the algorithm and corresponding beam ids. These values
// are used for next inference. select_next_tokens() returns empty, if all groups are completed
struct GroupBeamSearcher {
    Parameters parameters;
    std::vector<std::vector<Group>> prompts_groups;

    GroupBeamSearcher(Parameters parameters) : parameters{parameters}, prompts_groups{parameters.prompts.size()} {
        if (parameters.no_repeat_ngram_size == 0) {
            throw std::runtime_error("no_repeat_ngram_size must be positive");
        }
        for (std::vector<Group>& prompts_groups : prompts_groups) {
            prompts_groups.resize(parameters.n_groups);
            for (Group& group : prompts_groups) {
                group.ongoing.resize(parameters.group_size);
                group.ongoing.front().score = 0.0;
            }
        }
    }

    std::pair<std::vector<int64_t>, std::vector<int32_t>> select_next_tokens(const ov::Tensor& logits) {
        std::vector<int64_t> next_tokens;
        std::vector<int32_t> next_beams;

        const size_t promts_size = parameters.prompts.size();

        next_tokens.reserve(promts_size * parameters.n_groups * parameters.group_size);
        next_beams.reserve(promts_size * parameters.n_groups * parameters.group_size);

        size_t beam_count = 0;
        size_t prompt_id = 0;
        for (std::vector<Group>& groups : prompts_groups) {
            for (Group& group : groups) {
                if (group.done) {
                    continue;
                }
                for (Beam& beam : group.ongoing) {
                    // beam.tokens.empty() holds for the first select_next_tokens() call.
                    // Every beam is constructed from the single batch at first call
                    if (beam.tokens.empty()) {
                        beam.global_beam_idx = prompt_id;
                    } else {
                        beam.global_beam_idx = beam_count;
                        ++beam_count;
                    }
                }
            }

            prompt_id += 1;
        }

        for (int prompt_id = 0; prompt_id < promts_size; prompt_id++) {
            const std::vector<int64_t> prompt = parameters.prompts[prompt_id];
            std::vector<Group>& groups = prompts_groups[prompt_id];
            auto [prompt_next_tokens, prompt_next_beams] = select_prompt_next_tokens(logits, prompt, groups);

            next_tokens.insert(next_tokens.end(), prompt_next_tokens.begin(), prompt_next_tokens.end());
            next_beams.insert(next_beams.end(), prompt_next_beams.begin(), prompt_next_beams.end());
        }

        return {next_tokens, next_beams};
    }

    std::pair<std::vector<int64_t>, std::vector<int32_t>> select_prompt_next_tokens(const ov::Tensor& logits,
                                                                                    const std::vector<int64_t>& prompt,
                                                                                    std::vector<Group>& groups) {
        std::vector<int64_t> next_tokens;
        std::vector<int32_t> next_beams;
        next_tokens.reserve(parameters.n_groups * parameters.group_size);
        next_beams.reserve(parameters.n_groups * parameters.group_size);

        for (auto group = groups.begin(); group != groups.end(); ++group) {
            if (group->done) {
                continue;
            }
            std::vector<Beam> candidates;
            candidates.reserve(parameters.group_size * 2 * parameters.group_size);
            for (const Beam& beam : group->ongoing) {
                std::vector<Token> tokens = log_softmax(logits, beam.global_beam_idx);
                for (auto prev_group = groups.cbegin(); prev_group != group; ++prev_group) {
                    for (const Beam& prev_beam : prev_group->ongoing) {
                        if (prev_beam.tokens.size() > beam.tokens.size()) {
                            tokens.at(size_t(prev_beam.tokens.back())).log_prob -= parameters.diversity_penalty;
                        }
                    }
                }
                std::vector<int64_t> full_text{prompt};
                full_text.insert(full_text.end(), beam.tokens.begin(), beam.tokens.end());
                if (full_text.size() > 1 && full_text.size() >= parameters.no_repeat_ngram_size) {
                    auto tail_start = full_text.end() - ptrdiff_t(parameters.no_repeat_ngram_size) + 1;
                    for (int64_t banned_token : kmp_search(full_text, {tail_start, full_text.end()})) {
                        tokens.at(size_t(banned_token)).log_prob = -std::numeric_limits<float>::infinity();
                    }
                }
                std::sort(tokens.begin(), tokens.end(), [](Token left, Token right) {
                    return left.log_prob > right.log_prob;  // Most probable tokens in front
                });
                size_t add_count = 0;
                for (Token token : tokens) {
                    Beam new_candidate = beam;
                    new_candidate.score += token.log_prob;
                    new_candidate.tokens.push_back(token.idx);
                    if (parameters.early_finish(new_candidate)) {
                        group->finish(std::move(new_candidate), parameters);
                    } else {
                        candidates.push_back(std::move(new_candidate));
                        ++add_count;
                        if (add_count == 2 * parameters.group_size) {
                            break;
                        }
                    }
                }
            }
            // Sample 2 * group_size highest score tokens to get at least 1 non EOS token per beam
            if (candidates.size() < 2 * parameters.group_size) {
                throw std::runtime_error("No beams left to search");
            }
            auto to_sort = candidates.begin() + ptrdiff_t(2 * parameters.group_size);
            std::partial_sort(candidates.begin(), to_sort, candidates.end(), greater);
            group->ongoing.clear();
            for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
                if (parameters.eos_token == candidates.at(cand_idx).tokens.back()) {
                    // If beam_token does not belong to top num_beams tokens, it should not be added
                    if (cand_idx >= parameters.group_size) {
                        continue;
                    }
                    group->finish(std::move(candidates.at(cand_idx)), parameters);
                } else {
                    group->ongoing.push_back(std::move(candidates.at(cand_idx)));
                    if (group->ongoing.size() == parameters.group_size) {
                        break;
                    }
                }
            }
            group->is_done(parameters);
            if (!group->done) {
                for (const Beam& beam : group->ongoing) {
                    next_tokens.push_back(beam.tokens.back());
                    next_beams.push_back(int32_t(beam.global_beam_idx));
                }
            }
        }
        return {next_tokens, next_beams};
    }
};

// Consume group_beam_searcher because beams are consumed
std::vector<std::vector<std::vector<Beam>>> finalize(GroupBeamSearcher&& group_beam_searcher) {
    std::vector<std::vector<std::vector<Beam>>> finalized;
    finalized.resize(group_beam_searcher.prompts_groups.size());

    for (size_t prompt_id = 0; prompt_id < group_beam_searcher.prompts_groups.size(); prompt_id++) {
        std::vector<Group>& groups = group_beam_searcher.prompts_groups.at(prompt_id);
        finalized.at(prompt_id).reserve(groups.size());

        for (Group& group : groups) {
            if (!group.done) {
                for (Beam& beam : group.ongoing) {
                    group.finish(std::move(beam), group_beam_searcher.parameters);
                }
            }
            finalized.at(prompt_id).push_back(std::move(group.min_heap));
        }
    }

    return finalized;
}

void initialize_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, ov::InferRequest& request) {
    request.set_tensor("input_ids", input_ids);
    request.set_tensor("attention_mask", attention_mask);

    ov::Shape input_shape = input_ids.get_shape();

    ov::Tensor position_ids = request.get_tensor("position_ids");
    position_ids.set_shape(input_shape);
    ov::generate_utils::initialize_position_ids(position_ids, attention_mask);

    ov::Tensor beam_idx = request.get_tensor("beam_idx");
    beam_idx.set_shape({input_shape.at(0)});
    std::fill_n(beam_idx.data<int32_t>(), input_shape.at(0), 0);
}


void update_attention_mask_with_beams(ov::Tensor&& attention_mask, std::vector<int32_t> next_beams) {
    ov::Tensor original_mask{ov::element::i64, attention_mask.get_shape()};
    ov::Shape original_shape = original_mask.get_shape();
    attention_mask.copy_to(original_mask);

    ov::Shape new_shape{next_beams.size(), original_mask.get_shape().at(1) + 1};
    attention_mask.set_shape(new_shape);

    for (size_t beam_id = 0; beam_id < next_beams.size(); beam_id++) {
        const size_t original_prompt_offset = next_beams.at(beam_id) * original_shape.at(1);
        const size_t result_prompt_offset = beam_id * new_shape.at(1);

        int64_t* dest = attention_mask.data<int64_t>() + result_prompt_offset;
        const int64_t* src = original_mask.data<int64_t>() + original_prompt_offset;

        std::memcpy(dest, src, original_shape.at(1) * sizeof(int64_t));
        attention_mask.data<int64_t>()[result_prompt_offset + new_shape.at(1) - 1] = 1;
    }
}

void update_position_ids(ov::Tensor&& position_ids, const ov::Tensor&& attention_mask) {
    const size_t batch_size = attention_mask.get_shape().at(0);
    const size_t sequence_length = attention_mask.get_shape().at(1);
    position_ids.set_shape({batch_size, 1});

    for (size_t batch = 0; batch < batch_size; batch++) {
        int64_t* mask_start = attention_mask.data<int64_t>() + batch * sequence_length;
        position_ids.data<int64_t>()[batch] = std::accumulate(mask_start, mask_start + sequence_length - 1, 0);
    }
}

} // namespace


namespace ov {

EncodedResults beam_search(ov::InferRequest& lm, ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig config) {
    GenerationConfigHelper config_helper = config;
    OPENVINO_ASSERT(config.num_beams % config.num_beam_groups == 0, "number of beams should be divisible by number of groups");
    
    // Initialize beam search
    const int64_t* prompt_data = input_ids.data<const int64_t>();
    std::vector<std::vector<int64_t>> prompts;
    prompts.reserve(input_ids.get_shape().at(0));
    for (size_t batch = 0; batch < input_ids.get_shape().at(0); batch++) {
        size_t sequence_length = input_ids.get_shape().at(1);
        size_t batch_offset = batch * sequence_length;
        const int64_t* prompt_start = prompt_data + batch_offset;
        prompts.push_back(std::vector<int64_t>{prompt_start, prompt_start + sequence_length});
    }
    
    initialize_inputs(input_ids, attention_mask, lm);

    Parameters parameters{std::move(prompts)};
    parameters.max_new_tokens = config.max_new_tokens;
    parameters.eos_token = config.eos_token_id;
    parameters.n_groups = config.num_beam_groups;
    parameters.group_size = config.num_beams / config.num_beam_groups;
    parameters.diversity_penalty = config.diversity_penalty;
    parameters.length_penalty = config.length_penalty;
    parameters.stop_criteria = config.stop_criteria;
    parameters.no_repeat_ngram_size = config.no_repeat_ngram_size;
    GroupBeamSearcher group_beam_searcher{parameters};

    std::vector<int64_t> next_tokens;
    std::vector<int32_t> next_beams;

    for (size_t length_count = 0; length_count < parameters.max_new_tokens; ++length_count) {
        lm.infer();

        std::tie(next_tokens, next_beams) = group_beam_searcher.select_next_tokens(lm.get_tensor("logits"));
        if (next_tokens.empty()) {
            break;
        }
        size_t batch_size = next_tokens.size();
        // Set pointers
        lm.set_tensor("input_ids", ov::Tensor{ov::element::i64, {batch_size, 1}, next_tokens.data()});
        lm.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {batch_size}, next_beams.data()});
        // Set auxiliary inputs
        update_attention_mask_with_beams(lm.get_tensor("attention_mask"), next_beams);
        update_position_ids(lm.get_tensor("position_ids"), lm.get_tensor("attention_mask"));
    }

    std::vector<Beam> beams;
    for (const std::vector<std::vector<Beam>>& prompt_group : finalize(std::move(group_beam_searcher))) {
        for (const std::vector<Beam> group : prompt_group) {
            for (const Beam& beam : group) {
                beams.emplace_back(beam);
            }
        }
    }
    
    // return sorted scores
    auto compare_scores = [](Beam left, Beam right) { return (left.score > right.score); };
    std::sort(beams.begin(), beams.end(), compare_scores);
    
    ov::EncodedResults results;
    for (auto beam = beams.begin(); beam != beams.begin() + config.num_return_sequences; ++beam) {
        results.scores.emplace_back(beam->score);
        results.tokens.emplace_back(beam->tokens);
    }
    return results;
}

} // namespace ov
