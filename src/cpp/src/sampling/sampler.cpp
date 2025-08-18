// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <future>

#include "sampling/sampler.hpp"

namespace ov::genai {
// Modified Knuth–Morris–Pratt algorithm which returns tokens following after every needle occurrence in haystack
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

std::vector<Token> log_softmax(const ov::Tensor& logits, size_t batch_idx) {
    ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3);
    size_t batch = shape[0], seq_len = shape[1], vocab_size = shape[2];
    OPENVINO_ASSERT(batch_idx < batch, "Logits batch size doesn't match the number of beams");

    size_t batch_offset = batch_idx * seq_len * vocab_size, sequence_offset = (seq_len - 1) * vocab_size;
    const float* beam_logits = logits.data<const float>() + batch_offset + sequence_offset;
    float max_logit = *std::max_element(beam_logits, beam_logits + vocab_size);
    float log_sum = std::log(std::accumulate(
        beam_logits, beam_logits + vocab_size, 0.0f, [max_logit](float accumulated, float to_add) {
            return accumulated + std::exp(to_add - max_logit);
    }));

    std::vector<Token> tokens;
    tokens.reserve(vocab_size);
    for (size_t idx = 0; idx < vocab_size; ++idx)
        tokens.push_back({beam_logits[idx] - max_logit - log_sum, int64_t(idx)});

    return tokens;
}

std::vector<int64_t> wrap_tokens(const std::vector<int64_t>& tokens, const std::vector<int64_t>& prefix_tokens, const std::vector<int64_t>& suffix_tokens) {
    std::vector<int64_t> all_tokens = prefix_tokens;
    all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
    all_tokens.insert(all_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
    return all_tokens;
}

std::string clean_wrapped_text(const std::string& wrapped_text, const std::string& prefix, const std::string& suffix) {
    auto prefix_pos = wrapped_text.find(prefix);
    OPENVINO_ASSERT(prefix_pos != std::string::npos);
    auto suffix_pos = wrapped_text.rfind(suffix);
    OPENVINO_ASSERT(suffix_pos != std::string::npos);
    auto clean_text_start = prefix_pos + prefix.size();
    auto clean_text_length = suffix_pos - clean_text_start;
    std::string clean_text = wrapped_text.substr(clean_text_start, clean_text_length);
    return clean_text;
}

std::vector<int64_t> encode_and_process_string(const std::string& stop_string, ov::genai::Tokenizer& tokenizer) {
    // encode stop_string
    std::string stop_string_copy = stop_string;
    ov::Tensor ov_encoded_stop_string = tokenizer.encode(stop_string_copy, ov::genai::add_special_tokens(false)).input_ids;
    size_t tensor_size = ov_encoded_stop_string.get_size();
    std::vector<int64_t> encoded_stop_string(tensor_size);
    std::copy_n(ov_encoded_stop_string.data<int64_t>(), tensor_size, encoded_stop_string.begin());
    return encoded_stop_string;
}

struct MatchStopStringResult {
    size_t to_remove = 0;
    // int64_t last_token_id = 0;
    // bool is_to_update_last_token = false;
    bool is_matched = false;
};

// Return number of last tokens that match one of the stop_strings. If there's no match 0 is returned.
MatchStopStringResult match_stop_string(Tokenizer& tokenizer,
                      const TokenIds& generated_tokens,
                      const std::pair<size_t, std::set<std::string>>& stop_strings,
                      bool is_include_to_output,
                      size_t draft_generated_tokens = 0) {
    MatchStopStringResult result;
    if (generated_tokens.size() >= stop_strings.first) {
        // draft_generated_tokens is to handle case with >= 1 generated tokens per step
        size_t offset = generated_tokens.size() - draft_generated_tokens;
        if (offset < stop_strings.first) {
            return result;
        }
        offset -= stop_strings.first;
        TokenIds buffer(generated_tokens.begin() + offset, generated_tokens.end());
        std::string decoded_buffer = tokenizer.decode(buffer);
        for (const auto& stop_string : stop_strings.second) {
            auto pos = decoded_buffer.find(stop_string);
            if (pos != std::string::npos) {
                result.is_matched = true;

                auto stop_string_len = is_include_to_output ? stop_string.length() : 0;
                decoded_buffer = decoded_buffer.substr(0, pos + stop_string_len);
                // to remove word splitting symbols from tail
                while (decoded_buffer.back() == ' ' || decoded_buffer.back() == '\n') {
                    decoded_buffer.pop_back();
                }
                if (decoded_buffer.empty()) {
                    result.to_remove = buffer.size();
                    return result;
                }

                // find token cnt to be removed from sequence by decoding token by token
                std::string decoded_partially_string;
                for (size_t i = 0; i < buffer.size(); ++i) {
                    decoded_partially_string = tokenizer.decode(TokenIds{buffer.begin(), buffer.begin() + i + 1});
                    if (decoded_partially_string.find(decoded_buffer) != std::string::npos) {
                        result.to_remove = buffer.size() - i - 1;
                        break;
                    }
                }
                return result;
            }
        }
    }
    return result;
}

// Return number of last tokens that match one of the stop_strings. If there's no match 0 is returned.
// Number of tokens might not be exact as if there's no direct token match, we decode generated tokens incrementally expanding decoding scope
// with 4 next tokens with each iteration until we check all tokens.
int match_stop_string2(Tokenizer & tokenizer, const TokenIds & generated_tokens, const std::set<std::string> & stop_strings) {
    for (auto stop_string: stop_strings) {
        auto stop_tokens_ov = tokenizer.encode(stop_string).input_ids;
        size_t num_tokens = stop_tokens_ov.get_size();
        if(num_tokens > generated_tokens.size())
            continue;

        // Check direct token match
        std::vector<int64_t> stop_tokens(stop_tokens_ov.data<int64_t>(), stop_tokens_ov.data<int64_t>() + num_tokens);
        std::vector<int64_t> last_generated_tokens(generated_tokens.end()-num_tokens, generated_tokens.end());
        if (stop_tokens == last_generated_tokens)
            return num_tokens;
        
        // Continue checking chunks of 4 tokens
        num_tokens += 4;
        while (num_tokens <= generated_tokens.size()) {
            std::vector<int64_t> last_generated_tokens(generated_tokens.end()-num_tokens, generated_tokens.end());
            std::string decoded_last_tokens = tokenizer.decode(last_generated_tokens);
            if (decoded_last_tokens.find(stop_string) != std::string::npos) {
                return num_tokens;
            }
            num_tokens += 4;
        }
    }
    return 0;
}

void Sampler::GroupBeamSearcher::finalize(SamplerOutput& sampler_output) {
    for (Group& group : m_groups) {
        if (!group.done) {
            for (Beam& beam : group.ongoing) {
                uint64_t sequence_id = beam.m_sequence->get_id();

                int64_t preempted_id = group.finish(beam, m_parameters);
                if (preempted_id >= 0) {
                    // remove preempted one
                    m_sequence_group->remove_sequence(preempted_id);
                }

                // mark current sequence as finished
                beam.m_sequence->set_status(SequenceStatus::FINISHED);
                // Setting length since this function is used when sequence generated tokens number reaches max_new_tokens 
                beam.m_sequence->set_finish_reason(GenerationFinishReason::LENGTH);
                // we also need to drop add ongoing / forked sequences from scheduler
                sampler_output.m_dropped_sequences.push_back(sequence_id);
            }
        }
    }
}

Sampler::GroupBeamSearcher::GroupBeamSearcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer)
    : m_sequence_group(sequence_group),
        m_parameters{m_sequence_group->get_sampling_parameters()},
        m_groups{m_parameters.num_beam_groups},
        m_tokenizer(tokenizer) {
    OPENVINO_ASSERT(m_sequence_group->num_running_seqs() == 1);
    assert(m_parameters.num_beams % m_parameters.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = m_parameters.num_beams / m_parameters.num_beam_groups;

    for (Group& group : m_groups) {
        group.ongoing.reserve(group_size);
        // initially we just add our "base" sequence to beams inside each group
        for (size_t i = 0; i < group_size; ++i)
            group.ongoing.push_back(Beam((*sequence_group)[0]));
        // to avoid selecting the same tokens for beams within group, let's just initialize score
        // for the front one
        group.ongoing.front().m_score = 0.0f;
    }
}


std::map<size_t, int32_t> Sampler::GroupBeamSearcher::get_beam_idxs() {
    std::map<size_t, int32_t> next_beams;

    for (Group& group : m_groups) {
        if (!group.done) {
            for (Beam& beam : group.ongoing) {
                next_beams.insert({beam.m_sequence->get_id(), beam.m_global_beam_idx});
            }
        }
    }

    return next_beams;
}

std::pair<std::map<std::string, float>, std::vector<float>> Sampler::get_structured_output_times() {
    if (m_structured_output_controller) {
        return m_structured_output_controller->get_times();
    } else {
        // If compiled without structured output support, return empty times
        return {{}, {}};
    }
}

void Sampler::clear_structured_output_compile_times() {
    if (m_structured_output_controller) {
        m_structured_output_controller->clear_compile_times();
    }
}

void Sampler::GroupBeamSearcher::select_next_tokens(const ov::Tensor& logits,
    SamplerOutput& sampler_output,
    const std::pair<size_t, std::set<std::string>>& stop_strings) {
    assert(m_parameters.num_beams % m_parameters.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = m_parameters.num_beams / m_parameters.num_beam_groups;
    std::vector<int64_t> next_tokens;
    std::vector<int32_t> next_beams;
    next_tokens.reserve(m_parameters.num_beams);
    next_beams.reserve(m_parameters.num_beams);

    // parent sequence ID -> number of child sequences
    std::map<uint64_t, uint64_t> parent_2_num_childs_map;

    for (Group& group : m_groups) {
        if (!group.done) {
            for (Beam& beam : group.ongoing) {
                sampler_output.num_generated_tokens++;
                uint64_t parent_seq_id = beam.m_sequence->get_id();

                // here we need to map index of sequence in beam search group(s) and sequence group
                beam.m_global_beam_idx = [this] (uint64_t seq_id) -> size_t {
                    std::vector<Sequence::Ptr> running_seqs = m_sequence_group->get_running_sequences();
                    for (size_t seq_global_index = 0; seq_global_index < running_seqs.size(); ++seq_global_index) {
                        if (seq_id == running_seqs[seq_global_index]->get_id())
                            return seq_global_index;
                    }
                    OPENVINO_THROW("Internal error in beam search: should not be here");
                } (parent_seq_id);

                // zero out all parent forks counts
                parent_2_num_childs_map[parent_seq_id] = 0;
            }
        }
    }

    auto try_to_finish_candidate = [&] (Group& group, Beam& candidate, bool include_candidate_token = true) -> void {
        uint64_t seq_id = candidate.m_sequence->get_id();
        // try to finish candidate
        int64_t preempted_seq_id = group.finish(candidate, m_parameters);

        // if candidate has lower score than others finished
        if (preempted_seq_id == seq_id) {
            // do nothing and just ignore current finished candidate
        } else {
            if (preempted_seq_id >= 0) {
                m_sequence_group->remove_sequence(preempted_seq_id);
            }

            // need to insert candidate to a sequence group
            Sequence::Ptr forked_sequence = m_sequence_group->fork_sequence(candidate.m_sequence);
            // and finish immediately
            forked_sequence->set_status(SequenceStatus::FINISHED);
            // Setting stop since this function is used when sequence generated eos token
            forked_sequence->set_finish_reason(GenerationFinishReason::STOP);

            // TODO: make it simpler
            // currently, we finish sequence and then fork it in current code
            {
                for (size_t i = 0; i < group.min_heap.size(); ++i) {
                    if (group.min_heap[i].m_sequence->get_id() == seq_id) {
                        group.min_heap[i].m_sequence = forked_sequence;
                        break;
                    }
                }
            }

            // append token from candidate to actual sequence
            if (include_candidate_token)
                forked_sequence->append_token(candidate.m_token_id, candidate.m_log_prob);
        }
    };

    // group ID => child beams
    std::map<int, std::vector<Beam>> child_beams_per_group;

    for (size_t group_id = 0; group_id < m_groups.size(); ++group_id) {
        Group & group = m_groups[group_id];
        if (group.done)
            continue;

        std::vector<Beam> candidates;
        candidates.reserve(group_size * 2 * group_size);
        for (const Beam& beam : group.ongoing) {
            std::vector<Token> tokens = log_softmax(logits, beam.m_global_beam_idx);

            // apply diversity penalty
            for (auto prev_group_id = 0; prev_group_id < group_id; ++prev_group_id) {
                for (const Beam& prev_beam : child_beams_per_group[prev_group_id]) {
                    tokens[prev_beam.m_token_id].m_log_prob -= m_parameters.diversity_penalty;
                }
            }

            // apply n_gramm
            std::vector<int64_t> full_text{m_sequence_group->get_prompt_ids()};
            full_text.insert(full_text.end(), beam.m_sequence->get_generated_ids().begin(), beam.m_sequence->get_generated_ids().end());
            if (full_text.size() > 1 && full_text.size() >= m_parameters.no_repeat_ngram_size) {
                auto tail_start = full_text.end() - ptrdiff_t(m_parameters.no_repeat_ngram_size) + 1;
                for (int64_t banned_token : kmp_search(full_text, {tail_start, full_text.end()})) {
                    tokens[banned_token].m_log_prob = -std::numeric_limits<float>::infinity();
                }
            }

            // sort tokens
            std::sort(tokens.begin(), tokens.end(), [](Token left, Token right) {
                return left.m_log_prob > right.m_log_prob;  // Most probable tokens in front
            });

            size_t add_count = 0;
            for (Token token : tokens) {
                Beam new_candidate = beam;
                new_candidate.m_score += new_candidate.m_log_prob = token.m_log_prob;
                new_candidate.m_token_id = token.m_index;

                // TODO: fix it
                // and ensure cumulative_log prob is used
                if (/* m_parameters.early_finish(new_candidate) */ false) {
                    try_to_finish_candidate(group, new_candidate);
                } else {
                    candidates.push_back(new_candidate);
                    if (++add_count == 2 * group_size) {
                        break;
                    }
                }
            }
        }

        // Sample 2 * group_size highest score tokens to get at least 1 non EOS token per beam
        OPENVINO_ASSERT(candidates.size() >= 2 * group_size, "No beams left to search");

        auto to_sort = candidates.begin() + ptrdiff_t(2 * group_size);
        std::partial_sort(candidates.begin(), to_sort, candidates.end(), greater);

        for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
            Beam & candidate = candidates[cand_idx];
            if (is_stop_token_id_hit(candidate.m_token_id, m_sequence_group->get_sampling_parameters().stop_token_ids)) {
                // If beam_token does not belong to top num_beams tokens, it should not be added
                if (cand_idx >= group_size)
                    continue;

                // try to finish candidate
                try_to_finish_candidate(group, candidate);
                continue;
            }

            if (!m_parameters.stop_strings.empty()) {
                // We need to include candidate token to already generated tokens to check if stop string has been generated
                // There's probably a better way to do that, than copying whole vector...
                std::vector<int64_t> token_ids = candidate.m_sequence->get_generated_ids();
                token_ids.push_back(candidate.m_token_id);
                auto match_result = match_stop_string(m_tokenizer, token_ids, stop_strings, m_parameters.include_stop_str_in_output);
                if (match_result.is_matched) {
                    // If beam_token does not belong to top num_beams tokens, it should not be added
                    if (cand_idx >= group_size)
                        continue;

                    // remove tokens that match stop_string from output (last token is not included in candidate.m_sequence at this point)
                    candidate.m_sequence->remove_last_tokens(match_result.to_remove);

                    // try to finish candidate
                    try_to_finish_candidate(group, candidate);
                    continue;
                }
            }

            parent_2_num_childs_map[candidate.m_sequence->get_id()] += 1;
            child_beams_per_group[group_id].push_back(candidate);

            // if num childs are enough
            if (child_beams_per_group[group_id].size() == group_size) {
                break;
            }
        }

        // check whether group has finished
        group.is_done();

        // group cannot continue if there are no valid child beams
        if (child_beams_per_group[group_id].size() == 0) {
            group.done = true;
        }

        if (group.done) {
            // group has finished, group all running sequences
            for (const Beam& beam : group.ongoing) {
                uint64_t seq_id = beam.m_sequence->get_id();
                m_sequence_group->remove_sequence(seq_id);
                sampler_output.m_dropped_sequences.push_back(seq_id);
            }
            group.ongoing.clear();
        }
    }

    // fork child sequences for non-finished groups

    for (size_t group_id = 0; group_id < m_groups.size(); ++group_id) {
        Group & group = m_groups[group_id];

        if (!group.done) {
            for (Beam& child_beam : child_beams_per_group[group_id]) {
                uint64_t parent_sequence_id = child_beam.m_sequence->get_id();
                uint64_t& num_childs = parent_2_num_childs_map[parent_sequence_id];

                // if current beam is forked multiple times
                if (num_childs > 1) {
                    child_beam.m_sequence = m_sequence_group->fork_sequence(child_beam.m_sequence);
                    child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);

                    // reduce forks count, since fork already happened and next loop iteration
                    // will go by the second branch (num_childs == 1)
                    --num_childs;

                    // fill out sampler output
                    sampler_output.m_forked_sequences[parent_sequence_id].push_back(child_beam.m_sequence->get_id());
                } else if (num_childs == 1) {
                    // keep current sequence going and add a new token
                    child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);
                }
            }

            // drop beams which are not forked by current group
            for (const Beam& beam : group.ongoing) {
                size_t num_childs = parent_2_num_childs_map[beam.m_sequence->get_id()];
                if (num_childs == 0) {
                    // drop sequence as not forked
                    sampler_output.m_dropped_sequences.push_back(beam.m_sequence->get_id());
                    m_sequence_group->remove_sequence(beam.m_sequence->get_id());
                }
            }

            // child become parents
            group.ongoing = child_beams_per_group[group_id];
        }
    }
}

Token Sampler::_greedy_sample(const Logits& logits, size_t top_logprobs) const {
    // For greedy sampling we do not expect sorting or shrinking considered tokens
    // so we can operate directly on the data buffer
    size_t m = std::max(size_t(1), top_logprobs); // ensure m is at least 1
    std::vector<float> top_values(m, -std::numeric_limits<float>::infinity());
    std::vector<size_t> top_indexes(m, 0);

    for (size_t i = 0; i < logits.m_size; ++i) {
        if (logits.m_data[i] > top_values.back()) {
            top_values.back() = logits.m_data[i];
            top_indexes.back() = i;

            for (size_t j = top_values.size() - 1; j > 0 && top_values[j] > top_values[j - 1]; --j) {
                std::swap(top_values[j], top_values[j - 1]);
                std::swap(top_indexes[j], top_indexes[j - 1]);
            }
        }
    }

    size_t max_index = top_indexes.front();
    float max_value = 0.0;

    if (top_logprobs) {
        // apply log softmax to max value
        max_value = top_values.front();
        float log_sum = std::log(std::accumulate(
            logits.m_data, logits.m_data + logits.m_size, 0.0f, [max_value](float accumulated, float to_add) {
                return accumulated + std::exp(to_add - max_value);
        }));
        max_value = -log_sum;
    }

    return Token(max_value, max_index);
}

std::vector<Token> Sampler::_multinomial_sample(const Logits& logits, size_t num_tokens_per_sequence) {
    // If top_p or top_k was applied we use sorted vector, if not we go with original buffer.
    std::vector<float> multinomial_weights;
    multinomial_weights.reserve(logits.m_size);
    if (logits.is_vector_initialized())
        for (auto& logit: logits.m_vector) multinomial_weights.emplace_back(logit.m_log_prob);
    else
        multinomial_weights.assign(logits.m_data, logits.m_data + logits.m_size);

    // std::discrete_distribution returns corrupted results when applied to log probabilities
    // which result returning NAN only logprobs.
    // so log() is applied after this line
    auto dist = std::discrete_distribution<size_t>(multinomial_weights.begin(), multinomial_weights.end()); // equivalent to multinomial with number of trials == 1

    std::vector<Token> out_tokens;
    for (size_t token_idx = 0; token_idx < num_tokens_per_sequence; ++token_idx) {
        size_t element_to_pick = dist(rng_engine);
        if (logits.is_vector_initialized()) {
            auto logit = logits.m_vector[element_to_pick];
            logit.m_log_prob = std::log(logit.m_log_prob);
            out_tokens.push_back(logit);
        }
        else
            out_tokens.emplace_back(std::log(logits.m_data[element_to_pick]), element_to_pick);
    }
    return out_tokens;
}

std::vector<int64_t> Sampler::_try_finish_generation(SequenceGroup::Ptr & sequence_group) {
    auto sampling_params = sequence_group->get_sampling_parameters();
    std::vector<int64_t> dropped_seq_ids;
    for (auto& running_sequence : sequence_group->get_running_sequences()) {
        const auto generated_len = running_sequence->get_generated_len();
        if (sequence_group->get_max_new_tokens() <= generated_len || 
            is_stop_token_id_hit(running_sequence->get_generated_ids().back(), sampling_params.stop_token_ids) && !sampling_params.ignore_eos) {
            // stop sequence by max_new_tokens or stop token (eos included)
            running_sequence->set_status(SequenceStatus::FINISHED);

            if (is_stop_token_id_hit(running_sequence->get_generated_ids().back(), sampling_params.stop_token_ids) && !sampling_params.ignore_eos) {
                running_sequence->set_finish_reason(GenerationFinishReason::STOP);
            } else if (sequence_group->get_max_new_tokens() == generated_len) {
                running_sequence->set_finish_reason(GenerationFinishReason::LENGTH);
            }

            dropped_seq_ids.push_back(running_sequence->get_id());
            continue;
        }

        if (!sampling_params.stop_strings.empty()) {
            auto& stop_strings = m_stop_strings.at(sequence_group->get_request_id());
            auto match_result = match_stop_string(m_tokenizer, running_sequence->get_generated_ids(), stop_strings,
                                                  sampling_params.include_stop_str_in_output, sequence_group->get_num_tokens_to_validate());
            if (match_result.is_matched) {
                running_sequence->remove_last_tokens(match_result.to_remove);

                running_sequence->set_status(SequenceStatus::FINISHED);
                running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                dropped_seq_ids.push_back(running_sequence->get_id());
            }
        }
    }
    return dropped_seq_ids;
}

void register_new_token(const Token& sampled_token,
                        Sequence::Ptr running_sequence,
                        LogitProcessor& logit_processor,
                        bool is_extend_sequence,
                        bool is_validation_mode_enabled) {
    logit_processor.register_new_generated_token(sampled_token.m_index);
    if (is_extend_sequence) {
        running_sequence->append_token(sampled_token.m_index, sampled_token.m_log_prob);
    }
    if (!is_validation_mode_enabled &&
        logit_processor.get_assistant_confidence_threshold() > 0 &&
        (std::fabs(std::exp(sampled_token.m_log_prob)) < logit_processor.get_assistant_confidence_threshold() || sampled_token.m_log_prob == 0)) {
        auto sequence_group = running_sequence->get_sequence_group_ptr();
        sequence_group->pause_generation(true);
    }
};

std::map<size_t, int32_t> Sampler::get_beam_idxs(SequenceGroup::CPtr sequence_group) {
    size_t request_id = sequence_group->get_request_id();
    auto beam_searcher = m_beam_search_info.find(request_id);
    if (m_beam_search_info.find(request_id) == m_beam_search_info.end()) {
        std::map<size_t, int32_t> beams;
        for (auto& seq : sequence_group->get_running_sequences())
            beams.insert({seq->get_id(), 0});
        return beams;
    }

    return beam_searcher->second.get_beam_idxs();
}

std::list<uint64_t>
create_n_forked_sequences(SequenceGroup::Ptr sequence_group,
                          LogitProcessor& logit_processor,
                          const std::vector<Token>& sampled_tokens) {
    const auto& running_sequences = sequence_group->get_running_sequences();
    OPENVINO_ASSERT(running_sequences.size() == 1);
    Sequence::Ptr sequence_to_fork = running_sequences[0];
    if (sequence_to_fork->get_generated_len() > 0) {
        logit_processor.update_generated_len(0);
        sequence_to_fork->remove_last_tokens(sequence_to_fork->get_generated_len());
    }
    std::list<uint64_t> forked_seq_ids;
    for (size_t i = 1; i < sampled_tokens.size(); ++i) {
        const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
        const auto forked_seq_id = forked_sequence->get_id();
        forked_seq_ids.push_back(forked_seq_id);
        register_new_token(sampled_tokens[i], forked_sequence, logit_processor, true, false);
    }
    return forked_seq_ids;
}

void
stop_sample_tokens(Sequence::Ptr running_sequence,
                   size_t token_idx,
                   size_t max_gen_len,
                   size_t& max_removed_tokens_per_request) {
    running_sequence->remove_last_tokens(token_idx);
    max_removed_tokens_per_request = std::max(max_removed_tokens_per_request, token_idx);
}

void
align_all_sequence_len(SequenceGroup::Ptr& sequence_group,
                       size_t min_generated_tokens,
                       LogitProcessor& logit_processor) {
    for (auto& sequence : sequence_group->get_running_sequences()) {
        const auto generated_token_ids = sequence->get_generated_ids();
        auto generated_len = sequence->get_generated_len();
        if (generated_len > min_generated_tokens) {
            auto removed_token_cnt = generated_len - min_generated_tokens;
            for (size_t i = min_generated_tokens + 1; i < generated_len; ++i) {
                logit_processor.decrease_generated_token_occurance(generated_token_ids[i]);
            }
            sequence->remove_last_tokens(removed_token_cnt);
        }
    }
    logit_processor.update_generated_len(min_generated_tokens);
}

void pad_sequence_lengths(SequenceGroup::Ptr& sequence_group) {
    auto running_sequences = sequence_group->get_running_sequences();
    if (running_sequences.empty()) {
        return;
    }

    size_t max_length = 0;
    for (const auto& seq : running_sequences) {
        max_length = std::max(max_length, seq->get_generated_ids().size());
    }

    for (auto& seq : running_sequences) {
        while (seq->get_generated_ids().size() < max_length) {
            seq->append_token(-1, 0.0f);
        }
    }
}

void adjust_sequence_to_match_path(Sequence::Ptr sequence,
                                   const std::vector<int64_t>& target_path,
                                   size_t common_prefix_len) {
    /*const auto& current_generated_ids = sequence->get_generated_ids();

    if (current_generated_ids.size() > target_path.size()) {
        size_t tokens_to_remove = current_generated_ids.size() - target_path.size();
        sequence->remove_last_tokens(tokens_to_remove);
    }

    if (target_path.size() > current_generated_ids.size()) {
        for (size_t i = current_generated_ids.size(); i < target_path.size(); ++i) {
            sequence->append_token(target_path[i], 0.0f);
        }
    }

    for (size_t i = common_prefix_len; i < target_path.size(); ++i) {
        if (i < current_generated_ids.size()) {
            sequence->update_generated_token(i, target_path[i]);
        } else {
            sequence->append_token(target_path[i], 0.0f);
        }
    }*/
}
Eagle2ValidationResult Sampler::validate_eagle2_tree(const std::vector<std::vector<int64_t>>& candidate_paths,
                                                     const std::vector<std::vector<float>>& candidate_log_probs,
                                                     const std::vector<int> beam_id,
                                                     const ov::Tensor& main_model_logits,
                                                     LogitProcessor& logit_processor,
                                                     bool do_sample) {
    Eagle2ValidationResult result;

    if (candidate_paths.empty()) {
        return result;
    }

    // Find the longest common prefix among all candidate paths
    size_t max_common_prefix = 0;
    if (candidate_paths.size() > 1) {
        max_common_prefix = find_common_prefix_length(candidate_paths); // the first one is generated from main
    } else {
        max_common_prefix = candidate_paths[0].size();
    }
    auto num_tokens_to_process = candidate_paths[0].size();
    Logits sample_p(nullptr, 0);
    // Validate tokens position by position
    for (size_t pos = 0; pos < max_common_prefix; ++pos) {
        // All paths should have the same token at this position due to common prefix
        int64_t candidate_token = candidate_paths[0][pos];
        float candidate_log_prob = candidate_log_probs[0][pos];

        // Get main model logits for this position
        auto logit_vector = _get_logit_vector(main_model_logits, 0, num_tokens_to_process - pos); // since this token is common in all candidates, sample from first beam
        logit_processor.apply(logit_vector);
        sample_p = logit_vector; // save the logits for further sampling
        // Find the candidate token in main model distribution
        auto token_prob_pair = find_token_probability(logit_vector, candidate_token);
        if (!token_prob_pair.first) {
            // Token not found in main model distribution
            result.rejected_at_position = pos;
            break;
        }

        float main_model_log_prob = token_prob_pair.second;

        // Apply acceptance criteria
        bool is_accepted = false;
        if (do_sample) {
            // Speculative sampling: probability ratio test
            float main_model_prob = std::exp(main_model_log_prob);
            float draft_model_prob = 1.0; //std::exp(candidate_log_prob);
            float probability_ratio = std::min(1.0f, main_model_prob / draft_model_prob);

            auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
            float r = dist(rng_engine);
            is_accepted = r <= probability_ratio;
        } else {
            // Greedy validation: exact token match with highest probability
            auto highest_prob_token = get_highest_probability_token(logit_vector);
            is_accepted = (candidate_token == highest_prob_token.m_index);
        }

        if (is_accepted) {
            result.accepted_tokens.push_back(candidate_token);
            result.updated_log_probs.push_back(main_model_log_prob);
            result.accepted_path_length++;
        } else {
            result.rejected_at_position = pos;
            break;
        }
    }
    size_t best_path_idx = 0;
    // If we validated the common prefix successfully, try to extend with one of the paths
    if (result.accepted_path_length == max_common_prefix && candidate_paths.size() > 1) {
        // Select the best path to continue validation beyond common prefix
        // at this point, we either have sample_p for common prefix of beam 0 or none
        best_path_idx = select_best_continuation_path(candidate_paths,
                                                             candidate_log_probs,
                                                             beam_id,
                                                             max_common_prefix,
                                                             sample_p,
                                                             main_model_logits,
                                                             logit_processor);

        // Continue validation on the selected path
        const auto& selected_path = candidate_paths[best_path_idx];
        const auto& selected_log_probs = candidate_log_probs[best_path_idx];

        for (size_t pos = max_common_prefix; pos < selected_path.size(); ++pos) {
            int64_t candidate_token = selected_path[pos];
            float candidate_log_prob = selected_log_probs[pos];

            auto logit_vector = _get_logit_vector(main_model_logits, beam_id[best_path_idx], num_tokens_to_process - pos);
            logit_processor.apply(logit_vector);
            auto token_prob_pair = find_token_probability(logit_vector, candidate_token);
            if (!token_prob_pair.first) {
                result.rejected_at_position = pos;
                sample_p = logit_vector;
                break;
            }

            float main_model_log_prob = token_prob_pair.second;

            bool is_accepted = false;
            if (do_sample) {
                float main_model_prob = std::exp(main_model_log_prob);
                float draft_model_prob = 1.0; //std::exp(candidate_log_prob);
                float probability_ratio = std::min(1.0f, main_model_prob / draft_model_prob);

                auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
                float r = dist(rng_engine);
                is_accepted = r <= probability_ratio;
            } else {
                auto highest_prob_token = get_highest_probability_token(logit_vector);
                is_accepted = (candidate_token == highest_prob_token.m_index);
            }

            if (is_accepted) {
                result.accepted_path_id = beam_id[best_path_idx];
                result.accepted_tokens.push_back(candidate_token);
                result.updated_log_probs.push_back(main_model_log_prob);
                result.accepted_path_length++;
            } else {
                result.rejected_at_position = pos;
                sample_p = logit_vector;
                // reference code update gtp of candidate token to 0?
                break;
            }
        }
    }

    // Generate one additional token if no rejection occurred
    if (result.accepted_path_length == num_tokens_to_process) {
        size_t next_pos = result.accepted_path_length;
        auto logit_vector = _get_logit_vector(main_model_logits, beam_id[best_path_idx], 0);
        logit_processor.apply(logit_vector);

        if (do_sample) {
            auto sampled_tokens = _multinomial_sample(logit_vector, 1);
            result.extra_sampled_token = sampled_tokens[0];
        } else {
            result.extra_sampled_token = _greedy_sample(logit_vector, 0);
        }
    } else if (sample_p.m_data) {
         if (do_sample) {
            auto sampled_tokens = _multinomial_sample(sample_p, 1);
            result.extra_sampled_token = sampled_tokens[0];
        } else {
            result.extra_sampled_token = _greedy_sample(sample_p, 0);
        }
    } else {
        OPENVINO_THROW("should not reach here");
    }
    result.accepted_path_id = beam_id[best_path_idx];
    result.is_path_accepted = (result.accepted_path_length > 0);
    return result;
}

// Helper function to find common prefix length among multiple paths
size_t Sampler::find_common_prefix_length(const std::vector<std::vector<int64_t>>& paths) {
    if (paths.empty())
        return 0;

    size_t min_length = paths[0].size();
    for (const auto& path : paths) {
        min_length = std::min(min_length, path.size());
    }

    size_t common_length = 0;
    for (size_t pos = 0; pos < min_length; ++pos) {
        int64_t first_token = paths[0][pos];
        bool all_match = true;

        for (size_t path_idx = 1; path_idx < paths.size(); ++path_idx) {
            if (paths[path_idx][pos] != first_token) {
                all_match = false;
                break;
            }
        }

        if (all_match) {
            common_length++;
        } else {
            break;
        }
    }

    return common_length;
}

// Helper function to find token probability in logit distribution
std::pair<bool, float> Sampler::find_token_probability(const Logits& logits, int64_t token_id) {
    if (logits.is_vector_initialized()) {
        for (const auto& token : logits.m_vector) {
            if (token.m_index == token_id) {
                return {true, token.m_log_prob};
            }
        }
    } else {
        if (token_id >= 0 && token_id < static_cast<int64_t>(logits.m_size)) {
            // Apply log softmax to get proper log probability
            float max_logit = *std::max_element(logits.m_data, logits.m_data + logits.m_size);
            float log_sum = std::log(std::accumulate(logits.m_data,
                                                     logits.m_data + logits.m_size,
                                                     0.0f,
                                                     [max_logit](float accumulated, float to_add) {
                                                         return accumulated + std::exp(to_add - max_logit);
                                                     }));
            float log_prob = logits.m_data[token_id] - max_logit - log_sum;
            return {true, log_prob};
        }
    }
    return {false, 0.0f};
}

// Helper function to get highest probability token
Token Sampler::get_highest_probability_token(const Logits& logits) {
    if (logits.is_vector_initialized()) {
        auto max_it =
            std::max_element(logits.m_vector.begin(), logits.m_vector.end(), [](const Token& a, const Token& b) {
                return a.m_log_prob < b.m_log_prob;
            });
        return *max_it;
    } else {
        auto max_it = std::max_element(logits.m_data, logits.m_data + logits.m_size);
        size_t max_idx = std::distance(logits.m_data, max_it);

        // Apply log softmax
        float max_logit = *max_it;
        float log_sum = std::log(std::accumulate(logits.m_data,
                                                 logits.m_data + logits.m_size,
                                                 0.0f,
                                                 [max_logit](float accumulated, float to_add) {
                                                     return accumulated + std::exp(to_add - max_logit);
                                                 }));
        float log_prob = max_logit - max_logit - log_sum;

        return Token(log_prob, max_idx);
    }
}

// Helper function to select best continuation path beyond common prefix
size_t Sampler::select_best_continuation_path(const std::vector<std::vector<int64_t>>& candidate_paths,
                                              const std::vector<std::vector<float>>& candidate_log_probs,
                                              const std::vector<int>& beam_id,
                                              const size_t& common_prefix_length,
                                              Logits& logits,
                                              const ov::Tensor& main_model_logits,
                                              LogitProcessor& logit_processor) {
    if (candidate_paths.size() <= 1)
        return 0;

    float best_score = -std::numeric_limits<float>::infinity();
    size_t best_path_idx = 0;
    auto num_tokens_to_process = candidate_paths[0].size();
    for (size_t path_idx = 0; path_idx < candidate_paths.size(); ++path_idx) {
        const auto& path = candidate_paths[path_idx];
        const auto& log_probs = candidate_log_probs[path_idx];

        if (path.size() <= common_prefix_length)
            continue;

        // Score based on next token probability from main model
        int64_t next_token = path[common_prefix_length];
        Logits logit_vector(nullptr, 0);
        if (beam_id[path_idx] == 0 && logits.m_data) {
            logit_vector = logits; // use logits from main model for beam 0
        } else {
            logit_vector = _get_logit_vector(main_model_logits, beam_id[path_idx] , num_tokens_to_process - common_prefix_length);
            logit_processor.apply(logit_vector);
        }

        auto token_prob_pair = find_token_probability(logit_vector, next_token);
        if (token_prob_pair.first) {
            float score = token_prob_pair.second;  // Use main model log probability as score
            if (score > best_score) {
                best_score = score;
                best_path_idx = path_idx;
            }
        }
    }

    return best_path_idx;
}
// Enhanced validation function for EAGLE2 that integrates with existing sampler
int Sampler::validate_eagle2_candidates(SequenceGroup::Ptr seq_group,
                                         const ov::Tensor& main_model_logits,
                                         LogitProcessor& logit_processor,
                                         size_t& accepted_tokens_count,
                                         size_t& max_removed_tokens,
                                         size_t& num_tokens_to_process,
                                         bool do_sample) {
    std::vector<std::vector<int64_t>> candidate_tokens;
    std::vector<std::vector<float>> candidate_log_probs;
    std::vector<int> beam_idxs;
    for (auto& running_sequence : seq_group->get_running_sequences()) {
        auto generated_ids = running_sequence->get_generated_ids();
        size_t start_idx = generated_ids.size() > num_tokens_to_process ? generated_ids.size() - num_tokens_to_process : 0;
        // Extract the tokens to validate
        std::vector<int64_t> tokens_to_validate(generated_ids.begin() + start_idx, generated_ids.end());
        std::vector<float> log_probs_to_validate(running_sequence->get_generated_log_probs().begin() + start_idx,
                                                 running_sequence->get_generated_log_probs().end());
        candidate_tokens.push_back(tokens_to_validate);
        candidate_log_probs.push_back(log_probs_to_validate);
        beam_idxs.push_back([&] (uint64_t seq_id) -> size_t {
            std::vector<Sequence::Ptr> running_seqs = seq_group->get_running_sequences();
            for (size_t seq_global_index = 0; seq_global_index < running_seqs.size(); ++seq_global_index) {
                if (seq_id == running_seqs[seq_global_index]->get_id())
                    return seq_global_index;
            }
            OPENVINO_THROW("should not be here");
        } (running_sequence->get_id()));
    }
    auto validation_result = validate_eagle2_tree(candidate_tokens,
                                                  candidate_log_probs,
                                                  beam_idxs,
                                                  main_model_logits,
                                                  logit_processor,
                                                  do_sample);
    std::cout << "seq group" << seq_group->get_request_id() << " accepted: " << validation_result.accepted_path_length << std::endl;

    if (!validation_result.is_path_accepted) {
        // return false;
        // nothing passed validation, to be further handled, shall we stop the draft pipeline?
    }

    auto selected_sequence = seq_group->get_running_sequences()[validation_result.accepted_path_id];
    // update accepted sequence with the validated tokens

    // Remove any existing generated tokens that weren't accepted
    size_t current_generated_len = selected_sequence->get_generated_len();
    auto num_tokens_to_validate_org = seq_group->get_num_tokens_to_validate();
    auto start_idx = current_generated_len > num_tokens_to_process ? current_generated_len - num_tokens_to_process : 0;
    const auto generated_token_ids = selected_sequence->get_generated_ids();
    if (num_tokens_to_process > validation_result.accepted_path_length) {
        size_t tokens_to_remove = num_tokens_to_process - validation_result.accepted_path_length;
        selected_sequence->remove_last_tokens(tokens_to_remove);
        //logit_processor.update_generated_len(current_generated_len - tokens_to_remove);
        /*for (size_t i = validation_result.accepted_path_length; i < current_generated_len; ++i) {
            logit_processor.decrease_generated_token_occurance(generated_token_ids[i]);
        }*/
        max_removed_tokens = std::max(max_removed_tokens, tokens_to_remove);
    } else if (num_tokens_to_process < num_tokens_to_validate_org) {
        // when validation scheduling is limited due to lack of kv block or max batch size limitation
        size_t tokens_to_remove = num_tokens_to_validate_org -  validation_result.accepted_path_length;
        selected_sequence->remove_last_tokens(tokens_to_remove);
        // fill in the correct max_removed_tokens, which is the removed token in validation stage only
        max_removed_tokens = std::max(max_removed_tokens, num_tokens_to_process - validation_result.accepted_path_length);
    }
    // Add the bonus token with updated probabilities
    selected_sequence->append_token(validation_result.extra_sampled_token.m_index, validation_result.extra_sampled_token.m_log_prob);
    logit_processor.register_new_generated_token(validation_result.extra_sampled_token.m_index);
    return validation_result.accepted_path_id;
}

// need to clear the selector after the request is finished
void Sampler::clear_top_k_selector(uint64_t request_id) {
    auto it = m_top_k_selector_info.find(request_id);
    if (it != m_top_k_selector_info.end()) {
        m_top_k_selector_info.erase(it);
    }
}

Sampler::TopKSelector::TopKSelector(SequenceGroup::Ptr sequence_group, ov::Tensor d2t)
    : m_sequence_group(sequence_group),
        m_parameters{m_sequence_group->get_sampling_parameters()},
        m_d2t(d2t? d2t.data<int64_t>() : nullptr) {
    OPENVINO_ASSERT(m_sequence_group->num_running_seqs() == 1); // for eagle, support 1 running seq at the very beginning
    tree_reset(m_sequence_group);
}

void Sampler::TopKSelector::tree_reset(SequenceGroup::Ptr& sequence_group) {
    m_beams.reserve(m_parameters.eagle_tree_params.branching_factor);
    Beam root_beam((*m_sequence_group)[0]);
    root_beam.m_score = 0.0f;
    m_eagle2_candidate_graph = std::make_shared<Eagle2CandidateGraph>(root_beam,
                                                                        m_parameters.eagle_tree_params.total_tokens,
                                                                        m_parameters.eagle_tree_params.tree_depth);
    m_beams.push_back(root_beam);

}

void Sampler::TopKSelector::finalize_eagle2_candidates(SamplerOutput& sampler_output) {
    auto final_candidates =
        m_eagle2_candidate_graph->get_top_k_candidates();  // currently draft model output wrong candidates
    auto leaf_nodes = m_eagle2_candidate_graph->get_leaf_nodes_from_candidates(final_candidates);
    std::vector<std::vector<int64_t>> retrieve_indices;
    retrieve_indices.reserve(leaf_nodes.size());
    for (const Beam& leaf : leaf_nodes) {
        // Get the path from root to this leaf
        std::vector<int64_t> path = m_eagle2_candidate_graph->get_path_to_node(leaf.m_node_id);
        retrieve_indices.push_back(path);
    }

    // now we have all leaf nodes and their paths, we can update sequences
    // search for existing sequences in sequence group
    auto all_sequences = m_sequence_group->get_sequences();
    std::vector<Sequence::Ptr> available_sequences = all_sequences;
    std::vector<Sequence::Ptr> used_sequences;

    std::set<uint64_t> used_sequence_ids;
    std::vector<std::vector<int64_t>> remaining_retrieve_indices;

    for (size_t i = 0; i < retrieve_indices.size(); ++i) {
        const std::vector<int64_t>& path = retrieve_indices[i];
        bool found_exact_match = false;

        for (auto it = available_sequences.begin(); it != available_sequences.end(); ++it) {
            Sequence::Ptr seq = *it;

            if (used_sequence_ids.count(seq->get_id())) {
                continue;
            }
            const auto& generated_ids = seq->get_generated_ids();
            if (generated_ids.size() < path.size()) {
                continue;  // cannot match if generated ids are shorter than path
            }
            bool is_exact_match = true;
            auto start_idx = generated_ids.size() - path.size();
            for (size_t j = 0; j < path.size(); ++j) {
                if (generated_ids[start_idx + j] != path[j]) {
                    is_exact_match = false;
                    break;
                }
            }

            if (is_exact_match) {
                seq->set_status(SequenceStatus::RUNNING);
                used_sequences.push_back(seq);
                used_sequence_ids.insert(seq->get_id());

                available_sequences.erase(it);
                found_exact_match = true;
                break;
            }
        }

        if (!found_exact_match) {
            remaining_retrieve_indices.push_back(path);
        }
    }
    for (const auto& path : remaining_retrieve_indices) {
        // find best matching sequence in the remaining caching sequences
    }

    pad_sequence_lengths(m_sequence_group);
    // drop all waiting sequences
    auto seqs = m_sequence_group->get_sequences();
    for (auto& seq : seqs) {
        if (seq->is_caching()) { // remaining cached sequences can be now released
            sampler_output.m_dropped_sequences.push_back(seq->get_id());
            m_sequence_group->remove_sequence(seq->get_id());
        }
    }
}

void Sampler::TopKSelector::select_top_k(const ov::Tensor& logits, SamplerOutput& sampler_output) {
    // parent sequence ID -> number of child sequences
    std::map<uint64_t, uint64_t> parent_2_num_childs_map;
    if (m_tree_layer_counter == 0 && m_beams.empty()) {
        tree_reset(m_sequence_group);
    }

    for (Beam& beam : m_beams) {
        sampler_output.num_generated_tokens++;
        uint64_t parent_seq_id = beam.m_sequence->get_id();

        // here we need to map index of sequence in beam search group(s) and sequence group
        beam.m_global_beam_idx = [this](uint64_t seq_id) -> size_t {
            std::vector<Sequence::Ptr> running_seqs = m_sequence_group->get_running_sequences();
            for (size_t seq_global_index = 0; seq_global_index < running_seqs.size(); ++seq_global_index) {
                if (seq_id == running_seqs[seq_global_index]->get_id())
                    return seq_global_index;
            }
            OPENVINO_THROW("Internal error in beam search: should not be here");
        }(parent_seq_id);

        // zero out all parent forks counts
        parent_2_num_childs_map[parent_seq_id] = 0;
    }

    std::vector<Beam> candidates;
    std::vector<Beam> child_beams;                                       // beams for next execution in step()
    candidates.reserve(m_parameters.eagle_tree_params.branching_factor * m_beams.size());  // num_beams for each beam
    m_tree_layer_counter++;
    for (const Beam& beam : m_beams) {
        // do not need log softmax to match paper?
        std::vector<Token> tokens = log_softmax(logits, beam.m_global_beam_idx);

        // sort tokens
        std::sort(tokens.begin(), tokens.end(), [](Token left, Token right) {
            return left.m_log_prob > right.m_log_prob;  // Most probable tokens in front
        });

        size_t add_count = 0;
        for (Token token : tokens) {
            Beam new_candidate = beam;
            new_candidate.m_score += new_candidate.m_log_prob = token.m_log_prob;
            new_candidate.m_token_id = (token.m_index + (m_d2t? m_d2t[token.m_index] : 0));
            m_eagle2_candidate_graph->add_candidate(new_candidate, beam.m_node_id);
            candidates.push_back(new_candidate);
            if (++add_count == m_parameters.eagle_tree_params.branching_factor) {
                break;
            }
        }
    }

    // Sample 2 * group_size highest score tokens to get at least 1 non EOS token per beam
    // OPENVINO_ASSERT(candidates.size() >= 2 * group_size, "No beams left to search");

    std::sort(candidates.begin(), candidates.end(), greater);  // select top k of cumulative probs
    // size_t next_layer_size = std::min(candidates.size(),
    // m_parameters.eagle_tree_width);

    for (size_t cand_idx = 0; cand_idx < m_parameters.eagle_tree_params.branching_factor; ++cand_idx) {
        Beam& candidate = candidates[cand_idx];

        parent_2_num_childs_map[candidate.m_sequence->get_id()] += 1;
        child_beams.push_back(candidate);  // select top beams
    }

    // fork child sequences
    for (Beam& child_beam : child_beams) {
        uint64_t parent_sequence_id = child_beam.m_sequence->get_id();
        uint64_t& num_childs = parent_2_num_childs_map[parent_sequence_id];

        // if current beam is forked multiple times
        if (num_childs > 1) {
            child_beam.m_sequence = m_sequence_group->fork_sequence(child_beam.m_sequence);
            child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);

            // reduce forks count, since fork already happened and next loop iteration
            // will go by the second branch (num_childs == 1)
            --num_childs;

            // fill out sampler output
            sampler_output.m_forked_sequences[parent_sequence_id].push_back(child_beam.m_sequence->get_id());
        } else if (num_childs == 1) {
            // keep current sequence going and add a new token
            child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);
        }
    }

    // drop beams which are de-selected during top-k selection
    for (const Beam& beam : m_beams) {
        size_t num_childs = parent_2_num_childs_map[beam.m_sequence->get_id()];
        if (num_childs == 0) {
            // do not drop, keep for further trace back
            beam.m_sequence->set_status(SequenceStatus::CACHING);
        }
    }

    // child become parents
    m_beams = child_beams;

    // finalize the candidates after depth of iterations of draft pipeline
    if (m_tree_layer_counter == m_parameters.eagle_tree_params.tree_depth + 1) { // to match paper
        for (auto& iter : m_sequence_group->get_running_sequences()) {
            iter->set_status(SequenceStatus::CACHING);
        }
        finalize_eagle2_candidates(sampler_output);
        m_tree_layer_counter = 0;  // reset counter
        m_beams.clear();
        return;
    }
}

Logits Sampler::_get_logit_vector(ov::Tensor logits, size_t batch_idx, size_t token_idx) {
    ov::Shape logits_shape = logits.get_shape();
    size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2];
    OPENVINO_ASSERT(batch_idx <= batch_size);
    OPENVINO_ASSERT(token_idx < seq_len);
    size_t batch_offset = batch_idx * seq_len * vocab_size;
    size_t sequence_offset = (seq_len - token_idx - 1) * vocab_size;
    float* logits_data = logits.data<float>() + batch_offset + sequence_offset;

    return Logits{logits_data, vocab_size};
}

bool Sampler::validate_candidate(
    Sequence::Ptr running_sequence,
    size_t& token_idx,
    Token& sampled_token,
    bool& is_extend_sequence,
    size_t& max_removed_tokens,
    bool do_sample,
    bool has_real_probolities) {
    OPENVINO_ASSERT(token_idx > 0);
    const auto& generated_tokens = running_sequence->get_generated_ids();
    auto it_token_id = generated_tokens.rbegin();
    std::advance(it_token_id, token_idx - 1);

    bool is_candidate_accepted = false;
    // first tokens in case of speculative decoding should be generated by main model
    if (do_sample && has_real_probolities &&
        running_sequence->get_generated_len() != running_sequence->get_sequence_group_ptr()->get_num_tokens_to_validate()) {
        const auto& generated_log_probs = running_sequence->get_generated_log_probs();
        auto it_log_prob = generated_log_probs.rbegin();
        std::advance(it_log_prob, token_idx - 1);

        float p_i = std::exp(*it_log_prob),
                q_i = std::exp(sampled_token.m_log_prob),
                probability_ratio = p_i / q_i;
        
        auto dist = std::uniform_int_distribution<>(0, 100); // equivalent to multinomial with number of trials == 1
        float r_i = dist(rng_engine);
        r_i /= 100;
        is_candidate_accepted = r_i <= probability_ratio;
    } else {
        is_candidate_accepted = *it_token_id == sampled_token.m_index;
    }

    // to validate candidates from assisting model and remove incorrect ones from generated sequence
    if (!is_candidate_accepted) {
        // we need to make resample in speculative sampling, if candidates have real values of logits
        if (do_sample && has_real_probolities) {
            return false;
        }
        running_sequence->remove_last_tokens(token_idx);
        max_removed_tokens = std::max(max_removed_tokens, token_idx);
        is_extend_sequence = true;
        return false;
    } else {
        sampled_token.m_index = *it_token_id;
    }

    return true;
}

float get_p_prime(Sequence::Ptr& running_sequence,
                  const Token& sampled_token,
                  size_t token_offset) {
    auto generated_log_probs = running_sequence->get_generated_log_probs();
    auto it_log_prob = generated_log_probs.rbegin();
    std::advance(it_log_prob, token_offset - 1);

    running_sequence->remove_last_tokens(token_offset);

    float cumulative_prob = 0;
    for (auto& log_prob : running_sequence->get_generated_log_probs()) {
        cumulative_prob += std::exp(log_prob);
    }

    if (cumulative_prob == 0.f) {
        return 1.f;
    }
    
    float p_n = std::exp(sampled_token.m_log_prob),
          q_n = std::exp(*it_log_prob),
          p_prime = std::max(0.f, (p_n - q_n)) / std::log(cumulative_prob);

    return p_prime;
}

std::pair<size_t, std::set<std::string>>
process_stop_strings(const std::set<std::string>& stop_strings, Tokenizer& tokenizer) {
    std::pair<size_t, std::set<std::string>> result;
    for (const auto& stop_string : stop_strings) {
        auto encoded_stop_string = encode_and_process_string(stop_string, tokenizer);
        if (result.first < encoded_stop_string.size()) {
            result.first = encoded_stop_string.size();
        }
        result.second.insert(stop_string);
    }
    return result;
}

SequenceGroupSamplingInfo Sampler::sample_from_sequence_group(SequenceGroup::Ptr sequence_group, ov::Tensor sequence_group_logits, 
                                                              LogitProcessor& logit_processor, const std::pair<size_t, std::set<std::string>>& stop_strings, 
                                                              bool is_validation_mode_enabled) {
    SequenceGroupSamplingInfo sg_sampling_info;
    // Assistant pipeline info is relevant for speculative and prompt lookup decoding
    AssistingPipelineInfo& assisting_pipeline_info = sg_sampling_info.get_assisting_pipeline_info();
    const ov::genai::GenerationConfig& sampling_params = sequence_group->get_sampling_parameters();
    const size_t output_seq_len = sequence_group->get_output_seq_len();
    // get number of tokens to be validated
    size_t num_tokens_to_process = sequence_group->get_num_tokens_to_validate();
    size_t num_generated_tokens_to_validate = num_tokens_to_process;

    if (num_tokens_to_process > output_seq_len - 1) {
        auto delta = num_tokens_to_process - (output_seq_len - 1);
        assisting_pipeline_info.updated_validation_len = std::max(assisting_pipeline_info.updated_validation_len, delta);
        num_tokens_to_process -= delta;
    }

    if (sampling_params.is_greedy_decoding() || sampling_params.is_multinomial()) {
        std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
        size_t num_running_sequences = sequence_group->num_running_seqs();
        if (sampling_params.is_greedy_decoding() && sequence_group->get_num_tokens_to_validate() == 0) {
            OPENVINO_ASSERT(num_running_sequences == 1);
        }
        if (is_validation_mode_enabled && num_generated_tokens_to_validate > 0 ) {
            // trigger group validation for eagle mode
            auto selected_path = validate_eagle2_candidates(sequence_group,
                                       sequence_group_logits,
                                       logit_processor,
                                       sg_sampling_info.sampler_output.num_generated_tokens,
                                       assisting_pipeline_info.max_removed_tokens_per_request,
                                       num_tokens_to_process,
                                       sampling_params.do_sample);
            // drop other sequences
            auto running_sequences = sequence_group->get_running_sequences();
            for (size_t i = 0; i < running_sequences.size(); ++i) {
                if (i != selected_path) {
                    auto& running_sequence = running_sequences[i];
                    running_sequence->set_status(SequenceStatus::FINISHED);
                    running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                    sg_sampling_info.sampler_output.m_dropped_sequences.push_back(running_sequence->get_id());
                    sequence_group->remove_sequence(running_sequence->get_id());
                }
            }
            assisting_pipeline_info.min_generated_len = std::min(assisting_pipeline_info.min_generated_len, sequence_group->get_running_sequences().front()->get_generated_len());
            auto sampling_params = sequence_group->get_sampling_parameters();
            auto running_sequence = sequence_group->get_running_sequences()[0];
            auto sampled_token = running_sequence->get_generated_ids().back();
            for (const auto& dropped_seq_id : _try_finish_generation(sequence_group)) {
                sg_sampling_info.sampler_output.m_dropped_sequences.push_back(dropped_seq_id);
            }
            /*if (is_stop_token_id_hit(sampled_token, sampling_params.stop_token_ids) &&
                !sampling_params.ignore_eos) {
                running_sequence->set_status(SequenceStatus::FINISHED);
                running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                sg_sampling_info.sampler_output.m_dropped_sequences.push_back(running_sequence->get_id());
            }*/
        } else {
            for (size_t running_sequence_id = 0; running_sequence_id < num_running_sequences; ++running_sequence_id) {
                auto& running_sequence = running_sequences[running_sequence_id];
                bool is_validation_passed = true;
                // make `num_tokens_to_process` iteration to validate a candidate generated by `draft_model` + 1
                // iteration to generate one more token by `main_model`
                for (size_t i = 0; i <= num_tokens_to_process; ++i) {
                    if (running_sequence->has_finished())
                        break;
                    sg_sampling_info.sampler_output.num_generated_tokens++;
                    // calculate token offset from the end of logit
                    size_t token_offset = num_tokens_to_process - i;
                    // max counter of needed to be sampled tokens
                    OPENVINO_ASSERT(running_sequence->get_generated_len() >= token_offset);
                    size_t generated_and_verified_len = running_sequence->get_generated_len() - token_offset;
                    OPENVINO_ASSERT(sequence_group->get_max_new_tokens() >= generated_and_verified_len);
                    size_t max_num_sampled_token = sequence_group->get_max_new_tokens() - generated_and_verified_len;
                    if (max_num_sampled_token == 0) {
                        stop_sample_tokens(running_sequence,
                                           token_offset,
                                           max_num_sampled_token,
                                           assisting_pipeline_info.max_removed_tokens_per_request);
                        break;
                    }

                    // do sampling only for token validation/generation.
                    // continue in case of extending draft model sequences by main model generated tokens which
                    // should be taken to KV cache without validation
                    if (!is_validation_mode_enabled && token_offset > 0) {
                        continue;
                    }

                    auto logit_vector = _get_logit_vector(sequence_group_logits, running_sequence_id, token_offset);
                    logit_processor.apply(logit_vector);

                    Token sampled_token;
                    bool is_generate_n_tokens = false;
                    if (sampling_params.is_greedy_decoding()) {
                        sampled_token = {_greedy_sample(logit_vector, sampling_params.logprobs)};
                    } else {
                        // is_multinomial()
                        is_generate_n_tokens = sequence_group->num_total_seqs() == 1;
                        const size_t num_tokens_per_sequence =
                            is_generate_n_tokens ? sampling_params.num_return_sequences : 1;
                        is_generate_n_tokens &= (num_tokens_per_sequence > 1);
                        auto sampled_token_ids = _multinomial_sample(logit_vector, num_tokens_per_sequence);
                        OPENVINO_ASSERT(sampled_token_ids.size(), num_tokens_per_sequence);
                        // to create n sequence just in case of `sequence_group->num_total_seqs() == 1` and
                        // `sampling_params.num_return_sequences > 1`
                        if (is_generate_n_tokens) {
                            const auto forked_seq_ids =
                                create_n_forked_sequences(sequence_group, logit_processor, sampled_token_ids);
                            sg_sampling_info.sampler_output.m_forked_sequences.insert(
                                {running_sequences[0]->get_id(), forked_seq_ids});
                        }
                        sampled_token = sampled_token_ids.front();
                        // make `_speculative_sampling` in case of previous token was not accepted in speculative
                        // decoding
                        if (!is_validation_passed) {
                            float p_prime = get_p_prime(running_sequence, sampled_token, token_offset + 1);
                            assisting_pipeline_info.max_removed_tokens_per_request =
                                std::max(assisting_pipeline_info.max_removed_tokens_per_request, token_offset);
                            // update prob only in case candidate prob > sampled token prob
                            if (p_prime > 0.f) {
                                auto prob = std::exp(sampled_token.m_log_prob);
                                prob /= p_prime;
                                sampled_token.m_log_prob = std::log(prob);
                            }
                        }
                    }
                    // flag to add sampled token to generated sequence or extend logit processors only
                    bool is_extend_sequence = token_offset == 0 || is_generate_n_tokens || !is_validation_passed;
                    if (is_validation_mode_enabled && !is_extend_sequence) {
                        is_validation_passed =
                            validate_candidate(running_sequences[running_sequence_id],
                                               token_offset,
                                               sampled_token,
                                               is_extend_sequence,
                                               assisting_pipeline_info.max_removed_tokens_per_request,
                                               sampling_params.do_sample, !sampling_params.is_prompt_lookup());
                        // doing resample in case of non accepted tokens in specualtive sampling
                        if (!is_validation_passed && sampling_params.do_sample) {
                            continue;
                        }
                        // update log prob just while validation process
                        if (!is_extend_sequence) {
                            OPENVINO_ASSERT(generated_and_verified_len <
                                            running_sequences[running_sequence_id]->get_generated_len());
                            running_sequence->update_generated_log_prob(generated_and_verified_len,
                                                                        sampled_token.m_log_prob);
                        }
                    }
                    register_new_token(sampled_token,
                                       running_sequences[running_sequence_id],
                                       logit_processor,
                                       is_extend_sequence,
                                       is_validation_mode_enabled);
                    // to exit from sampling in case of failed token validation
                    if (!is_validation_passed) {
                        break;
                    } else {
                        auto sampling_params = sequence_group->get_sampling_parameters();
                        if (is_stop_token_id_hit(sampled_token.m_index, sampling_params.stop_token_ids) &&
                            !sampling_params.ignore_eos) {
                            running_sequence->set_status(SequenceStatus::FINISHED);
                            running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                            sg_sampling_info.sampler_output.m_dropped_sequences.push_back(running_sequence->get_id());
                        }
                    }
                }
                assisting_pipeline_info.min_generated_len =
                    std::min(assisting_pipeline_info.min_generated_len, running_sequence->get_generated_len());
            }
            align_all_sequence_len(sequence_group, assisting_pipeline_info.min_generated_len, logit_processor);
            for (const auto& dropped_seq_id : _try_finish_generation(sequence_group)) {
                sg_sampling_info.sampler_output.m_dropped_sequences.push_back(dropped_seq_id);
            }
        }
    } else if (sampling_params.is_eagle_tree()) {
        TopKSelector* topk_searcher;
        {
            uint64_t request_id = sequence_group->get_request_id();
            std::lock_guard<std::mutex> lock(m_beam_search_info_mutex);
            if (m_top_k_selector_info.find(request_id) == m_top_k_selector_info.end()) {
                m_top_k_selector_info.emplace(request_id, TopKSelector(sequence_group, m_d2t->get_tensor_view()));
            }
            topk_searcher = &m_top_k_selector_info.at(request_id);
        }
            topk_searcher->select_top_k(sequence_group_logits, sg_sampling_info.sampler_output);
    
    } else if (sampling_params.is_beam_search()) {
        uint64_t request_id = sequence_group->get_request_id();

        // create beam search info if we are on the first generate
        GroupBeamSearcher* beam_searcher;
        {
            std::lock_guard<std::mutex> lock(m_beam_search_info_mutex);
            if (m_beam_search_info.find(request_id) == m_beam_search_info.end()) {
                m_beam_search_info.emplace(request_id, GroupBeamSearcher(sequence_group, m_tokenizer));
            }
            beam_searcher = &m_beam_search_info.at(request_id);
        }

        // current algorithm already adds new tokens to running sequences and
        beam_searcher->select_next_tokens(sequence_group_logits, sg_sampling_info.sampler_output, stop_strings);

        // check max length stop criteria
        std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
        if (!sequence_group->has_finished() &&
            running_sequences[0]->get_generated_len() == sequence_group->get_max_new_tokens()) {
            // stop sequence by max_new_tokens
            beam_searcher->finalize(sg_sampling_info.sampler_output);
        }
    }
    // Notify handle after sampling is done. 
    // For non-streaming this is effective only when the generation is finished.
    OPENVINO_ASSERT(num_generated_tokens_to_validate >= assisting_pipeline_info.max_removed_tokens_per_request);
    sequence_group->notify_handle();
    return sg_sampling_info;
}

SamplerOutput Sampler::sample(const std::vector<SequenceGroup::Ptr> & sequence_groups,
                              ov::Tensor logits,
                              bool is_validation_mode_enabled) {
    const float * logits_data = logits.data<float>();
    ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    size_t vocab_size = logits_shape[2];

    SamplerOutput sampler_output;
    std::unordered_map<uint64_t, std::future<SequenceGroupSamplingInfo>> sg_sampling_future_map;
    for (size_t sequence_group_id = 0, currently_processed_tokens = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
        SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
        if (!sequence_group->is_scheduled())
            continue;

        const size_t num_running_sequences = sequence_group->num_running_seqs();
        const size_t output_seq_len = sequence_group->get_output_seq_len();
        const ov::genai::GenerationConfig& sampling_params = sequence_group->get_sampling_parameters();

        const auto request_id = sequence_group->get_request_id();
        if (!m_logit_processors.count(request_id)) {
            if (!m_structured_output_controller) {
                m_structured_output_controller = std::make_shared<StructuredOutputController>(m_tokenizer, vocab_size);
            }
            m_logit_processors.insert({request_id, LogitProcessor(sampling_params, sequence_group->get_prompt_ids(), m_structured_output_controller)});
        }
        if (!m_stop_strings.count(request_id)) {
            auto processed_stop_string = process_stop_strings(sampling_params.stop_strings, m_tokenizer);
            m_stop_strings.insert({request_id, processed_stop_string});
            sequence_group->set_stream_window_size(processed_stop_string.first);
        }
        const auto& stop_strings = m_stop_strings.at(request_id);
        auto& logit_processor = m_logit_processors.at(request_id);
        const void * sequence_group_logits_data = logits_data + vocab_size * currently_processed_tokens;
        ov::Tensor sequence_group_logits(ov::element::f32, ov::Shape{num_running_sequences, output_seq_len, vocab_size}, (void *)sequence_group_logits_data);
        if (sequence_group->requires_sampling()) {
            // Call sample_from_sequence_group asynchronously
            sg_sampling_future_map[request_id] = m_thread_pool.submit(&Sampler::sample_from_sequence_group, this, sequence_group, sequence_group_logits,
                                                                      logit_processor, stop_strings, is_validation_mode_enabled);
        } else {
            // we are in prompt processing phase when prompt is split into chunks and processed step by step
        }
        // accumulate a number of processed tokens
        currently_processed_tokens += output_seq_len * num_running_sequences;
    }

    // Update sequence groups internal states after sampling is done
    for (auto& sequence_group : sequence_groups) {
        if (!sequence_group->is_scheduled())
            continue;
        SequenceGroupSamplingInfo sg_sampling_info;
        const auto request_id = sequence_group->get_request_id();
        if (sg_sampling_future_map.find(request_id) != sg_sampling_future_map.end()) {
            // If there is a future assigned to a sequence group we read it's result (blocking if results not available yet)
            sg_sampling_info = sg_sampling_future_map[request_id].get();
            sampler_output.num_generated_tokens += sg_sampling_info.sampler_output.num_generated_tokens;

            // Merge sampler output from sequence group to the main one
            sampler_output.m_dropped_sequences.insert(
                sampler_output.m_dropped_sequences.end(),
                sg_sampling_info.sampler_output.m_dropped_sequences.begin(),
                sg_sampling_info.sampler_output.m_dropped_sequences.end()
            );

            for (const auto& forked_seq : sg_sampling_info.sampler_output.m_forked_sequences) {
                sampler_output.m_forked_sequences[forked_seq.first].insert(
                    sampler_output.m_forked_sequences[forked_seq.first].end(),
                    forked_seq.second.begin(),
                    forked_seq.second.end()
                );
            }
        }
        // NOTE: it should be before 'get_num_scheduled_tokens' is used
        // update internal state of sequence group to reset scheduler tokens and update currently processed ones
        const AssistingPipelineInfo& assisting_pipeline_info = std::as_const(sg_sampling_info.get_assisting_pipeline_info());
        sequence_group->finish_iteration();
        // decrease sequence_group context in case of candidates generated by draft_model were not accepted by main_model
        if (assisting_pipeline_info.max_removed_tokens_per_request) {
            auto min_processed_tokens = sequence_group->get_prompt_len() + assisting_pipeline_info.min_generated_len - 1;
            sequence_group->update_processed_tokens_num(min_processed_tokens);
            auto& logit_processor = get_logit_processor(sequence_group->get_request_id());
            logit_processor.update_generated_len(min_processed_tokens);
        }
        if (assisting_pipeline_info.updated_validation_len) {
            sequence_group->set_num_validated_tokens(assisting_pipeline_info.updated_validation_len);
        }
    }
    return sampler_output;
}

LogitProcessor& Sampler::get_logit_processor(uint64_t request_id) {
    OPENVINO_ASSERT(m_logit_processors.count(request_id));
    return m_logit_processors.at(request_id);
}


void Sampler::create_logit_processor(uint64_t request_id, const GenerationConfig& sampling_params, const TokenIds& prompt) {
    if (!m_structured_output_controller) {
        // We don't have vocab size (actually logits size) and also we don't have access to the logits.
        // vocab size will be taken from the tokenizer during LogitProcessor initialization.
        m_structured_output_controller = std::make_shared<StructuredOutputController>(m_tokenizer);
    }

    m_logit_processors.insert({request_id, LogitProcessor(sampling_params, prompt, m_structured_output_controller)});
}

void Sampler::clear_request_info(uint64_t request_id) {
    m_beam_search_info.erase(request_id);
    m_logit_processors.erase(request_id);
    m_stop_strings.erase(request_id);
}

int64_t Sampler::GroupBeamSearcher::Group::finish(Beam beam, const ov::genai::GenerationConfig& sampling_params) {
    int64_t preeempted_sequence_id = -1;
    float generated_len = beam.get_generated_len() + (is_stop_token_id_hit(beam.m_token_id, sampling_params.stop_token_ids) ? 1 : 0); // HF counts EOS token in generation length
    beam.m_score /= std::pow(generated_len, sampling_params.length_penalty);

    min_heap.push_back(beam);
    std::push_heap(min_heap.begin(), min_heap.end(), greater);
    assert(sampling_params.num_beams % sampling_params.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = sampling_params.num_beams / sampling_params.num_beam_groups;
    if (min_heap.size() > group_size) {
        std::pop_heap(min_heap.begin(), min_heap.end(), greater);
        preeempted_sequence_id = min_heap.back().m_sequence->get_id();
        min_heap.pop_back();
    }

    return preeempted_sequence_id;
}

void Sampler::GroupBeamSearcher::Group::is_done() {
    const auto sequence_group = ongoing.front().m_sequence->get_sequence_group_ptr();
    const ov::genai::GenerationConfig& sampling_params = sequence_group->get_sampling_parameters();

    assert(sampling_params.num_beams % sampling_params.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = sampling_params.num_beams / sampling_params.num_beam_groups;
    if (min_heap.size() < group_size)
        return;

    const Beam& best_running_sequence = ongoing.front(), & worst_finished_sequence = min_heap.front();
    size_t cur_len = best_running_sequence.m_sequence->get_generated_len();
    float best_sum_logprobs = best_running_sequence.m_score;
    float worst_score = worst_finished_sequence.m_score;
    switch (sampling_params.stop_criteria) {
    case ov::genai::StopCriteria::EARLY:
        done = true;
        return;
    case ov::genai::StopCriteria::HEURISTIC: {
        float highest_attainable_score = best_sum_logprobs / std::pow(float(cur_len), sampling_params.length_penalty);
        done = worst_score >= highest_attainable_score;
        return;
    }
    case ov::genai::StopCriteria::NEVER: {
        size_t length = sampling_params.length_penalty > 0.0 ? sequence_group->get_max_new_tokens() : cur_len;
        float highest_attainable_score = best_sum_logprobs / std::pow(float(length), sampling_params.length_penalty);
        done = worst_score >= highest_attainable_score;
        return;
    }
    default:
        OPENVINO_THROW("Beam search internal error: unknown mode");
    }
}
}
