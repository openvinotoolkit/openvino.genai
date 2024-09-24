
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <map>
#include <algorithm>
#include <cmath>
#include <random>
#include <set>

#include "openvino/runtime/tensor.hpp"

#include "logit_processor.hpp"
#include "scheduler.hpp"
#include "sequence_group.hpp"

namespace ov::genai {
// Modifyed Knuth–Morris–Pratt algorithm which returns tokens following after every needle occurance in haystack
inline std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, const std::vector<int64_t>& needle) {
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

inline std::vector<Token> log_softmax(const ov::Tensor& logits, size_t batch_idx) {
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

inline std::vector<int64_t>
wrap_tokens(const std::vector<int64_t>& tokens, const std::vector<int64_t>& prefix_tokens, const std::vector<int64_t>& suffix_tokens) {
    std::vector<int64_t> all_tokens = prefix_tokens;
    all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
    all_tokens.insert(all_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
    return all_tokens;
}

inline std::string clean_wrapped_text(const std::string& wrapped_text, const std::string& prefix, const std::string& suffix) {
    auto prefix_pos = wrapped_text.find(prefix);
    OPENVINO_ASSERT(prefix_pos != std::string::npos);
    auto suffix_pos = wrapped_text.rfind(suffix);
    OPENVINO_ASSERT(suffix_pos != std::string::npos);
    auto clean_text_start = prefix_pos + prefix.size();
    auto clean_text_length = suffix_pos - clean_text_start;
    std::string clean_text = wrapped_text.substr(clean_text_start, clean_text_length);
    return clean_text;
}

// Return number of last tokens that match one of the stop_strings. If there's no match 0 is returned.
inline int
match_stop_string(Tokenizer & tokenizer, const TokenIds & generated_tokens, const std::set<std::string> & stop_strings) {
    /*
    For catching stop_string hit we run comparisons character-wise to catch cases where stop string 
    overlaps with part of another token on both sides or is just a part of a single token. 
    For every stop_string we iterate over generated tokens starting from the last one and going backwards. 
    Every token is wrapped with prefix tokens to ensure tokenizer doesn't remove prefix whitespace of the actual token.
    After that all tokens are decoded and prefix is removed from the decoded text, so we end up with decoded token.
    Its characters are compared to the stop_string character at a current_position 
    (position of a character in the stop_string counting from the last one) - at the begining position is 0.
    When characters match we increase current_position and check if we have a full match already, if not we continue.
    If we have already matched some characters (current_position > 0) and next character is not matching 
    before we reach the full match, then we reset current_position to 0. 
    */ 
    std::string prefix = "a";
    auto prefix_ov = tokenizer.encode(prefix).input_ids;
    std::vector<int64_t> prefix_tokens(prefix_ov.data<int64_t>(), prefix_ov.data<int64_t>() + prefix_ov.get_size());
    std::string suffix = "b";
    auto suffix_ov = tokenizer.encode(suffix).input_ids;
    std::vector<int64_t> suffix_tokens(suffix_ov.data<int64_t>(), suffix_ov.data<int64_t>() + suffix_ov.get_size());

    // Since whitespace can be added at the beginning of the suffix we also try to capture that behavior here
    // and get suffix string that will actually be part of the decoded string so we can remove it correctly
    auto wrapped_suffix_tokens = suffix_tokens;
    wrapped_suffix_tokens.insert(wrapped_suffix_tokens.begin(), prefix_tokens.begin(), prefix_tokens.end());
    std::string wrapped_suffix = tokenizer.decode(wrapped_suffix_tokens);
    auto wrapper_pos = wrapped_suffix.find(prefix);
    suffix = wrapped_suffix.substr(wrapper_pos + prefix.size());
    
    for (auto stop_string: stop_strings) {
        int current_position = 0;
        int num_matched_tokens = 0; 
        // Getting reverse iterator to check tokens starting from the last one generated and going backwards
        auto generated_tokens_rit = generated_tokens.rbegin();
        std::vector<int64_t> tokens_buffer;
        while (generated_tokens_rit != generated_tokens.rend()) {
            num_matched_tokens++;
            tokens_buffer.insert(tokens_buffer.begin(), *generated_tokens_rit);

            std::vector<int64_t> wrapped_tokens = wrap_tokens(tokens_buffer, prefix_tokens, suffix_tokens);
            std::string wrapped_text = tokenizer.decode(wrapped_tokens);
            std::string clean_text = clean_wrapped_text(wrapped_text, prefix, suffix);

            if (clean_text == "" || (clean_text.size() >= 3 && (clean_text.compare(clean_text.size() - 3, 3, "�") == 0))) { 
                generated_tokens_rit++;
                continue;
            } else {
                tokens_buffer.clear();
            }
            // Checking clean_text characters starting from the last one
            for (auto clean_text_rit = clean_text.rbegin(); clean_text_rit != clean_text.rend(); clean_text_rit++) {
                // On character match increment current_position for the next comparisons
                if (*clean_text_rit == *(stop_string.rbegin() + current_position)) {
                    current_position++;
                    // If this is the last character from the stop_string we have a match
                    if ((stop_string.rbegin() + current_position) == stop_string.rend()) {
                        return num_matched_tokens;
                    } 
                } else if (current_position) {
                    // Already found matching characters, but the last one didn't match, so we reset current_position
                    current_position = 0;
                    // Looking for the match will start over from this character so we decrement iterator
                    clean_text_rit--;
                }
            }
            generated_tokens_rit++;
        }
    }
    return 0;
}

// Return number of last tokens that match one of the stop_strings. If there's no match 0 is returned.
// Number of tokens might not be exact as if there's no direct token match, we decode generated tokens incrementally expanding decoding scope
// with 4 next tokens with each iteration until we check all tokens.
inline int match_stop_string2(Tokenizer & tokenizer, const TokenIds & generated_tokens, const std::set<std::string> & stop_strings) {
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

// Handle stop_token_ids
inline bool is_stop_token_id_hit(int64_t generated_token, const std::set<int64_t> & stop_token_ids) {
    for (auto & stop_token_id : stop_token_ids) {
        if (generated_token == stop_token_id)
            return true;
    }
    return false;
}

struct Beam {
    Sequence::Ptr m_sequence;
    size_t m_global_beam_idx = 0;

    // beam is made on top of sequence
    float m_log_prob = 0.0f;
    int64_t m_token_id = -1;

    // cumulative log probabilities
    float m_score = -std::numeric_limits<float>::infinity();

    Beam(Sequence::Ptr sequence)
        : m_sequence(std::move(sequence)) { }

    size_t get_generated_len() const {
        return m_sequence->get_generated_len();
    }
};

inline bool greater(const Beam& left, const Beam& right) {
    return left.m_score > right.m_score;
}

struct Group {
    std::vector<Beam> ongoing;  // Best beams in front
    std::vector<Beam> min_heap;  // The worst of the best completed beams is the first
    bool done = false;

    int64_t finish(Beam beam, const ov::genai::GenerationConfig& sampling_params) {
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

    void is_done(const ov::genai::GenerationConfig& sampling_params) {
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
            size_t length = sampling_params.length_penalty > 0.0 ? sampling_params.max_new_tokens : cur_len;
            float highest_attainable_score = best_sum_logprobs / std::pow(float(length), sampling_params.length_penalty);
            done = worst_score >= highest_attainable_score;
            return;
        }
        default:
            OPENVINO_THROW("Beam search internal error: unkown mode");
        }
    }
};

struct SamplerOutput {
    // IDs of sequences that need to be dropped
    std::vector<uint64_t> m_dropped_sequences;
    // IDs of sequences that need to be forked (note, the same sequence can be forked multiple times)
    // it will later be used by scheduler to fork block_tables for child sequences
    std::unordered_map<uint64_t, std::list<uint64_t>> m_forked_sequences;
};

class GroupBeamSearcher {
    SequenceGroup::Ptr m_sequence_group;
    ov::genai::GenerationConfig m_parameters;
    std::vector<Group> m_groups;
    Tokenizer m_tokenizer;
public:
    explicit GroupBeamSearcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer);

    void select_next_tokens(const ov::Tensor& logits, SamplerOutput& sampler_output);

    void finalize(SamplerOutput& sampler_output) {
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
};

class Sampler {

    Logits _get_logit_vector(ov::Tensor logits, size_t batch_idx = 1) {
        ov::Shape logits_shape = logits.get_shape();
        size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2];
        OPENVINO_ASSERT(batch_idx <= batch_size);
        size_t batch_offset = batch_idx * seq_len * vocab_size;
        size_t sequence_offset = (seq_len - 1) * vocab_size;
        float* logits_data = logits.data<float>() + batch_offset + sequence_offset;

        return Logits{logits_data, vocab_size};
    }

    Token _greedy_sample(const Logits& logits) const {
        // For greedy sampling we do not expect sorting or shrinking considered tokens
        // so we can operate directly on the data buffer
        float max_value = -std::numeric_limits<float>::infinity();
        size_t max_index = 0;
        for (size_t i = 0; i < logits.m_size; ++i) {
            if (logits.m_data[i] > max_value) {
                max_value = logits.m_data[i];
                max_index = i;
            }
        }

        // apply log softmax to max value
        float log_sum = std::log(std::accumulate(
            logits.m_data, logits.m_data + logits.m_size, 0.0f, [max_value](float accumulated, float to_add) {
                return accumulated + std::exp(to_add - max_value);
        }));
        max_value = -log_sum;

        return Token(max_value, max_index);
    }

    std::vector<Token> _multinomial_sample(const Logits& logits, size_t num_tokens_per_sequence) {
        // If top_p or top_k was applied we use sorted vector, if not we go with original buffer.
        std::vector<float> multinomial_weights;
        multinomial_weights.reserve(logits.m_size);
        if (logits.is_vector_initialized())
            for (auto& logit: logits.m_vector) multinomial_weights.emplace_back(logit.m_log_prob);
        else
            multinomial_weights.assign(logits.m_data, logits.m_data + logits.m_size);

        auto dist = std::discrete_distribution<size_t>(multinomial_weights.begin(), multinomial_weights.end()); // equivalent to multinomial with number of trials == 1
        
        std::vector<Token> out_tokens;
        for (size_t token_idx = 0; token_idx < num_tokens_per_sequence; ++token_idx) {
            size_t element_to_pick = dist(rng_engine);
            if (logits.is_vector_initialized())
                out_tokens.push_back(logits.m_vector[element_to_pick]);
            else
                out_tokens.emplace_back(logits.m_data[element_to_pick], element_to_pick);
        }
        return out_tokens;
    }

    std::vector<int64_t> _try_finish_generation(SequenceGroup::Ptr & sequence_group) {
        auto sampling_params = sequence_group->get_sampling_parameters();
        std::vector<int64_t> dropped_seq_ids;
        for (auto& running_sequence : sequence_group->get_running_sequences()) {
            const auto generated_len = running_sequence->get_generated_len();
            if (sampling_params.max_new_tokens == generated_len || 
                is_stop_token_id_hit(running_sequence->get_generated_ids().back(), sampling_params.stop_token_ids) && !sampling_params.ignore_eos) {
                // stop sequence by max_new_tokens or stop token (eos included)
                running_sequence->set_status(SequenceStatus::FINISHED);

                if (is_stop_token_id_hit(running_sequence->get_generated_ids().back(), sampling_params.stop_token_ids) && !sampling_params.ignore_eos) {
                    running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                } else if (sampling_params.max_new_tokens == generated_len) {
                    running_sequence->set_finish_reason(GenerationFinishReason::LENGTH);
                }
                
                dropped_seq_ids.push_back(running_sequence->get_id());
                continue;
            }

            if (!sampling_params.stop_strings.empty()) {
                int num_matched_last_tokens = match_stop_string(m_tokenizer, running_sequence->get_generated_ids(), sampling_params.stop_strings);
                if (num_matched_last_tokens) {
                    if (!sampling_params.include_stop_str_in_output)
                        running_sequence->remove_last_tokens(num_matched_last_tokens);
                    running_sequence->set_status(SequenceStatus::FINISHED);
                    running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                    dropped_seq_ids.push_back(running_sequence->get_id());
                }
            }
        }
        return dropped_seq_ids;
    }

    // request ID => beam search tracking information
    std::map<uint64_t, GroupBeamSearcher> m_beam_search_info;

    std::mt19937 rng_engine;
    // { request_id, logit_processor }
    std::map<uint64_t, LogitProcessor> m_logit_processors;

    Tokenizer m_tokenizer;

public:

    Sampler(Tokenizer & tokenizer) : m_tokenizer(tokenizer) {};

    SamplerOutput sample(std::vector<SequenceGroup::Ptr> & sequence_groups, ov::Tensor logits);

    void set_seed(size_t seed) { rng_engine.seed(seed); }

    void clear_beam_search_info(uint64_t request_id);
};

}
