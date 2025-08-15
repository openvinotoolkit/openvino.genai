
// Copyright (C) 2023-2025 Intel Corporation
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

#include "sampling/logit_transformers.hpp"
#include "sampling/logit_processor.hpp"
#include "continuous_batching/scheduler.hpp"
#include "sequence_group.hpp"
#include "threadpool.hpp"
#include "sampling/structured_output/structured_output_controller.hpp"

namespace ov::genai {
// Handle stop_token_ids
inline bool is_stop_token_id_hit(int64_t generated_token, const std::set<int64_t> & stop_token_ids) {
    for (auto & stop_token_id : stop_token_ids) {
        if (generated_token == stop_token_id)
            return true;
    }
    return false;
}

inline bool is_stop_token_id_hit_in_sequence_group(SequenceGroup::Ptr sequence_group, const std::set<int64_t>& stop_token_ids) {
    for (auto& sequence : sequence_group->get_running_sequences()) {
        const TokenIds& generated_tokens = sequence->get_generated_ids();
        if (!generated_tokens.empty() && is_stop_token_id_hit(generated_tokens.back(), stop_token_ids)) {
            return true;
        }
    }
    return false;
}

std::vector<Token> log_softmax(const ov::Tensor& logits, size_t batch_idx);

struct SamplerOutput {
    // IDs of sequences that need to be dropped
    std::vector<uint64_t> m_dropped_sequences;
    // IDs of sequences that need to be forked (note, the same sequence can be forked multiple times)
    // it will later be used by scheduler to fork block_tables for child sequences
    std::unordered_map<uint64_t, std::list<uint64_t>> m_forked_sequences;
    // store number of generated_tokens
    size_t num_generated_tokens = 0;
};

struct AssistingPipelineInfo {
    size_t max_removed_tokens_per_request = 0; 
    size_t min_generated_len = std::numeric_limits<size_t>::max();
    size_t updated_validation_len = 0;
};

struct SequenceGroupSamplingInfo {
    SamplerOutput sampler_output;
    AssistingPipelineInfo assisting_pipeline_info;

    AssistingPipelineInfo& get_assisting_pipeline_info() {
        return assisting_pipeline_info;
    }
};

class Sampler {
    class GroupBeamSearcher;

    Logits _get_logit_vector(ov::Tensor logits, size_t batch_idx, size_t token_idx);
    Token _greedy_sample(const Logits& logits, size_t top_logprobs) const;
    std::vector<Token> _multinomial_sample(const Logits& logits, size_t num_tokens_per_sequence);
    std::vector<int64_t> _try_finish_generation(SequenceGroup::Ptr & sequence_group);

    bool validate_candidate(Sequence::Ptr running_sequence, size_t& token_idx, Token& sampled_token,
                            bool& is_extend_sequence, size_t& max_removed_tokens, bool do_sample, bool has_real_probolities);

    SequenceGroupSamplingInfo sample_from_sequence_group(SequenceGroup::Ptr sequence_group, ov::Tensor sequence_group_logits,
                                                        LogitProcessor& logit_processor, const std::pair<size_t, std::set<std::string>>& stop_strings,
                                                        bool is_validation_mode_enabled);

    // request ID => beam search tracking information
    std::map<uint64_t, GroupBeamSearcher> m_beam_search_info;
    std::mutex m_beam_search_info_mutex;

    std::mt19937 rng_engine;
    size_t seed = rng_engine.default_seed;
    // { request_id, logit_processor }
    std::map<uint64_t, LogitProcessor> m_logit_processors;
    // { request_id, { max_encoded_len, { stop_strings }}}
    std::map<int64_t, std::pair<size_t, std::set<std::string>>> m_stop_strings;

    Tokenizer m_tokenizer;

    ThreadPool m_thread_pool;
public:
    Sampler(const Sampler& rhs) = delete;
    Sampler(Sampler&& rhs) = delete;
    Sampler(size_t num_threads = 1): m_thread_pool(num_threads) {};
    explicit Sampler(const Tokenizer & tokenizer, size_t num_threads = 1) : m_tokenizer(tokenizer), m_thread_pool(num_threads) {};

    SamplerOutput sample(const std::vector<SequenceGroup::Ptr> & sequence_groups, ov::Tensor logits, bool is_validation_mode_enabled = false);
    void set_seed(size_t new_seed) {
        rng_engine.seed(new_seed);
        seed = new_seed;
    }
    size_t get_seed() { return seed; }

    void set_tokenizer(const Tokenizer& tokenizer) {
        m_tokenizer = tokenizer;
    }

    void clear_request_info(uint64_t request_id);

    LogitProcessor& get_logit_processor(uint64_t request_id);
    void create_logit_processor(uint64_t request_id, const GenerationConfig& sampling_parameters, const TokenIds& prompt);

    std::map<size_t, int32_t> get_beam_idxs(SequenceGroup::CPtr sequence_group);
    // pair with map with backend name and corresponding compiler init time, and vector of compile times for each concrete grammar
    std::pair<std::map<std::string, float>, std::vector<float>> get_structured_output_times();
    void clear_structured_output_compile_times();
};

class Sampler::GroupBeamSearcher {
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

    static bool greater(const Beam& left, const Beam& right) {
        return left.m_score > right.m_score;
    }

    struct Group {
        std::vector<Beam> ongoing;  // Best beams in front
        std::vector<Beam> min_heap;  // The worst of the best completed beams is the first
        bool done = false;

        int64_t finish(Beam beam, const ov::genai::GenerationConfig& sampling_params);
        void is_done();
    };

    SequenceGroup::Ptr m_sequence_group;
    ov::genai::GenerationConfig m_parameters;
    std::vector<Group> m_groups;
    Tokenizer m_tokenizer;
public:
    explicit GroupBeamSearcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer);

    void select_next_tokens(const ov::Tensor& logits, SamplerOutput& sampler_output, const std::pair<size_t, std::set<std::string>>& stop_strings);
    void finalize(SamplerOutput& sampler_output);
    std::map<size_t, int32_t> get_beam_idxs();
};
}
