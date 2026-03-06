
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "continuous_batching/scheduler.hpp"
#include "openvino/runtime/tensor.hpp"
#include "sampling/logit_processor.hpp"
#include "sampling/logit_transformers.hpp"
#include "sampling/structured_output/structured_output_controller.hpp"
#include "sequence_group.hpp"
#include "threadpool.hpp"

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

// Tree of draft token candidates for EAGLE speculative decoding.
// Stores token IDs and cumulative scores without any KV-cache or Sequence references.
class EagleCandidateGraph {
public:
    struct Node {
        uint64_t id = 0;
        int64_t token_id = -1;
        float score = -std::numeric_limits<float>::infinity();
        int tree_layer = 0;
    };

    EagleCandidateGraph(int64_t root_token_id, float root_score, int max_tokens, int max_depth);

    // Adds a child of parent_node_id. Returns the new node's ID, or 0 if beyond max_depth.
    uint64_t add_node(int64_t token_id, float score, uint64_t parent_node_id);

    // Returns at most (max_candidate_nodes + 1) top-scoring nodes (root always included), sorted by tree layer.
    std::vector<Node> select_candidate_nodes() const;

    // From a set of nodes, returns those with no children present in the set (leaf nodes).
    std::vector<Node> get_leaf_nodes(const std::vector<Node>& selected) const;

    // Returns the path from root to node_id as a sequence of node_ids (root first).
    std::vector<uint64_t> get_path_to_node(uint64_t node_id) const;

    // Returns true if ancestor_id is an ancestor of node_id (self-inclusive).
    bool is_ancestor(uint64_t ancestor_id, uint64_t node_id) const;

private:
    struct InternalNode {
        Node data;
        uint64_t parent_id = 0;
        std::vector<uint64_t> child_ids;
        std::unordered_set<uint64_t> ancestor_ids;  // self-inclusive; enables O(1) is_ancestor
    };

    std::unordered_map<uint64_t, InternalNode> m_nodes;
    uint64_t m_next_node_id = 1;
    int m_max_candidate_nodes;
    int m_max_depth;
};

class Sampler {
    class Searcher;
    class TreeSearcher;
    class GroupBeamSearcher;

    // Bundles all per-request sampler state that needs to be accessed during sampling. 
    // This is used to keep track of logit processors, RNG engines, and stop strings for each request.
    struct RequestSamplerContext {
        std::mt19937 rng_engine;
        LogitProcessor logit_processor;
        // { max_encoded_len, stop_strings }
        std::pair<size_t, std::set<std::string>> stop_strings;

        RequestSamplerContext(size_t seed, LogitProcessor&& lp)
            : rng_engine(seed), logit_processor(std::move(lp)) {}
    };

    Logits _get_logit_vector(ov::Tensor logits, size_t batch_idx, size_t token_idx);
    Token _greedy_sample(const Logits& logits, size_t top_logprobs) const;
    std::vector<Token> _multinomial_sample(const Logits& logits, size_t num_tokens_per_sequence, std::mt19937& rng_engine);
    std::vector<int64_t> _try_finish_generation(SequenceGroup::Ptr& sequence_group,
                                                 const std::pair<size_t, std::set<std::string>>& stop_strings);

    bool validate_candidate(Sequence::Ptr running_sequence, size_t& token_idx, Token& sampled_token,
                            bool& is_extend_sequence, size_t& max_removed_tokens, bool do_sample, bool has_real_probabilities,
                            std::mt19937& rng_engine);

    // Validate tree results from the target model using retrieve_indices and logits
    // Returns the number of valid tokens, and truncates sequence if mismatch is found
    size_t validate_tree_candidates(Sequence::Ptr& running_sequence,
                                    const ov::Tensor& sequence_group_logits,
                                    LogitProcessor& logit_processor,
                                    size_t num_tokens_to_validate);

    SequenceGroupSamplingInfo sample_from_sequence_group(SequenceGroup::Ptr sequence_group, ov::Tensor sequence_group_logits,
                                                        RequestSamplerContext& context,
                                                        bool is_validation_mode_enabled);

    // request ID => beam search tracking information (kept separate — has its own mutex)
    std::map<uint64_t, GroupBeamSearcher> m_beam_search_info;
    std::mutex m_beam_search_info_mutex;

    std::map<uint64_t, RequestSamplerContext> m_request_contexts;
    size_t m_default_seed = std::mt19937::default_seed;  // kept for set_seed/get_seed API compat
    // request ID => tree search tracking information
    std::map<uint64_t, TreeSearcher> m_tree_search_info;
    std::mutex m_tree_search_info_mutex;

    Tokenizer m_tokenizer;

    ThreadPool m_thread_pool;
    std::shared_ptr<ov::op::v0::Constant> m_d2t_mapping;  // vocab index offset from draft to target token space (EAGLE)
public:
    Sampler(const Sampler& rhs) = delete;
    Sampler(Sampler&& rhs) = delete;
    Sampler(size_t num_threads = 1): m_thread_pool(num_threads) {};
    explicit Sampler(const Tokenizer & tokenizer, size_t num_threads = 1) : m_tokenizer(tokenizer), m_thread_pool(num_threads) {};

    SamplerOutput sample(const std::vector<SequenceGroup::Ptr> & sequence_groups, ov::Tensor logits, bool is_validation_mode_enabled = false);

    // Non-CB pipelines required API for seed. The CB path uses per-request engines from m_request_contexts.
    void set_seed(size_t new_seed) { m_default_seed = new_seed; }
    size_t get_seed() const { return m_default_seed; }

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

    void set_d2t_for_decoding(const std::shared_ptr<ov::op::v0::Constant>& d2t) {
        m_d2t_mapping = d2t;
    };
};

class Sampler::Searcher {
public:
    struct Beam {
        Sequence::Ptr m_sequence;
        size_t m_global_beam_idx = 0;

        // beam is made on top of sequence
        float m_log_prob = 0.0f;
        int64_t m_token_id = -1;
        int m_tree_layer = 0;
        int64_t m_node_id = 0;
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

    SequenceGroup::Ptr m_sequence_group;
    ov::genai::GenerationConfig m_parameters;
    Tokenizer m_tokenizer;

    explicit Searcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer)
        : m_sequence_group(std::move(sequence_group)),
          m_parameters{m_sequence_group->get_sampling_parameters()},
          m_tokenizer(std::move(tokenizer)) {}

protected:
    explicit Searcher(SequenceGroup::Ptr sequence_group)
        : m_sequence_group(std::move(sequence_group)),
          m_parameters{m_sequence_group->get_sampling_parameters()} {}
};

class Sampler::TreeSearcher : public Sampler::Searcher {
    // A branch in the current draft frontier: tracks which sequence is executing which graph node.
    struct DraftBeam {
        uint64_t node_id;
        Sequence::Ptr m_sequence;
        float score = 0.0f;  // cumulative log-probability
    };

    // A newly sampled child candidate, carrying all information needed for forking.
    struct CandidateBeam {
        uint64_t node_id;
        Sequence::Ptr parent_sequence;
        int64_t token_id;
        float log_prob;
        float score;
    };

    enum class Phase { IDLE, DRAFTING };

    Phase m_phase = Phase::IDLE;
    std::optional<EagleCandidateGraph> m_candidate_graph;
    std::vector<DraftBeam> m_frontier;
    size_t m_current_draft_layer = 0;
    size_t m_pre_draft_generated_len = 0;
    uint64_t m_original_grouped_id = 0;
    ov::Tensor m_d2t_tensor;  // keeps draft-to-target vocab offset tensor alive

    void tree_reset();
    auto build_top_k_frontier(const ov::Tensor& logits) -> std::vector<CandidateBeam>;
    void advance_draft_layer(const std::vector<CandidateBeam>& candidates, SamplerOutput& sampler_output);
    void finalize_tree(SamplerOutput& sampler_output, LogitProcessor& logit_processor);

public:
    explicit TreeSearcher(SequenceGroup::Ptr sequence_group, ov::Tensor d2t);

    void advance_draft_step(const ov::Tensor& logits, SamplerOutput& sampler_output, LogitProcessor& logit_processor);
};

class Sampler::GroupBeamSearcher : public Sampler::Searcher {
    using Sampler::Searcher::Beam;
    struct Group {
        std::vector<Beam> ongoing;  // Best beams in front
        std::vector<Beam> min_heap;  // The worst of the best completed beams is the first
        bool done = false;

        int64_t finish(Beam beam, const ov::genai::GenerationConfig& sampling_params);
        void is_done();
    };

    std::vector<Group> m_groups;

public:
    explicit GroupBeamSearcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer);

    void select_next_tokens(const ov::Tensor& logits, SamplerOutput& sampler_output, const std::pair<size_t, std::set<std::string>>& stop_strings);
    void finalize(SamplerOutput& sampler_output);
    std::map<size_t, int32_t> get_beam_idxs();
};
}
