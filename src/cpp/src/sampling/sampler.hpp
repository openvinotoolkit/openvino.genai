
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

#include "sampling/logit_processor.hpp"
#include "continuous_batching/scheduler.hpp"
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
struct Eagle2ValidationResult {
    size_t accepted_path_id = -1;
    size_t accepted_path_length = 0;
    std::vector<int64_t> accepted_tokens;
    std::vector<float> updated_log_probs;
    bool is_path_accepted = false;
    size_t rejected_at_position = 0;
    Token extra_sampled_token; // sampled in main model from the rejection pos
};

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
    class TopKSelector;

    Logits _get_logit_vector(ov::Tensor logits, size_t batch_idx, size_t token_idx);
    Token _greedy_sample(const Logits& logits, size_t top_logprobs) const;
    std::vector<Token> _multinomial_sample(const Logits& logits, size_t num_tokens_per_sequence);
    std::vector<int64_t> _try_finish_generation(SequenceGroup::Ptr & sequence_group);

    bool validate_candidate(Sequence::Ptr running_sequence, size_t& token_idx, Token& sampled_token,
                            bool& is_extend_sequence, size_t& max_removed_tokens, bool do_sample);
    // EAGLE2 tree validation functions
    Eagle2ValidationResult validate_eagle2_tree(
        const std::vector<std::vector<int64_t>>& candidate_paths,
        const std::vector<std::vector<float>>& candidate_log_probs,
        const std::vector<int> beam_id,
        const ov::Tensor& main_model_logits,
        LogitProcessor& logit_processor,
        bool do_sample = false);

    // Helper functions for EAGLE2 validation
    size_t find_common_prefix_length(const std::vector<std::vector<int64_t>>& paths);

    std::pair<bool, float> find_token_probability(const Logits& logits, int64_t token_id);

    Token get_highest_probability_token(const Logits& logits);

    size_t select_best_continuation_path(
        const std::vector<std::vector<int64_t>>& candidate_paths,
        const std::vector<std::vector<float>>& candidate_log_probs,
        const std::vector<int>& beam_id,
        const size_t& common_prefix_length,
        Logits& logit,
        const ov::Tensor& main_model_logits,
        LogitProcessor& logit_processor);

    SequenceGroupSamplingInfo sample_from_sequence_group(SequenceGroup::Ptr sequence_group, ov::Tensor sequence_group_logits,
                                                        LogitProcessor& logit_processor, const std::pair<size_t, std::set<std::string>>& stop_strings,
                                                        bool is_validation_mode_enabled);

    // request ID => beam search tracking information
    std::map<uint64_t, GroupBeamSearcher> m_beam_search_info;

    std::map<uint64_t, TopKSelector> m_top_k_selector_info;
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

    int validate_eagle2_candidates(
        SequenceGroup::Ptr seq_group,
        const ov::Tensor& main_model_logits,
        LogitProcessor& logit_processor,
        size_t& accepted_tokens_count,
        size_t& max_removed_tokens,
        size_t& num_tokens_to_process,
        bool do_sample = false);

    std::map<size_t, int32_t> get_beam_idxs(SequenceGroup::CPtr sequence_group);
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

class Sampler::TopKSelector {
    struct Beam {
        Sequence::Ptr m_sequence;
        size_t m_global_beam_idx = 0;

        // beam is made on top of sequence
        float m_log_prob = 0.0f;
        int64_t m_token_id = -1;
        int m_tree_layer = 0;   // layer in the tree structure
        int64_t m_node_id = 0;  // unique ID for the beam in the tree structure
        // cumulative log probabilities
        float m_score = -std::numeric_limits<float>::infinity();

        Beam(Sequence::Ptr sequence) : m_sequence(std::move(sequence)) {}

        size_t get_generated_len() const {
            return m_sequence->get_generated_len();
        }
    };

    static bool greater(const Beam& left, const Beam& right) {
        return left.m_score > right.m_score;
    }
    struct CandidateNode {
        Beam candidate_beam;
        std::vector<std::shared_ptr<CandidateNode>> children;
        std::weak_ptr<CandidateNode> parent;

        uint64_t get_id() const {
            return candidate_beam.m_node_id;
        }
        CandidateNode(Beam beam) : candidate_beam(std::move(beam)) {}
    };
    class Eagle2CandidateGraph {
    public:
        Eagle2CandidateGraph(Beam root_beam, int k = 0, int max_depth = 0)
            : total_tokens(k),
              max_depth(max_depth),
              current_depth(0),
              next_node_id(1) {
            root_beam.m_node_id = 0;  // Root always has ID 0
            root_beam.m_tree_layer = 0;
            root = std::make_shared<CandidateNode>(std::move(root_beam));

            node_map[0] = root;
            layer_to_nodes[0].push_back(root);  // for access to nodes on same layer
        }

        void add_candidate(Beam& new_beam, uint64_t parent_node_id) {
            if (new_beam.m_tree_layer > max_depth) {
                return;
            }

            auto parent_it = node_map.find(parent_node_id);
            if (parent_it == node_map.end()) {
                OPENVINO_THROW("Parent node not found in candidate graph");
            }

            auto parent_node = parent_it->second;

            // Assign new node ID and update beam
            new_beam.m_node_id = next_node_id++;
            new_beam.m_tree_layer = parent_node->candidate_beam.m_tree_layer + 1;

            auto new_node = std::make_shared<CandidateNode>(new_beam);
            new_node->parent = parent_node;

            // Add to parent's children
            parent_node->children.push_back(new_node);

            // Update mappings
            node_map[new_node->get_id()] = new_node;
            layer_to_nodes[new_node->candidate_beam.m_tree_layer].push_back(new_node);

            // Update current depth
            current_depth = std::max(current_depth, new_node->candidate_beam.m_tree_layer);
        }

        std::vector<Beam> get_top_k_candidates() {
            if (total_tokens <= 0)
                return {};

            // Use min-heap to efficiently get top-k candidates
            auto cmp = [](const std::shared_ptr<CandidateNode>& a, const std::shared_ptr<CandidateNode>& b) {
                return a->candidate_beam.m_score > b->candidate_beam.m_score;  // min-heap
            };

            std::priority_queue<std::shared_ptr<CandidateNode>,
                                std::vector<std::shared_ptr<CandidateNode>>,
                                decltype(cmp)>
                min_heap(cmp);

            // BFS traversal to find all candidates (excluding root)
            std::queue<std::shared_ptr<CandidateNode>> bfs_queue;
            bfs_queue.push(root);

            while (!bfs_queue.empty()) {
                auto node = bfs_queue.front();
                bfs_queue.pop();

                // Add non-root nodes to heap
                if (node != root) {
                    if (min_heap.size() < static_cast<size_t>(total_tokens)) {
                        min_heap.push(node);
                    } else if (node->candidate_beam.m_score > min_heap.top()->candidate_beam.m_score) {
                        min_heap.pop();
                        min_heap.push(node);
                    }
                }

                // Add children to BFS queue
                for (const auto& child : node->children) {
                    bfs_queue.push(child);
                }
            }

            // Extract results and sort by score (descending)
            std::vector<Beam> result;
            result.reserve(min_heap.size());

            while (!min_heap.empty()) {
                result.push_back(min_heap.top()->candidate_beam);
                min_heap.pop();
            }

            std::sort(result.begin(), result.end(), [](const Beam& a, const Beam& b) {
                return a.m_score > b.m_score;
            });

            return result;
        }

        std::vector<Beam> get_leaf_nodes_from_candidates(const std::vector<Beam>& candidates) {
            std::vector<Beam> leaf_nodes;
            std::unordered_set<uint64_t> candidate_ids;

            // Build set of candidate node IDs
            for (const auto& beam : candidates) {
                candidate_ids.insert(beam.m_node_id);
            }

            // Check each candidate to see if it's a leaf in the selected set
            for (const auto& candidate_beam : candidates) {
                auto node_it = node_map.find(candidate_beam.m_node_id);
                if (node_it == node_map.end())
                    continue;

                auto node = node_it->second;

                // Check if this node has any children in the candidate set
                bool has_candidate_child = false;
                for (const auto& child : node->children) {
                    if (candidate_ids.count(child->get_id()) > 0) {
                        has_candidate_child = true;
                        break;
                    }
                }

                if (!has_candidate_child) {
                    leaf_nodes.push_back(candidate_beam);
                }
            }

            return leaf_nodes;
        }

        std::vector<int64_t> get_path_to_node(uint64_t node_id) {
            auto node_it = node_map.find(node_id);
            if (node_it == node_map.end()) {
                return {};
            }

            std::vector<int64_t> path;
            auto current_node = node_it->second;

            // Traverse from node to root, collecting token IDs
            while (current_node && current_node != root) {
                path.push_back(current_node->candidate_beam.m_token_id);
                current_node = current_node->parent.lock();
            }
            // path.push_back(root->candidate_beam.m_token_id); // Add root
            // Reverse to get path from root to node
            std::reverse(path.begin(), path.end());
            return path;
        }

        void print_tree() {  // for debugging purposes
            std::cout << "Eagle2 Candidate Tree (Depth: " << current_depth << ")\n";
            print_node(root, 0);
        }

    private:
        std::shared_ptr<CandidateNode> root;
        uint64_t next_node_id;  // for new node

        std::unordered_map<uint64_t, std::shared_ptr<CandidateNode>> node_map;
        std::unordered_map<size_t, std::vector<std::shared_ptr<CandidateNode>>> layer_to_nodes;

        int total_tokens;
        int max_depth;
        int current_depth;

        void print_node(const std::shared_ptr<CandidateNode>& node, int depth) {
            std::string indent(depth * 2, ' ');

            if (node == root) {
                std::cout << indent << "[ROOT] ID: " << node->get_id() << "\n";
            } else {
                std::cout << indent << "ID: " << node->get_id() << " Token: " << node->candidate_beam.m_token_id
                          << " Score: " << node->candidate_beam.m_score
                          << " Layer: " << node->candidate_beam.m_tree_layer << "\n";
            }

            for (const auto& child : node->children) {
                print_node(child, depth + 1);
            }
        }
    };
    size_t m_tree_layer_counter;
    SequenceGroup::Ptr m_sequence_group;
    std::shared_ptr<Eagle2CandidateGraph> m_eagle2_candidate_graph;
    std::vector<Beam> m_beams;
    ov::genai::GenerationConfig m_parameters;

public:
    explicit TopKSelector(SequenceGroup::Ptr sequence_group);

    void select_top_k(const ov::Tensor& logits, SamplerOutput& sampler_output);
    void finalize_eagle2_candidates(SamplerOutput& sampler_output);
    // float get_eagle2_layer_weight(size_t layer) {
    // return std::ext(-m_parameter.eagle_layer_decay * (layer - 1));
    //}
    void apply_eagle2_scoring() {}  // to be implemented
};
}  // namespace ov::genai
