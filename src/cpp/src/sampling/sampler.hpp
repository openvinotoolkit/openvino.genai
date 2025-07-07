
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
        float m_log_prob = 0.0f;
        int64_t m_token_id = -1;
        uint64_t m_beam_id = 0;
        // cumulative log probabilities
        float m_score = -std::numeric_limits<float>::infinity();
        size_t m_tree_layer = 0;
        Beam(Sequence::Ptr sequence)
            : m_sequence(std::move(sequence)) { }

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
        
        uint64_t node_id;
        CandidateNode(Beam beam, uint64_t id) : candidate_beam(beam), node_id(id) {}
    };
    class Eagle2CandidateGraph {
    public:
        Eagle2CandidateGraph(Beam beam, int k = 0, int max_depth = 0) : total_tokens(k), max_depth(max_depth), current_depth(0), next_node_id(1) {
            root = std::make_shared<CandidateNode>(beam, 0);
            node_map[0] = root;

            layer_to_nodes[0].push_back(root); // for access to nodes on same layer
        }

        void add_candidate(Beam& new_beam, const Beam& parent_beam) {
            if (current_depth > max_depth && new_beam.m_tree_layer > max_depth) {
                return;
            }
            std::shared_ptr<CandidateNode> parent_node = nullptr;
            if (beam_to_node_id.find(get_beam_key(parent_beam)) != beam_to_node_id.end()) {
                uint64_t parent_id = beam_to_node_id[get_beam_key(parent_beam)];
                parent_node = node_map[parent_id];
            } else {
                size_t parent_layer = parent_beam.m_tree_layer;
                if (layer_to_nodes.find(parent_layer) != layer_to_nodes.end()) {
                    for (auto& node : layer_to_nodes[parent_layer]) {
                        if (is_same_beam(node->candidate_beam, parent_beam)) {
                            parent_node = node;
                            break;
                        }
                    }
                }
            }
            if (!parent_node) {
                OPENVINO_THROW("cannot locate parent beam in the candidate graph");
            }
            uint64_t new_id = next_node_id++;
            auto new_node = std::make_shared<CandidateNode>(new_beam, new_id);
            new_node->candidate_beam.m_beam_id = new_id; // update beam ID to match node ID
            new_node->parent = parent_node;
            
            parent_node->children.push_back(new_node);
            node_map[new_id] = new_node;
            beam_to_node_id[get_beam_key(new_beam)] = new_id;
            layer_to_nodes[new_beam.m_tree_layer].push_back(new_node);
            
            if (new_beam.m_tree_layer > current_depth) {
                current_depth = new_beam.m_tree_layer;
            }
        }

        std::vector<Beam> get_top_k_candidates() {
            auto cmp = [](const std::shared_ptr<CandidateNode>& a, const std::shared_ptr<CandidateNode>& b) {
                if (a->candidate_beam.m_score == b->candidate_beam.m_score) {
                    // TBD: tie-breaker if needed
                }
                return a->candidate_beam.m_score > b->candidate_beam.m_score;
            };
            std::priority_queue<std::shared_ptr<CandidateNode>,
                                std::vector<std::shared_ptr<CandidateNode>>,
                                decltype(cmp)> min_heap(cmp);
            std::queue<std::shared_ptr<CandidateNode>> bfs_queue;
            bfs_queue.push(root);
            while (!bfs_queue.empty()) {
                auto node = bfs_queue.front();
                bfs_queue.pop();
                if (node != root) {
                    if (min_heap.size() < static_cast<size_t>(total_tokens)) {
                        min_heap.push(node);
                    } else if (node->candidate_beam.m_score > min_heap.top()->candidate_beam.m_score) {
                        min_heap.pop();
                        min_heap.push(node);
                    }
                }

                // Add all child nodes to BFS queue
                for (auto& child : node->children) {
                    bfs_queue.push(child);
                }
            }
            std::vector<Beam> result;
            result.reserve(min_heap.size());

            while (!min_heap.empty()) {
                result.push_back(min_heap.top()->candidate_beam);
                min_heap.pop();
            }

            std::reverse(result.begin(), result.end());
            return result;
        }

        std::vector<Beam> get_leaf_nodes_from_candidates(const std::vector<Beam>& candidates) {
            std::vector<Beam> leaf_nodes;
            std::unordered_set<uint64_t> candidate_ids;
            for (const auto& beam : candidates) {
                candidate_ids.insert(beam.m_beam_id);
            }

            for (const auto& candidate_beam : candidates) {
                uint64_t node_id = candidate_beam.m_beam_id;
                
                if (node_map.find(node_id) != node_map.end()) {
                    auto node = node_map[node_id];
                    if (node->children.empty()) {
                        leaf_nodes.push_back(candidate_beam);
                        continue;
                    }
                }

                bool is_leaf = true;
                auto node_ptr = node_map[node_id];
                for (const auto& child : node_ptr->children) {
                    if (candidate_ids.count(child->candidate_beam.m_beam_id) > 0) {
                        is_leaf = false;
                        break;
                    }
                }

                if (is_leaf) {
                    leaf_nodes.push_back(candidate_beam);
                }
            }   
            return leaf_nodes;
        }
        std::vector<int64_t> get_path_to_node(uint64_t node_id) {
            if (node_map.find(node_id) == node_map.end()) {
                return {};
            }
            
            std::vector<int64_t> path;
            auto node = node_map[node_id];
            
            while (node && node != root) {
                path.push_back(node->candidate_beam.m_token_id);
                auto parent = node->parent.lock();
                node = parent;
            }
            
            std::reverse(path.begin(), path.end());
            return path;
        }

        void print_tree() {  // for debugging purposes
            std::cout << "Eagle2 Candidate Tree (Depth: " << current_depth << ")\n";
            print_node(root, 0);
        }

    private:
        std::shared_ptr<CandidateNode> root;
        uint64_t next_node_id; // for new node
        
        std::unordered_map<uint64_t, std::shared_ptr<CandidateNode>> node_map;
        std::unordered_map<std::string, uint64_t> beam_to_node_id;
        std::unordered_map<size_t, std::vector<std::shared_ptr<CandidateNode>>> layer_to_nodes;

        int total_tokens;
        int max_depth;
        int current_depth;

        std::string get_beam_key(const Beam& beam) {
            std::stringstream ss;
            ss << beam.m_beam_id << "_" 
            << beam.m_token_id << "_" 
            << beam.m_tree_layer;
            return ss.str();
        }
        bool is_same_beam(const Beam& a, const Beam& b) {
            return a.m_beam_id == b.m_beam_id &&
               a.m_token_id == b.m_token_id &&
               a.m_tree_layer == b.m_tree_layer;
        }

        void update_tree_depth() {
            int max_depth_found = 0;
            std::function<void(std::shared_ptr<CandidateNode>, int)> dfs = [&](std::shared_ptr<CandidateNode> node,
                                                                                int depth) {
                max_depth_found = std::max(max_depth_found, depth);
                for (auto& child : node->children) {
                    dfs(child, depth + 1);
                }
            };

            dfs(root, 0);
            current_depth = max_depth_found;
        }

        void print_node(const std::shared_ptr<CandidateNode>& node, int depth) {
            for (int i = 0; i < depth; i++)
                std::cout << "  ";

            if (node == root) {
                std::cout << "[ROOT]\n";
            } else {
                std::cout << "Token: " << node->candidate_beam.m_token_id << " | Score: " << node->candidate_beam.m_score << "\n";
            }

            for (auto& child : node->children) {
                print_node(child, depth + 1);
            }
        }
    };
    static size_t m_tree_layer_counter;
    SequenceGroup::Ptr m_sequence_group;
    std::shared_ptr<Eagle2CandidateGraph> m_eagle2_candidate_graph;
    std::vector<Beam> m_beams;
    ov::genai::GenerationConfig m_parameters;
public:
    explicit TopKSelector(SequenceGroup::Ptr sequence_group);

    void select_top_k(const ov::Tensor& logits, SamplerOutput& sampler_output);
    void finalize_eagle2_candidates(SamplerOutput& sampler_output);
    //float get_eagle2_layer_weight(size_t layer) {
        //return std::ext(-m_parameter.eagle_layer_decay * (layer - 1));
    //}
    void apply_eagle2_scoring() { } // to be implemented
};
}
