// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "continuous_batching/pipeline_impl.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "speculative_decoding/update_request_structs.hpp"

namespace ov::genai {
struct CandidateNode {
    int64_t token_id;
    float score;
    std::vector<std::shared_ptr<CandidateNode>> children;
    std::weak_ptr<CandidateNode> parent;

    CandidateNode(int64_t token, float sc) : token_id(token), score(sc) {}
};

// eagle 2 tree structure
class Eagle2CandidateGraph {
public:
    Eagle2CandidateGraph(int k = 0, int max_depth = 0) : beam_size(k), max_depth(max_depth), current_depth(0) {
        root = std::make_shared<CandidateNode>(-1, 1.0f);
    }

    void add_candidate(const std::vector<int>& token_seq, float sequence_score) {
        auto current = root;
        float cumulative_score = 0.0f;

        for (int i = 0; i < token_seq.size(); i++) {
            int token = token_seq[i];
            cumulative_score += sequence_score;

            auto it = std::find_if(current->children.begin(), current->children.end(), [token](const auto& child) {
                return child->token_id == token;
            });

            if (it == current->children.end()) {
                auto newNode = std::make_shared<CandidateNode>(token, cumulative_score);
                newNode->parent = current;
                current->children.push_back(newNode);
                current = newNode;
            } else {
                if (cumulative_score > (*it)->score) {
                    (*it)->score = cumulative_score;
                }
                current = *it;
            }
        }

        current_depth = std::max(current_depth, static_cast<int>(token_seq.size()));
    }

    std::vector<std::pair<std::vector<int>, float>> get_top_k_candidates() {
        auto cmp = [](const auto& a, const auto& b) {
            return a.second < b.second;
        };
        std::priority_queue<std::pair<std::shared_ptr<CandidateNode>, float>,
                            std::vector<std::pair<std::shared_ptr<CandidateNode>, float>>,
                            decltype(cmp)>
            pq(cmp);

        for (auto& child : root->children) {
            pq.push({child, child->score});
        }

        std::vector<std::pair<std::vector<int>, float>> top_candidates;
        while (!pq.empty() && top_candidates.size() < beam_size) {
            auto [node, score] = pq.top();
            pq.pop();

            auto sequence = backtrack_sequence(node);

            if (sequence.size() == current_depth) {
                top_candidates.push_back({sequence, score});
            }

            for (auto& child : node->children) {
                pq.push({child, child->score});
            }
        }

        return top_candidates;
    }

    void print_tree() {  // for debugging purposes
        std::cout << "Eagle2 Candidate Tree (Depth: " << current_depth << ")\n";
        print_node(root, 0);
    }

private:
    std::shared_ptr<CandidateNode> root;
    int beam_size;
    int max_depth;
    int current_depth;

    std::vector<int> backtrack_sequence(std::shared_ptr<CandidateNode> node) {
        std::vector<int> sequence;
        while (node != root) {
            sequence.push_back(node->token_id);
            node = node->parent.lock();
        }
        std::reverse(sequence.begin(), sequence.end());
        return sequence;
    }

    void print_node(const std::shared_ptr<CandidateNode>& node, int depth) {
        for (int i = 0; i < depth; i++)
            std::cout << "  ";

        if (node == root) {
            std::cout << "[ROOT]\n";
        } else {
            std::cout << "Token: " << node->token_id << " | Score: " << node->score << "\n";
        }

        for (auto& child : node->children) {
            print_node(child, depth + 1);
        }
    }
};
class ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl
    : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
public:
    ContinuousBatchingForSpeculativeDecodingImpl() = default;

    ContinuousBatchingForSpeculativeDecodingImpl(const std::shared_ptr<ov::Model>& model,
                                                 const Tokenizer& tokenizer,
                                                 const GenerationConfig& generation_config,
                                                 const SchedulerConfig& scheduler_config,
                                                 const std::string& device,
                                                 const ov::AnyMap& plugin_config,
                                                 bool is_validation_mode_enabled);

    void multistep();

    void finish_request(int64_t request_id = -1);
    void pull_awaiting_requests(bool is_pause_request = false);
    GeneratedRequests get_generated_requests();
    UpdateRequestResult update_request(uint64_t request_id, const GeneratedSequences& candidates, bool is_update_logit_processor);
    bool is_requests_empty();

    size_t get_processed_tokens_per_iteration();

    UpdateRequestResult init_request_by_candidate(uint64_t request_id, const GeneratedSequences& candidates);

protected:
    void finish_request(SequenceGroup::Ptr request);
    void _pull_awaiting_requests() override {};
};

class ContinuousBatchingPipeline::ContinuousBatchingForEagleDecodingImpl
    : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
public:
    ContinuousBatchingForEagleDecodingImpl() = default;

    ContinuousBatchingForEagleDecodingImpl(const std::shared_ptr<ov::Model>& model,
                                                 const Tokenizer& tokenizer,
                                                 const GenerationConfig& generation_config,
                                                 const SchedulerConfig& scheduler_config,
                                                 const std::string& device,
                                                 const ov::AnyMap& plugin_config,
                                                 bool is_validation_mode_enabled);

    void multistep();

    void finish_request(int64_t request_id = -1);
    void pull_awaiting_requests(bool is_pause_request = false);
    GeneratedRequests get_generated_requests();
    UpdateRequestResult update_request(uint64_t request_id, const GeneratedSequences& candidates, bool is_update_logit_processor);
    bool is_requests_empty();

    size_t get_processed_tokens_per_iteration();

    UpdateRequestResult init_request_by_candidate(uint64_t request_id, const GeneratedSequences& candidates);

protected:
    void finish_request(SequenceGroup::Ptr request);
    void _pull_awaiting_requests() override {};
};
}  // namespace ov::genai
