// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "continuous_batching/pipeline_impl.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "speculative_decoding/update_request_structs.hpp"

namespace ov::genai {
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

class Eagle2CandidateManager {
public:
    Eagle2CandidateManager(const GenerationConfig& config) 
        : m_config(config),
          m_candidate_graph(config.eagle_tree_width, config.eagle_depth, config.eagle_final_candidates) {}
    
    void initialize_for_request(uint64_t request_id, const std::vector<int64_t>& prompt_ids) {
        m_request_graphs[request_id] = Eagle2CandidateGraph(
            m_config.eagle_tree_width, 
            m_config.eagle_depth, 
            m_config.eagle_final_candidates
        );
    }
    
    void add_tree_layer(uint64_t request_id, const std::vector<std::vector<Token>>& layer_tokens) {
        if (m_request_graphs.find(request_id) != m_request_graphs.end()) {
            m_request_graphs[request_id].add_candidate_layer(layer_tokens);
        }
    }
    
    std::vector<std::vector<int64_t>> finalize_candidates(uint64_t request_id) {
        if (m_request_graphs.find(request_id) != m_request_graphs.end()) {
            auto paths = m_request_graphs[request_id].get_top_k_paths();
            m_request_graphs[request_id].reset();
            return paths;
        }
        return {};
    }
    
    void cleanup_request(uint64_t request_id) {
        m_request_graphs.erase(request_id);
    }

private:
    GenerationConfig m_config;
    Eagle2CandidateGraph m_candidate_graph;
    std::unordered_map<uint64_t, Eagle2CandidateGraph> m_request_graphs;
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
