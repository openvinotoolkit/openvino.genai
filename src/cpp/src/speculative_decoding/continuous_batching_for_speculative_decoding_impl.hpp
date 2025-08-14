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

    RawPerfMetrics raw_perf_metrics;

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
    EagleGeneratedRequests get_generated_requests();
    UpdateRequestResult update_main_request(uint64_t request_id, const EagleGeneratedSequences& candidates);
    UpdateRequestResult update_draft_request(uint64_t request_id, const EagleGeneratedSequences& candidates);
    void clear_sampler_top_k_selector(uint64_t request_id) {
        if (m_sampler) {
            m_sampler->clear_top_k_selector(request_id);
        }
    }
    bool is_requests_empty();

    void set_d2t_for_draft_decoding(std::shared_ptr<ov::op::v0::Constant>& d2t) {
        if (m_sampler) {
            m_sampler->set_d2t_for_decoding(d2t);
        }
    }
    void set_hidden_state_export_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->set_hidden_state_export_needed(is_needed);
        }
    }

    void set_hidden_state_import_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->set_hidden_state_import_needed(is_needed);
        }
    }

    void set_hidden_state_internal_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->set_hidden_state_internal_needed(is_needed);
        }
    }
    size_t get_processed_tokens_per_iteration();

    UpdateRequestResult init_request_by_candidate(uint64_t request_id, const GeneratedSequences& candidates);
    RawPerfMetrics raw_perf_metrics;
protected:
    void finish_request(SequenceGroup::Ptr request);
    void _pull_awaiting_requests() override {};
    ov::Tensor truncate_hidden_state_from_end(const ov::Tensor& hidden_state, size_t tokens_to_remove) {
        if (hidden_state.get_size() == 0 || tokens_to_remove == 0) {
            return hidden_state;
        }

        auto shape = hidden_state.get_shape();
        if (shape.size() < 2) {
            return hidden_state;
        }

        size_t seq_len_dim = 0;
        size_t current_seq_len = shape[seq_len_dim];

        if (tokens_to_remove >= current_seq_len) {
            ov::Shape new_shape = shape;
            new_shape[seq_len_dim] = 0;
            return ov::Tensor(hidden_state.get_element_type(), new_shape);
        }

        size_t new_seq_len = current_seq_len - tokens_to_remove;

        ov::Coordinate start_coord(shape.size(), 0);
        ov::Coordinate end_coord(shape.size(), 0);

        for (size_t i = 0; i < shape.size(); ++i) {
            start_coord[i] = 0;
            if (i == seq_len_dim) {
                end_coord[i] = new_seq_len;
            } else {
                end_coord[i] = shape[i];
            }
        }

        return ov::Tensor(hidden_state, start_coord, end_coord);
    }
};
}  // namespace ov::genai
