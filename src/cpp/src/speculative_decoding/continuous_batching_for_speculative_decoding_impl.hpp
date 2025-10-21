// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"

#include "continuous_batching/pipeline_impl.hpp"
#include "speculative_decoding/update_request_structs.hpp"

namespace ov::genai {
class ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
public:
    const std::size_t default_num_assistant_tokens = 5;
 
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
    bool eagle_mode_enabled = false;
};

class ContinuousBatchingPipeline::ContinuousBatchingForEagle3DecodingImpl
    : public ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl {
public:
    ContinuousBatchingForEagle3DecodingImpl() = default;

    ContinuousBatchingForEagle3DecodingImpl(const std::shared_ptr<ov::Model>& model,
                                                 const Tokenizer& tokenizer,
                                                 const GenerationConfig& generation_config,
                                                 const SchedulerConfig& scheduler_config,
                                                 const std::string& device,
                                                 const ov::AnyMap& plugin_config,
                                                 bool is_validation_mode_enabled) : ContinuousBatchingForSpeculativeDecodingImpl(
                                                     model, tokenizer, generation_config,
                                                     scheduler_config, device, plugin_config,
                                                     is_validation_mode_enabled) {
                                                        eagle_mode_enabled = true;
                                                     };

    bool is_requests_empty();

    void set_d2t_for_draft_decoding(std::shared_ptr<ov::op::v0::Constant>& d2t) {
        if (m_sampler) {
            m_sampler->set_d2t_for_decoding(d2t);
        }
    }
    void set_hidden_state_export_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_hidden_state_export(is_needed);
        }
    }

    void set_hidden_state_import_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_hidden_state_import(is_needed);
        }
    }

    void set_hidden_state_internal_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_hidden_state_internal(is_needed);
        }
    }

    void set_adjust_factor(size_t adjust_factor) {
        if (m_model_runner) {
            m_model_runner->set_adjust_factor(adjust_factor);
        }
    }
};
}
