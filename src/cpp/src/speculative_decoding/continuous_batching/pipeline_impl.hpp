// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "continuous_batching/pipeline_impl.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "update_request_structs.hpp"

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
                                            bool is_validation_mode_enabled)
        : ContinuousBatchingForSpeculativeDecodingImpl(model,
                                                       tokenizer,
                                                       generation_config,
                                                       scheduler_config,
                                                       device,
                                                       plugin_config,
                                                       is_validation_mode_enabled) {
        eagle_mode_enabled = true;
    };

    bool is_requests_empty();

    /**
     * @brief Sets the draft-to-token (d2t) mapping for draft decoding.
     *
     * This function assigns the provided d2t constant to the sampler, if it exists,
     * by calling the sampler's set_d2t_for_decoding method. The d2t mapping is used
     * during the draft decoding process to tune the draft vocabs to match target vocabs.
     *
     * @param d2t A shared pointer to an ov::op::v0::Constant representing the draft-to-token mapping.
     */
    void set_d2t_for_draft_decoding(const std::shared_ptr<ov::op::v0::Constant>& d2t) {
        if (m_sampler) {
            m_sampler->set_d2t_for_decoding(d2t);
        }
    }

    /**
     * @brief Sets whether the export of hidden states is needed during model execution.
     *
     * This function enables or disables the export of hidden states by delegating
     * the request to the underlying model runner, if it exists.
     *
     * @param is_needed Boolean flag indicating whether hidden state export is required.
     */
    void set_hidden_state_export_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_hidden_state_export(is_needed);
        }
    }

    /**
     * @brief Sets whether the import of hidden state is needed for the model runner.
     *
     * This function enables or disables the import of hidden state in the underlying
     * model runner, which is for the first draft model inference in each speculative decode step.
     *
     * @param is_needed Boolean flag indicating whether hidden state import is required.
     */
    void set_hidden_state_import_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_hidden_state_import(is_needed);
        }
    }

    /**
     * @brief Sets whether the internal hidden state is required for the model runner.
     *
     * This function enables or disables the use of the internal hidden state in the model runner,
     * which is for the draft model 2...num_assistant forwards in each speculative decode step.
     *
     * @param is_needed Boolean flag indicating whether the internal hidden state should be enabled (true) or disabled (false).
     */
    void set_hidden_state_internal_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_hidden_state_internal(is_needed);
        }
    }
};
}
