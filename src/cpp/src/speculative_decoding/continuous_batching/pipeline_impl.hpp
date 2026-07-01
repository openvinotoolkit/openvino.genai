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

    /**
     * @brief Enables/disables export of the model's `last_hidden_state`, delegating to the model runner.
     *        Shared by EAGLE3 and MTP hidden-state pairing.
     */
    void set_hidden_state_export_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_hidden_state_export(is_needed);
        }
    }

    /**
     * @brief Enables/disables import of the main model's hidden state into this (draft) model runner,
     *        used for the first draft forward in each speculative-decode step.
     */
    void set_hidden_state_import_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_hidden_state_import(is_needed);
        }
    }

    /**
     * @brief Enables/disables use of the internally stored hidden state for the draft model runner,
     *        used for the 2nd..num_assistant draft forwards in each speculative-decode step.
     */
    void set_hidden_state_internal_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_hidden_state_internal(is_needed);
        }
    }

protected:
    void finish_request(SequenceGroup::Ptr request);
    void _pull_awaiting_requests() override {};
    bool eagle_mode_enabled = false;
    bool mtp_mode_enabled = false;
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
};

/**
 * @brief Continuous-batching pipeline for a Multi-Token Prediction (MTP) draft or main model.
 *
 * MTP reuses the EAGLE3 hidden-state pairing machinery (import/internal/export gated on the same
 * `mtp_mode_enabled` path as `eagle_mode_enabled`). Both the MTP main and draft models consume
 * `inputs_embeds` produced by a shared `EmbeddingsModel`, so this impl takes an `InputsEmbedder`
 * and runs on the EMBEDDINGS input path. The draft additionally imports the main model's
 * `last_hidden_state` and pairs it with the embedding of the main-predicted token.
 */
class ContinuousBatchingPipeline::ContinuousBatchingForMtpDecodingImpl
    : public ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl {
public:
    ContinuousBatchingForMtpDecodingImpl() = default;

    ContinuousBatchingForMtpDecodingImpl(const std::shared_ptr<ov::Model>& model,
                                         const std::shared_ptr<InputsEmbedder>& inputs_embedder,
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
        mtp_mode_enabled = true;
        // Both MTP main and draft consume inputs_embeds from the shared embeddings model.
        m_inputs_embedder = inputs_embedder;
        m_model_runner->set_inputs_embedder(inputs_embedder);
        m_model_input_type = ModelInputType::EMBEDDINGS;
    }

    /**
     * @brief Switches the draft model runner to plain sequential position_ids.
     *
     * The MTP draft consumes rank-1 sequential positions, unlike the main VLM language model whose
     * shared InputsEmbedder produces rank-3 M-RoPE positions.
     */
    void set_mtp_draft_positions_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_mtp_draft_positions(is_needed);
        }
    }
};
}
