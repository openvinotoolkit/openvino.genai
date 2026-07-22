// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>

#include "continuous_batching/pipeline_impl.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "update_request_structs.hpp"

namespace ov::genai {

namespace detail {

struct MtpDraftUpdatePlan {
    size_t hidden_state_start = 0;
    size_t hidden_state_count = 0;
    size_t processed_tokens_to_rewind = 0;
    size_t num_tokens_to_validate = 0;
};

// Plans the first draft forward after target validation. The target hidden-state window contains
// one base token followed by all draft candidates; removed_draft_tokens is the rejected suffix.
MtpDraftUpdatePlan make_mtp_draft_update_plan(size_t main_hidden_state_len, size_t removed_draft_tokens);

}  // namespace detail

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

    ContinuousBatchingForSpeculativeDecodingImpl(const std::shared_ptr<ov::Model>& model,
                                                 std::shared_ptr<InputsEmbedder> inputs_embedder,
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
    void sync_generated_embeddings();
    bool is_requests_empty();

    size_t get_processed_tokens_per_iteration();

    // Rewinds an awaiting request to an earlier processed-prefix position and synchronizes
    // physical block tables with the updated logical context.
    bool rewind_awaiting_request_prefix(uint64_t request_id, size_t processed_tokens);

    UpdateRequestResult init_request_by_candidate(uint64_t request_id, const GeneratedSequences& candidates);

    RawPerfMetrics raw_perf_metrics;

    ov::Any get_model_property(const std::string& name) {
        OPENVINO_ASSERT(m_model_runner, "get_model_property('", name, "') called before model runner is initialized");
        auto compiled_model = m_model_runner->get_infer_request().get_compiled_model();
        const auto supported = compiled_model.get_property(ov::supported_properties);
        OPENVINO_ASSERT(std::find(supported.begin(), supported.end(), name) != supported.end(),
                        "Compiled model does not support property '",
                        name,
                        "'");
        return compiled_model.get_property(name);
    }

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

    ContinuousBatchingForEagle3DecodingImpl(const std::shared_ptr<ov::Model>& model,
                                            std::shared_ptr<InputsEmbedder> inputs_embedder,
                                            const Tokenizer& tokenizer,
                                            const GenerationConfig& generation_config,
                                            const SchedulerConfig& scheduler_config,
                                            const std::string& device,
                                            const ov::AnyMap& plugin_config,
                                            bool is_validation_mode_enabled)
        : ContinuousBatchingForSpeculativeDecodingImpl(model,
                                                       inputs_embedder,
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

    void collect_block_update_info(const GeneratedRequests& main_generated_requests,
                                   std::vector<int32_t>& block_update_indices,
                                   std::vector<int32_t>& block_update_begins) const;

    ov::Tensor get_tensor_by_name(const std::string& name) {
        if (m_model_runner) {
            return m_model_runner->get_infer_request().get_tensor(name);
        }
        return {};
    }
};

// EMBEDDINGS pipeline used by MTP main and draft models.
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
        m_inputs_embedder = inputs_embedder;
        m_model_runner->set_inputs_embedder(inputs_embedder);
        m_model_input_type = ModelInputType::EMBEDDINGS;
    }

    void set_mtp_draft_positions_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_mtp_draft_positions(is_needed);
        }
    }
};
}
