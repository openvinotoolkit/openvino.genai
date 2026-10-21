// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include "continuous_batching/pipeline_impl.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "update_request_structs.hpp"
#include "utils.hpp"

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

// One DFlash draft window, [target hidden-state rows ; seed + mask block], planned by the draft update and read
// back by the draft multistep after the forward.
struct DFlashDraftWindow {
    size_t committed_rows = 0;  // draft KV rows already built; the window starts here
    size_t context_rows = 0;    // new target hidden-state rows to forward
    size_t block_rows = 0;      // seed + mask rows; 0 when there is no block
    size_t candidate_rows = 0;  // block rows sampled as candidates
};

// Plans the DFlash draft window of a request whose tokens match the target's. The draft request's processed
// length is its committed rows, and the target has hidden states up to target_processed_tokens.
DFlashDraftWindow make_dflash_draft_window(const SequenceGroup& request,
                                           size_t target_processed_tokens,
                                           size_t candidate_position_offset);

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

    void fail_pipeline(std::exception_ptr error) noexcept {
        _fail_pipeline(std::move(error));
    }

    void finish_request(int64_t request_id = -1);
    void pull_awaiting_requests(bool is_pause_request = false);
    GeneratedRequests get_generated_requests();
    UpdateRequestResult update_request(uint64_t request_id,
                                       const GeneratedSequences& candidates,
                                       bool is_update_logit_processor);
    void sync_generated_embeddings();
    bool is_requests_empty();

    size_t get_processed_tokens_per_iteration();

    // Rewinds an awaiting request to an earlier processed-prefix position and synchronizes
    // physical block tables with the updated logical context.
    bool rewind_awaiting_request_prefix(uint64_t request_id, size_t processed_tokens);

    bool is_prefix_caching_enabled() const {
        return m_scheduler->get_config().enable_prefix_caching;
    }

    // Returns the awaiting request with the given id, or nullptr if there is none.
    SequenceGroup::Ptr find_awaiting_request(uint64_t request_id) {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        const auto found = std::find_if(m_awaiting_requests.begin(), m_awaiting_requests.end(),
                                        [request_id](const SequenceGroup::Ptr& request) {
                                            return request->get_request_id() == request_id;
                                        });
        return found == m_awaiting_requests.end() ? nullptr : *found;
    }

    void discard_awaiting_request(uint64_t request_id) {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        for (auto request = m_awaiting_requests.begin(); request != m_awaiting_requests.end(); ++request) {
            if ((*request)->get_request_id() != request_id) {
                continue;
            }
            for (const auto& sequence : (*request)->get_sequences()) {
                m_scheduler->free_sequence(sequence->get_id());
            }
            m_awaiting_requests.erase(request);
            return;
        }
    }

    size_t restore_awaiting_prefix(uint64_t request_id, size_t ceiling) {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        for (const auto& group : m_awaiting_requests) {
            if (group->get_request_id() != request_id) {
                continue;
            }
            for (const auto& sequence : group->get_sequences()) {
                m_scheduler->free_sequence(sequence->get_id());
            }
            group->update_processed_tokens_num(0);
            if (m_scheduler->get_config().enable_prefix_caching) {
                m_scheduler->restore_cached_blocks(group, ceiling);
            }
            group->set_num_prefix_cache_hit_tokens(group->get_num_processed_tokens());
            return group->get_num_processed_tokens();
        }
        OPENVINO_THROW("Prefix restore requires an awaiting request: ", request_id);
    }

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


    // Returns the actual runtime element type of the compiled model's KV cache inputs.
    // Plugins may select a KV cache precision that differs from the ov::hint::kv_cache_precision
    // property value (for example, CPU promotes the cache to bf16 based on the resolved inference
    // precision regardless of the configured hint), so the precision must be read from the
    // compiled model's key_cache input port to match the tensors actually bound by the scheduler.
    ov::element::Type get_kv_cache_element_type() {
        OPENVINO_ASSERT(m_model_runner, "get_kv_cache_element_type() called before model runner is initialized");
        return utils::get_compiled_kv_cache_precision(m_model_runner->get_infer_request().get_compiled_model());
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
    bool _can_publish_kv_only_completed_blocks() const override {
        return false;
    }
    bool eagle_mode_enabled = false;
    bool mtp_mode_enabled = false;
    bool dflash_mode_enabled = false;
    int64_t m_dflash_mask_token_id = -1;
    size_t m_dflash_candidate_position_offset = 1;
    // Windows planned by the DFlash draft update for the next draft forward, by sequence id.
    std::map<uint64_t, detail::DFlashDraftWindow> m_dflash_draft_windows;
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
                                                       std::move(inputs_embedder),
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

    bool supports_embedding_prefix_verification() const override {
        return true;
    }

    void initialize_prefix_cache(const SequenceGroup::Ptr& group) override {
        for (const auto& sequence : group->get_sequences()) {
            sequence->set_prefix_cache_policy(group->get_prompt_len());
        }
    }

    bool _can_publish_kv_only_completed_blocks() const override {
        return true;
    }

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
        validate_prefix_cache_support(scheduler_config, is_validation_mode_enabled);
        m_inputs_embedder = inputs_embedder;
        m_model_runner->set_inputs_embedder(inputs_embedder);
        m_model_input_type = ModelInputType::EMBEDDINGS;
    }

    void set_mtp_draft_positions_needed(bool is_needed) {
        if (m_model_runner) {
            m_model_runner->enable_mtp_draft_positions(is_needed);
        }
    }

protected:
    void validate_prefix_cache_support(const SchedulerConfig& scheduler_config,
                                       bool is_validation_mode_enabled) const {
        OPENVINO_ASSERT(!scheduler_config.enable_prefix_caching || is_validation_mode_enabled ||
                    !m_scheduler->has_linear_attention_cache(),
                "Prefix-enabled MTP draft pipelines with linear-attention state are not supported");
    }
};

// Continuous-batching pipeline for both DFlash models. The validating main pipeline publishes its
// processed length with the exported hidden states. The draft pipeline forwards each request's window
// once per step; its logic is in ContinuousBatchingForSpeculativeDecodingImpl::update_request and multistep
// behind dflash_mode_enabled.
class ContinuousBatchingPipeline::ContinuousBatchingForDFlashDecodingImpl
    : public ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl {
public:
    ContinuousBatchingForDFlashDecodingImpl() = default;

    ContinuousBatchingForDFlashDecodingImpl(const std::shared_ptr<ov::Model>& model,
                                            const Tokenizer& tokenizer,
                                            const GenerationConfig& generation_config,
                                            const SchedulerConfig& scheduler_config,
                                            const std::string& device,
                                            const ov::AnyMap& plugin_config,
                                            bool is_validation_mode_enabled,
                                            int64_t mask_token_id = -1,
                                            size_t candidate_position_offset = 1)
        : ContinuousBatchingForSpeculativeDecodingImpl(model,
                                                       tokenizer,
                                                       generation_config,
                                                       scheduler_config,
                                                       device,
                                                       plugin_config,
                                                       is_validation_mode_enabled) {
        dflash_mode_enabled = true;
        if (!is_validation_mode_enabled) {
            enable_dflash_draft(mask_token_id, candidate_position_offset);
        }
    }

    ContinuousBatchingForDFlashDecodingImpl(const std::shared_ptr<ov::Model>& model,
                                            std::shared_ptr<InputsEmbedder> inputs_embedder,
                                            const Tokenizer& tokenizer,
                                            const GenerationConfig& generation_config,
                                            const SchedulerConfig& scheduler_config,
                                            const std::string& device,
                                            const ov::AnyMap& plugin_config,
                                            bool is_validation_mode_enabled,
                                            int64_t mask_token_id = -1,
                                            size_t candidate_position_offset = 1)
        : ContinuousBatchingForSpeculativeDecodingImpl(model,
                                                       std::move(inputs_embedder),
                                                       tokenizer,
                                                       generation_config,
                                                       scheduler_config,
                                                       device,
                                                       plugin_config,
                                                       is_validation_mode_enabled) {
        dflash_mode_enabled = true;
        if (!is_validation_mode_enabled) {
            // The draft is a text model: sequential positions instead of the target's M-RoPE positions.
            m_model_runner->enable_mtp_draft_positions(true);
            enable_dflash_draft(mask_token_id, candidate_position_offset);
        }
    }

    bool supports_embedding_prefix_verification() const override {
        return true;
    }

    // Only prompt blocks are published: the draft KV past the prompt holds mask and rewound block rows.
    // DFlashDecodingImpl restores main and draft to a common prefix after both requests are added.
    void initialize_prefix_cache(const SequenceGroup::Ptr& group) override {
        for (const auto& sequence : group->get_sequences()) {
            sequence->set_prefix_cache_policy(group->get_prompt_len());
        }
    }

protected:
    bool _can_publish_kv_only_completed_blocks() const override {
        return true;
    }

    void enable_dflash_draft(int64_t mask_token_id, size_t candidate_position_offset) {
        m_dflash_mask_token_id = mask_token_id;
        m_dflash_candidate_position_offset = candidate_position_offset;
        m_model_runner->enable_dflash_draft(candidate_position_offset);
    }
};
}
