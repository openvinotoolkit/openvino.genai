// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "continuous_batching/pipeline_base.hpp"

#include "openvino/genai/lora_adapter.hpp"
#include "continuous_batching/cache_eviction.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class ContinuousBatchingPipeline::ContinuousBatchingImpl : public ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<ModelRunner> m_model_runner;
    std::optional<AdapterController> m_adapter_controller;
    std::shared_ptr<Sampler> m_sampler;

    // current requests to process
    std::vector<SequenceGroup::Ptr> m_requests;
    // requests added to the pipeline that will be added to m_requests in the next iteration
    std::vector<SequenceGroup::Ptr> m_awaiting_requests;
    // Mutex protecting access to m_awaiting_requests, so add_request and step methods can be called from different threads
    std::mutex m_awaiting_requests_mutex;

    std::map<size_t, CacheEvictionAlgorithm> m_seq_group_id_to_cache_eviction_algo_map;

    static const size_t AVG_CACHE_USAGE_WINDOW_SIZE_IN_STEPS = 1000;
    std::deque<float> m_previous_step_cache_usages;

    // for perf metrics
    float m_load_time_ms = 0.0f;
    size_t m_batch_size = 0; // stored number of processed tokens on last step

    // flag to enable validation mode for sampler
    bool m_is_validation_mode_enabled = false;

    size_t m_num_decoder_layers = 0;
    size_t m_block_size = 0;

    // Pre-allocated per-layer storages for the per-token cache re-rotation deltas used in cache eviction case
    std::vector<ov::Tensor> m_rotation_deltas_stores;

    std::map<size_t, std::vector<std::set<size_t>>> m_previous_evicted_block_logical_indices_per_sequence;
    std::map<size_t, size_t> m_previous_num_blocks_before_eviction_per_sequence;

    std::vector<std::map<size_t, std::vector<size_t>>> m_current_step_rotated_block_indices_per_sequence;
    std::vector<ov::Tensor> m_current_step_rotation_deltas;

    std::shared_ptr<ov::genai::CacheRotationCalculator> m_cache_rotation_calculator;


#ifdef DEBUG_CACHE_STATE_DUMP
    size_t step_count = 0;
#endif

    // used by tests only
    ContinuousBatchingImpl() = default;

    void initialize_pipeline(std::shared_ptr<ov::Model> model,
                             const SchedulerConfig& scheduler_config,
                             const std::string& device,
                             const ov::AnyMap& plugin_config);

    /**
     * Pulls requests from awaiting queue to running queue
     * Should be called within each call of step()
     */
    virtual void _pull_awaiting_requests();

    /**
     * Releases non-running (finished, dropped or OOM) requests from running queue
     */
    void _free_non_running_requests();

    /**
     * Notify dropped requests by pushing empty output
     */
    void _notify_requests_dropped_by_handle();

    /**
     * Handles 'echo' generation parameter
     */
    void _fill_prompt_log_probs(std::vector<SequenceGroup::Ptr>& sequence_groups, ov::Tensor& logits);

    /**
     * Performs KV cache eviction is enabled / requireed
     */
    void _maybe_evict_cache_blocks(const SchedulerConfig& sched_config, const Scheduler::Output& scheduler_output);

    void _register_step_cache_usage(float step_cache_usage);
    void _reset_cache_usage_statistics();
    float _get_current_running_average_cache_usage() const;
    void _compute_cache_rotation_data(const std::vector<SequenceGroup::Ptr>& sequence_groups, const Scheduler::Output& scheduler_output);
    void _prepare_rotation_data_storage(const SchedulerConfig& normalized_config, size_t embedding_size);

    virtual void drop_requests();

public:
    ContinuousBatchingImpl(const std::shared_ptr<ov::Model>& model,
                           const Tokenizer& tokenizer,
                           const SchedulerConfig& scheduler_config,
                           const std::string& device,
                           const ov::AnyMap& properties,
                           const ov::genai::GenerationConfig& generation_config,
                           bool is_validation_mode_enabled = false);

    ContinuousBatchingImpl(const std::shared_ptr<ov::Model>& model,
                           std::shared_ptr<InputsEmbedder> inputs_embedder,
                           const Tokenizer& tokenizer,
                           const SchedulerConfig& scheduler_config,
                           const std::string& device,
                           const ov::AnyMap& properties,
                           const ov::genai::GenerationConfig& generation_config,
                           bool is_validation_mode_enabled = false);
    
    virtual ~ContinuousBatchingImpl();

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

    bool has_non_finished_requests() override;

    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             std::optional<std::vector<ov::Tensor>> token_type_ids = std::nullopt) override;

    /**
     * Updates LoRA adapters for current generation call
     */
    void set_adapters(const std::optional<AdapterConfig>& adapters);

    std::vector<SequenceGroup::Ptr> get_awaiting_requests();
};
} // namespace ov::genai
