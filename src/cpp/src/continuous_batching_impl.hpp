// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "icontinuous_batching.hpp"

#include "openvino/genai/lora_adapter.hpp"
#include "cache_eviction.hpp"

namespace ov::genai {

class ContinuousBatchingPipeline::ContinuousBatchingImpl : public ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<CacheManager> m_cache_manager;
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
    size_t m_batch_size = 0; // stored number of scheduled sequences on last step

    // flag to enable validation mode for sampler
    bool m_is_validation_mode_enabled = false;

#ifdef DEBUG_CACHE_STATE_DUMP
    size_t step_count = 0;
#endif

    // used by tests only
    ContinuousBatchingImpl() = default;

    void initialize_pipeline(std::shared_ptr<ov::Model> model,
                             const SchedulerConfig& scheduler_config,
                             const ov::AnyMap& plugin_config,
                             const DeviceConfig& device_config,
                             ov::Core& core);

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
    void _maybe_evict_cache_blocks(const SchedulerConfig& sched_config);

    void _register_step_cache_usage(float step_cache_usage);
    float _get_current_running_average_cache_usage() const;

    virtual void drop_requests();

public:
    ContinuousBatchingImpl(const std::shared_ptr<ov::Model>& model,
                           const Tokenizer& tokenizer,
                           const SchedulerConfig& scheduler_config,
                           const std::string& device,
                           const ov::AnyMap& properties,
                           const ov::genai::GenerationConfig& generation_config,
                           bool is_validation_mode_enabled = false);

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 ov::genai::GenerationConfig sampling_params) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 ov::genai::GenerationConfig sampling_params) override;

    bool has_non_finished_requests() override;

    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer) override;

    /**
     * Updates LoRA adapters for current generation call
     */
    void set_adapters(const std::optional<AdapterConfig>& adapters);
};

} // namespace ov::genai
