// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "continuous_batching_impl_interface.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "cache_eviction.hpp"

namespace ov::genai {
class ContinuousBatchingPipeline::ContinuousBatchingImpl : public ContinuousBatchingPipeline::ImplInterface {
protected:
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<CacheManager> m_cache_manager;
    std::shared_ptr<ModelRunner> m_model_runner;
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
    
    // flag to enable validation mode for sampler
    bool m_is_validation_mode_enabled = false;

#ifdef DEBUG_CACHE_STATE_DUMP
    size_t step_count = 0;
#endif

    // used by tests only
    ContinuousBatchingImpl() = default;

    void _free_non_running_requests();
    void _notify_requests_dropped_by_handle();
    void _register_step_cache_usage(float step_cache_usage);
    float _get_current_running_average_cache_usage() const;
    void maybe_evict_cache_blocks(const SchedulerConfig& sched_config);

    void init(std::shared_ptr<ov::Model> model,
              const SchedulerConfig& scheduler_config,
              const ov::AnyMap& plugin_config,
              const DeviceConfig& device_config,
              ov::Core& core);

    virtual void _pull_awaiting_requests();

    void _fill_prompt_log_probs(std::vector<SequenceGroup::Ptr>& sequence_groups, ov::Tensor& logits);
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

    void get_infer_duration(float& duration, int& number);
    void reset_infer_duration();

};
}