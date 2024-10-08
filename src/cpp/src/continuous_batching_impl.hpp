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

#ifdef DEBUG_CACHE_STATE_DUMP
    size_t step_count = 0;
#endif

    bool m_is_validation_mode_enabled = false;

    void _free_non_running_requests();
    void _notify_requests_dropped_by_handle();
    void _register_step_cache_usage(float step_cache_usage);

    float _get_current_running_average_cache_usage() const;

    void maybe_evict_cache_blocks(const SchedulerConfig& sched_config);

    inline void
    init(std::shared_ptr<ov::Model> model,
         const SchedulerConfig& scheduler_config,
         const ov::AnyMap& plugin_config,
         const DeviceConfig& device_config,
         ov::Core& core) {
        ov::InferRequest infer_request = core.compile_model(model, device_config.get_device(), plugin_config).create_infer_request();

        // setup KV caches
        m_cache_manager = std::make_shared<CacheManager>(device_config, core);
        for (size_t decoder_layer_id = 0; decoder_layer_id < device_config.get_num_layers(); ++decoder_layer_id) {
            infer_request.set_tensor(std::string("key_cache.") + std::to_string(decoder_layer_id), m_cache_manager->get_key_cache(decoder_layer_id));
            infer_request.set_tensor(std::string("value_cache.") + std::to_string(decoder_layer_id), m_cache_manager->get_value_cache(decoder_layer_id));
        }

        SchedulerConfig updated_config = scheduler_config;
        // update KV number in scheduler config
        if (scheduler_config.num_kv_blocks != device_config.get_num_kv_blocks()) {
            updated_config.num_kv_blocks = device_config.get_num_kv_blocks();
        }

        bool can_use_partial_preemption = true;
        if (device_config.get_device().find("GPU") != std::string::npos && !updated_config.dynamic_split_fuse) {
            // in case of executing a `vLLM-like` pipeline, it's better not to use partial eviction on the GPU,
            // as it may lead to performance slowdown
            can_use_partial_preemption = false;
        }

        m_scheduler = std::make_shared<Scheduler>(updated_config, device_config.get_num_layers(), can_use_partial_preemption);
        // and finally create model runner
        bool is_use_cache_eviction = m_scheduler->get_config().use_cache_eviction;
        if (is_use_cache_eviction) {
            m_model_runner = std::make_shared<ModelRunner>(infer_request, updated_config, device_config.get_num_layers(), true);
        } else {
            m_model_runner = std::make_shared<ModelRunner>(infer_request, updated_config, device_config.get_num_layers());
        }
        m_sampler = std::make_shared<Sampler>(m_tokenizer);
        m_sampler->set_seed(m_generation_config.rng_seed);
    };

    void _pull_awaiting_requests();

public:
    ContinuousBatchingImpl(const std::string& models_path,
                           const Tokenizer& tokenizer,
                           const SchedulerConfig& scheduler_config,
                           const std::string& device,
                           const ov::AnyMap& plugin_config);

    ContinuousBatchingImpl(const std::string& models_path,
                           const SchedulerConfig& scheduler_config,
                           const std::string& device,
                           const ov::AnyMap& llm_plugin_config,
                           const ov::AnyMap& tokenizer_plugin_config)
    : ContinuousBatchingImpl{models_path, Tokenizer(models_path, tokenizer_plugin_config), scheduler_config, device, llm_plugin_config} {}

    ContinuousBatchingImpl(ov::Core& core,
                           const std::shared_ptr<ov::Model>& model,
                           const Tokenizer& tokenizer,
                           const DeviceConfig& device_config,
                           const SchedulerConfig& scheduler_config,
                           const std::string& device,
                           const ov::AnyMap& plugin_config,
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

    // for speculative decoding
    void finish_request(int64_t request_id = -1);

    struct GeneratedSequence {
        uint64_t request_id = 0, sequence_id = 0;
        std::vector<int64_t> token_ids;
        std::vector<float> log_probs;

        GeneratedSequence(uint64_t req_id, uint64_t seq_id, const  std::vector<int64_t>& generated_token_ids, const std::vector<float>& generated_log_probs) :
            request_id(req_id),
            sequence_id(seq_id),
            token_ids(generated_token_ids),
            log_probs(generated_log_probs) {};
    };

    struct UpdateSeqResult {
        size_t to_insert, to_remove;
        UpdateSeqResult(size_t _to_insert = 0, size_t _to_remove = 0) : to_insert(_to_insert), to_remove(_to_remove) {};
    };

    std::vector<GeneratedSequence> get_generated_sequences();
    UpdateSeqResult update_generated_sequence(const GeneratedSequence& new_sequence);
};
}