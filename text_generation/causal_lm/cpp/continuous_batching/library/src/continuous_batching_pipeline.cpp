// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <mutex>
#include <memory>

#include "continuous_batching_pipeline.hpp"
#include "cache_manager.hpp"
#include "sampler.hpp"
#include "model_runner.hpp"
#include "scheduler.hpp"
#include "timer.hpp"
#include "tokenizer.hpp"

#include "debug_utils.hpp"

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, DeviceConfig& device_config);

class ContinuousBatchingPipeline::Impl {
    std::shared_ptr<Tokenizer> m_tokenizer;
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<CacheManager> m_cache_manager;
    std::shared_ptr<ModelRunner> m_model_runner;
    std::shared_ptr<Sampler> m_sampler;

    GenerationConfig m_generation_config;

    struct PerfTime {
        float m_paged_attention_time_ms = 0.0f;
        float m_matmul_time_ms = 0.0f;
        float m_infer_total_ms = 0.0f;

        ~PerfTime() {
            std::cout << "Inference requests aggregated statistic: " << std::endl;
            std::cout << "Paged attention % of inference execution: " << (m_paged_attention_time_ms / m_infer_total_ms) * 100 << std::endl;
            std::cout << "MatMul % of inference execution: " << (m_matmul_time_ms / m_infer_total_ms) * 100 << std::endl;
            std::cout << "Total inference execution secs: " << m_infer_total_ms / 1000. << std::endl;
            std::cout << std::endl;
        }
    } m_perf;

    // current requests to process
    std::vector<SequenceGroup::Ptr> m_requests;
    // requests added to the pipeline that will be added to m_requests in the next iteration
    std::vector<SequenceGroup::Ptr> m_awaiting_requests;
    // Mutex protecting access to m_awaiting_requests, so add_request and step methods can be called from different threads
    std::mutex m_mutex;


    void _free_non_running_requests() {
        auto new_end = std::remove_if(m_requests.begin(), m_requests.end(), [] (SequenceGroup::CPtr seq_group) -> bool {
            return seq_group->has_finished() || seq_group->out_of_memory();
        });
        m_requests.erase(new_end, m_requests.end());
    }

public:
    Impl(const std::string& models_path, const SchedulerConfig& scheduler_config) {
        ov::Core core;
        m_tokenizer = std::make_shared<Tokenizer>(models_path);

        // The model can be compiled for GPU as well
        std::shared_ptr<ov::Model> model = core.read_model(models_path + "/openvino_model.xml");

        const std::string device = "CPU";
        DeviceConfig device_config(core, scheduler_config, device);

        apply_paged_attention_transformations(model, device_config);
        ov::InferRequest infer_request = core.compile_model(model, device_config.get_device(), ov::enable_profiling(true)).create_infer_request();

        // setup KV caches
        m_cache_manager = std::make_shared<CacheManager>(device_config);
        for (size_t decoder_layer_id = 0; decoder_layer_id < device_config.get_num_layers(); ++decoder_layer_id) {
            infer_request.set_input_tensor(2 + decoder_layer_id * 2, m_cache_manager->get_key_cache(decoder_layer_id));
            infer_request.set_input_tensor(2 + decoder_layer_id * 2 + 1, m_cache_manager->get_value_cache(decoder_layer_id));
        }

        SchedulerConfig updated_config = scheduler_config;
        // update KV number in scheduler config
        if (scheduler_config.num_kv_blocks != device_config.get_num_kv_blocks()) {
            updated_config.num_kv_blocks = device_config.get_num_kv_blocks();
        }

        m_scheduler = std::make_shared<Scheduler>(updated_config);
        // and finally create model runner
        m_model_runner = std::make_shared<ModelRunner>(infer_request, updated_config);
        m_sampler = std::make_shared<Sampler>();
        m_sampler->set_seed(m_generation_config.rng_seed);

        // read default generation config
    }

    GenerationConfig get_config() const {
        return m_generation_config;
    }

    std::shared_ptr<Tokenizer> get_tokenizer() {
        return m_tokenizer;
    }

    GenerationHandle add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params) {
        if (sampling_params.eos_token_id < 0) {
            sampling_params.eos_token_id = m_tokenizer->get_eos_token_id();
        } else {
            OPENVINO_ASSERT(sampling_params.eos_token_id == m_tokenizer->get_eos_token_id(),
                "EOS token ID is different in generation config (", sampling_params.eos_token_id, ") and tokenizer (",
                m_tokenizer->get_eos_token_id(), ")");
        }

        ov::Tensor input_ids;
        {
            static ManualTimer timer("tokenize");
            timer.start();
            input_ids = m_tokenizer->encode(prompt);
            timer.end();
        }

        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(request_id, input_ids,
                                                                            sampling_params, m_scheduler->get_config().block_size);
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            m_awaiting_requests.push_back(sequence_group);
        }
        return std::make_unique<GenerationHandleImpl>(sequence_group->get_generation_stream(), sampling_params);
    }

    void update_requests_buffer() {
        // Remove sequence group if it's handle has been dropped
        m_requests.erase(std::remove_if(m_requests.begin(), m_requests.end(),
            [](const SequenceGroup::Ptr & request) { return request->handle_dropped(); }),
            m_requests.end());
        
        // Pull awaiting requests
        std::lock_guard<std::mutex> lock{m_mutex};
        m_requests.insert(m_requests.end(), m_awaiting_requests.begin(), m_awaiting_requests.end());
        m_awaiting_requests.clear();
    }

    void step() {
        static ManualTimer step_timer("step()");
        step_timer.start();

        update_requests_buffer();

        Scheduler::Output scheduler_output;
        {
            static ManualTimer timer("scheduling");
            timer.start();
            scheduler_output = m_scheduler->schedule(m_requests);
            m_cache_manager->copy_blocks(scheduler_output.m_block_copy_map);
            timer.end();
        }

        // if no tokens were scheduled, we are out of memory
        if (scheduler_output.m_total_num_scheduled_tokens == 0) {
            for (size_t i = 0; i < m_requests.size(); ++i) {
                SequenceGroup::Ptr sequence_group = m_requests[i];
                sequence_group->notify_handle();
                sequence_group->finish_generation_stream(GenerationResultStatus::IGNORED);
            }

            _free_non_running_requests();
        }

        ov::Tensor logits;
        {
            static ManualTimer timer("forward");
            timer.start();
            logits = m_model_runner->forward(m_requests, scheduler_output);
            timer.end();

            // collect detailed statistic
            std::vector<ov::ProfilingInfo> profiling_info = m_model_runner->get_infer_request().get_profiling_info();
            for (const ov::ProfilingInfo& info : profiling_info) {
                double current_time = info.real_time.count();
                if (info.node_type == "PagedAttentionExtension") {
                    m_perf.m_paged_attention_time_ms += current_time;
                } else if (info.node_type == "FullyConnected") {
                    m_perf.m_matmul_time_ms += current_time;
                }
                m_perf.m_infer_total_ms += current_time;
            }
        }

        SamplerOutput sampler_output;
        {
            static ManualTimer timer("sample");
            timer.start();
            sampler_output = m_sampler->sample(m_requests, logits);
            timer.end();
        }

        // process sampler_output (e.g. fork or drop sequences from BlockScheduler)
        {
            static ManualTimer timer("fork / free sequence");
            timer.start();

            for (const auto& pair : sampler_output.m_forked_sequences) {
                uint64_t parent_id = pair.first;
                const std::list<uint64_t>& child_ids = pair.second;
                for (auto & child_id : child_ids)
                    m_scheduler->fork_sequence(parent_id, child_id);
            }

            for (auto seq_id : sampler_output.m_dropped_sequences)
                m_scheduler->free_sequence(seq_id);

            timer.end();
        }

        // perform post-processing of current step

        {
            static ManualTimer timer("create finished results");
            timer.start();

            for (size_t i = 0; i < scheduler_output.m_scheduled_sequence_groups_ids.size(); ++i) {
                uint64_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                SequenceGroup::Ptr sequence_group = m_requests[seq_group_id];
                sequence_group->notify_handle();
                if (sequence_group->has_finished()) {
                   sequence_group->finish_generation_stream(GenerationResultStatus::FINISHED);
                }
            }

            _free_non_running_requests();

            timer.end();
        }

        step_timer.end();
    }

    bool has_running_requests() const {
        return !m_requests.empty();
    }

    bool has_awaiting_requests() {
        std::lock_guard<std::mutex> lock{m_mutex};
        return !m_awaiting_requests.empty();
    }

    std::vector<GenerationResult> generate(const std::vector<std::string> prompts, std::vector<GenerationConfig> sampling_params) {
        OPENVINO_ASSERT(!has_running_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
        OPENVINO_ASSERT(prompts.size() == sampling_params.size());

        std::vector<GenerationHandle> generations;
        for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
            generations.push_back(add_request(request_id, prompts[request_id], sampling_params[request_id]));
        }

        std::vector<GenerationResult> results;
        results.reserve(m_awaiting_requests.size());

        while (has_running_requests() || has_awaiting_requests()) {
            step();
        }

        for (auto& generation : generations) {
            GenerationResult result;
            result.m_request_id = 1;
            std::vector<GenerationOutput> generation_outputs = generation->read_all();

            std::sort(generation_outputs.begin(), generation_outputs.end(), [=] (GenerationOutput& r1, GenerationOutput& r2) {
                return r1.score > r2.score;
            });

            for (auto& generation_output : generation_outputs) {
                std::string output_text = m_tokenizer->decode(generation_output.generated_token_ids);
                result.m_generation_ids.push_back(output_text);
                result.m_scores.push_back(generation_output.score);
            }
            result.m_status = generation->get_finish_status();
            results.push_back(result);
        }

        OPENVINO_ASSERT(results.size() == prompts.size());
        return results;
    }
};

ContinuousBatchingPipeline::ContinuousBatchingPipeline(const std::string& models_path,
                     const SchedulerConfig& scheduler_config) {
    m_impl = std::make_shared<Impl>(models_path, scheduler_config);
}

std::shared_ptr<Tokenizer> ContinuousBatchingPipeline::get_tokenizer() {
    return m_impl->get_tokenizer();
}

GenerationConfig ContinuousBatchingPipeline::get_config() const{
    return m_impl->get_config();
}

GenerationHandle ContinuousBatchingPipeline::add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params) {
    return m_impl->add_request(request_id, prompt, sampling_params);
}

void ContinuousBatchingPipeline::step() {
    m_impl->step();
}

bool ContinuousBatchingPipeline::has_running_requests() const {
    return m_impl->has_running_requests();
}

bool ContinuousBatchingPipeline::has_awaiting_requests() const {
    return m_impl->has_awaiting_requests();
}

std::vector<GenerationResult> ContinuousBatchingPipeline::generate(const std::vector<std::string>& prompts, std::vector<GenerationConfig> sampling_params) {
    return m_impl->generate(prompts, sampling_params);
}
