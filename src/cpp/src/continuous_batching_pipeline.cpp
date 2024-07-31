// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mutex>
#include <memory>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "cache_manager.hpp"
#include "sampler.hpp"
#include "model_runner.hpp"
#include "scheduler.hpp"
#include "timer.hpp"
#include "debug_utils.hpp"

using namespace ov::genai;

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, DeviceConfig& device_config);

class ContinuousBatchingPipeline::Impl {
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<CacheManager> m_cache_manager;
    std::shared_ptr<ModelRunner> m_model_runner;
    std::shared_ptr<Sampler> m_sampler;
    bool is_validation_mode_enabled = false;

    // TODO (mzegla): GenerationConfig is request specific object
    // and pipeline only uses default rng_seed. 
    // ov::genai::GenerationConfig m_generation_config;

    PipelineMetrics m_pipeline_metrics;

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
    std::mutex m_awaiting_requests_mutex;


    void _free_non_running_requests() {
        std::vector<SequenceGroup::Ptr>::iterator requests_iterator = m_requests.begin();
        while (requests_iterator != m_requests.end()) {
            const auto& request = *requests_iterator;
            if(request->has_finished() || request->out_of_memory() || request->handle_dropped()) {
                for (const auto& sequence: request->get_sequences()) {
                    m_scheduler->free_sequence(sequence->get_id());
                }
                m_sampler->clear_beam_search_info(request->get_request_id());
                requests_iterator = m_requests.erase(requests_iterator);
            } else {
                requests_iterator++;
            }
        }
    }

public:
    Impl(const std::string& models_path,
         const Tokenizer& tokenizer,
         const SchedulerConfig& scheduler_config,
         const std::string& device,
         const ov::AnyMap& plugin_config) {
        ov::Core core;

        // The model can be compiled for GPU as well
        std::shared_ptr<ov::Model> model = core.read_model(models_path + "/openvino_model.xml");

        DeviceConfig device_config(core, scheduler_config, device);

        apply_paged_attention_transformations(model, device_config);

        ov::InferRequest infer_request = core.compile_model(model, device_config.get_device(), plugin_config).create_infer_request();

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
        // todo:iefode
        // m_sampler->set_seed(m_generation_config.rng_seed);
        m_sampler->set_seed(0);

        // read default generation config
    }

    Impl(const std::string& models_path, const SchedulerConfig& scheduler_config, const std::string& device, const ov::AnyMap& plugin_config)
        : Impl{models_path, Tokenizer(models_path), scheduler_config, device, plugin_config} {}


    PipelineMetrics get_metrics() const {
        return m_pipeline_metrics;
    }

    GenerationHandle add_request(uint64_t request_id, ov::Tensor input_ids, ov::genai::GenerationConfig sampling_params) {
        sampling_params.validate();

        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(request_id, input_ids,
                                                                            sampling_params, m_scheduler->get_config().block_size);
        {
            std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
            m_awaiting_requests.push_back(sequence_group);
        }
        return std::make_unique<GenerationHandleImpl>(sequence_group->get_generation_stream(), sampling_params);
    }

    void step() {
        static ManualTimer step_timer("step()");
        step_timer.start();

        // Pull awaiting requests
        {
            std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
            m_requests.insert(m_requests.end(), m_awaiting_requests.begin(), m_awaiting_requests.end());
            m_awaiting_requests.clear();
        }

        m_pipeline_metrics.requests = m_requests.size();
        Scheduler::Output scheduler_output;
        {
            static ManualTimer timer("scheduling");
            timer.start();
            // todo: iefode: to move to other place?
            m_scheduler->clean_empty_blocks(m_requests);
            scheduler_output = m_scheduler->schedule(m_requests);
            m_pipeline_metrics.scheduled_requests = scheduler_output.m_scheduled_sequence_groups_ids.size();
            m_pipeline_metrics.cache_usage = scheduler_output.m_cache_usage;
            m_cache_manager->copy_blocks(scheduler_output.m_block_copy_map);
            timer.end();
        }

        // if no tokens were scheduled, we are out of memory
        if (scheduler_output.m_total_num_scheduled_tokens == 0) {
            for (size_t i = 0; i < m_requests.size(); ++i) {
                SequenceGroup::Ptr sequence_group = m_requests[i];
                sequence_group->set_out_of_memory();
                sequence_group->notify_handle();
            }
            _free_non_running_requests();
            return;
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
            sampler_output = m_sampler->sample(m_requests, logits, is_validation_mode_enabled);
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

        // free non running requests for current step

        {
            static ManualTimer timer("free non running requests");
            timer.start();
            _free_non_running_requests();
            timer.end();
        }

        step_timer.end();
    }

    bool has_non_finished_requests() {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        return !m_awaiting_requests.empty() || !m_requests.empty();
    }

    std::vector<GenerationHandle> generate(const std::vector<ov::Tensor> prompts, std::vector<ov::genai::GenerationConfig> sampling_params) {
        OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
        OPENVINO_ASSERT(prompts.size() == sampling_params.size());

        std::vector<GenerationHandle> generations;
        for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
            generations.push_back(add_request(request_id, prompts[request_id], sampling_params[request_id]));
        }

        while (has_non_finished_requests()) {
            step();
        }

        return generations;

    }

    std::vector<ContinuousBatchingPipeline::GeneratedSequence> get_generated_sequences() {
        std::vector<ContinuousBatchingPipeline::GeneratedSequence> result;
        for (const auto& request : m_requests) {
            const auto request_id = request->get_request_id();
            for (const auto& sequence : request->get_sequences()) {
                auto generated_ids = sequence->get_generated_ids();
                auto log_probs = sequence->get_log_probs();
                result.emplace_back(request_id, sequence->get_grouped_id(), generated_ids, log_probs);
            }
        }
        return result;

    }

    ContinuousBatchingPipeline::UpdateSeqResult
    update_generated_sequence(const ContinuousBatchingPipeline::GeneratedSequence& candidate_sequence) {
        // Pull awaiting requests
        if (!m_awaiting_requests.empty()) {
            std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
            m_requests.insert(m_requests.end(), m_awaiting_requests.begin(), m_awaiting_requests.end());
            m_awaiting_requests.clear();
        }
        for (auto& request : m_requests) {
            if (candidate_sequence.request_id == request->get_request_id()) {
                bool is_seq_exists = false;
                // todo: iefode: multiseq
                size_t to_remove_tokens = 0, to_insert_tokens = 0;
                for (auto& sequence : request->get_sequences()) {
                    if (candidate_sequence.sequence_id == sequence->get_grouped_id()) {
                        is_seq_exists = true;
                        auto present_ids = sequence->get_generated_ids();
                        const auto& candidate_ids = candidate_sequence.token_ids;

                        // remove extra tokens from sequence
                        {
                            auto token_idx = std::min(present_ids.size(), candidate_ids.size());
                            if (token_idx) {
                                while (token_idx-- > 0) {
                                    if (present_ids[token_idx] == candidate_ids[token_idx]) {
                                        break;
                                    }
                                }
                                to_remove_tokens = present_ids.size() - (token_idx + 1);
                                if (to_remove_tokens > 0) {
                                    sequence->remove_last_n_tokens(to_remove_tokens);
                                    present_ids = sequence->get_generated_ids();
                                }
                            }
                        }
                        // insert new tokens to sequence
                        {
                            OPENVINO_ASSERT(candidate_ids.size() >= present_ids.size());
                            const auto& candidate_log_probs = candidate_sequence.log_probs;
                            const size_t start_id = std::min(present_ids.size(), candidate_ids.size()),
                                         stop_id = std::max(present_ids.size(), candidate_ids.size());
                            to_insert_tokens = stop_id - start_id;
                            for (size_t i = start_id; i < stop_id; ++i) {
                                sequence->append_token(candidate_ids[i], candidate_log_probs[i]);
                            }
                        }
                    }
                    break;
                }
                if (!is_seq_exists) {
                    Sequence::Ptr new_sequence(new Sequence(candidate_sequence.sequence_id));
                    const auto& generated_tokens = candidate_sequence.token_ids;
                    const auto& generated_log_probs = candidate_sequence.log_probs;
                    for (size_t i = 0; i < generated_tokens.size(); ++i) {
                        new_sequence->append_token(generated_tokens[i], generated_log_probs[i]);
                    }
                    request->add_sequence(new_sequence);
                }
                if (to_remove_tokens > 0)
                    request->decrease_processed_tokens(to_remove_tokens);
                // to validate tokens/extend kv-cache before generation
                if (request->get_num_processed_tokens() > 0) {
                    // in case of non-prompt we need to take prev tokens + token to validate
                    ++to_insert_tokens;
                }
                request->set_validation_len(to_insert_tokens);
                return ContinuousBatchingPipeline::UpdateSeqResult(to_insert_tokens, to_remove_tokens);
            }
        }
    }

    void enable_validation_mode() {
        is_validation_mode_enabled = true;
    }
};

ContinuousBatchingPipeline::ContinuousBatchingPipeline( const std::string& models_path,
                                                        const SchedulerConfig& scheduler_config,
                                                        const std::string& device,
                                                        const ov::AnyMap& plugin_config ) {
    m_impl = std::make_shared<Impl>(models_path, scheduler_config, device, plugin_config);
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline(
    const std::string& model_path,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config
) : m_impl{std::make_shared<Impl>(model_path, scheduler_config, device, plugin_config)} {
    m_tokenizer = tokenizer;
}

ov::genai::PipelineMetrics ContinuousBatchingPipeline::get_metrics() const {
    return m_impl->get_metrics();
}

GenerationHandle ContinuousBatchingPipeline::add_request(uint64_t request_id, ov::Tensor tokenized_prompt, ov::genai::GenerationConfig sampling_params) {
    return m_impl->add_request(request_id, tokenized_prompt, sampling_params);
}

GenerationHandle
ContinuousBatchingPipeline::add_request(
    uint64_t request_id,
    std::string prompt,
    ov::genai::GenerationConfig sampling_params) {
    sampling_params.validate();
    ov::Tensor encoded_prompt = encode(prompt);
    return add_request(request_id, encoded_prompt, sampling_params);
}

void ContinuousBatchingPipeline::step() {
    m_impl->step();
}

bool ContinuousBatchingPipeline::has_non_finished_requests() {
    return m_impl->has_non_finished_requests();
}

std::vector<ov::genai::GenerationHandle>
ContinuousBatchingPipeline::generate_sequences(
    const std::vector<ov::Tensor> tokenized_prompts,
    std::vector<ov::genai::GenerationConfig> sampling_params) {
    return m_impl->generate(tokenized_prompts, sampling_params);
}

std::vector<ContinuousBatchingPipeline::GeneratedSequence> ContinuousBatchingPipeline::get_generated_sequences() {
    return m_impl->get_generated_sequences();
}

ContinuousBatchingPipeline::UpdateSeqResult
ContinuousBatchingPipeline::update_generated_sequence(const ContinuousBatchingPipeline::GeneratedSequence& new_sequence) {
    return m_impl->update_generated_sequence(new_sequence);
}

void ContinuousBatchingPipeline::enable_validation_mode() {
    m_impl->enable_validation_mode();
}