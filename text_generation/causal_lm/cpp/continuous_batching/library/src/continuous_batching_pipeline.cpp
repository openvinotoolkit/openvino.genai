// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching_pipeline.hpp"
#include "cache_manager.hpp"
#include "sampler.hpp"
#include "model_runner.hpp"
#include "scheduler.hpp"
#include "timer.hpp"
#include "tokenizer.hpp"

#include "debug_utils.hpp"

namespace {

GenerationResult from_sequence_group(std::shared_ptr<Tokenizer> tokenizer, SequenceGroup::CPtr sequence_group) {
    GenerationResult result;
    result.m_request_id = sequence_group->get_request_id();

    std::vector<Sequence::CPtr> finished_sequences = sequence_group->get_finished_sequences();

    OPENVINO_ASSERT(finished_sequences.size() == sequence_group->num_total_seqs() && sequence_group->has_finished());
    for (size_t sequence_id = 0; sequence_id < finished_sequences.size(); ++sequence_id) {
        Sequence::CPtr sequence = finished_sequences[sequence_id];

        result.m_scores.push_back(sequence->get_beam_search_score(sequence_group->get_sampling_parameters()));

        {
            static ManualTimer timer("detokenize");
            timer.start();
            std::string output_text = tokenizer->decode(sequence->get_generated_ids());
            timer.end();
            result.m_generation_ids.push_back(output_text);
        }
    }

    return result;
}

} // namespace

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

    void _free_finished_requests() {
        auto new_end = std::remove_if(m_requests.begin(), m_requests.end(), [] (SequenceGroup::CPtr seq_group) -> bool {
            return seq_group->has_finished();
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

        m_scheduler = std::make_shared<Scheduler>(scheduler_config);
        // and finally create model runner
        m_model_runner = std::make_shared<ModelRunner>(infer_request, scheduler_config);
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

    void add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params) {
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
        m_requests.push_back(sequence_group);
    }

    std::vector<GenerationResult> step() {
        static ManualTimer step_timer("step()");
        step_timer.start();

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
            for (size_t sequence_group_id = 0; sequence_group_id < m_requests.size(); ++sequence_group_id) {
                m_requests[sequence_group_id]->set_out_of_memory();
            }
            return {};
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
        std::vector<GenerationResult> currently_finished_requests;
        {
            static ManualTimer timer("create finished results");
            timer.start();

            for (size_t i = 0; i < scheduler_output.m_scheduled_sequence_groups_ids.size(); ++i) {
                uint64_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                SequenceGroup::CPtr sequence_group = m_requests[seq_group_id];
                if (sequence_group->has_finished()) {
                   currently_finished_requests.push_back(from_sequence_group(m_tokenizer, sequence_group));
                }
            }

            _free_finished_requests();

            timer.end();
        }

        step_timer.end();
        return currently_finished_requests;
    }

    bool has_running_requests() const {
        return !m_requests.empty();
    }

    bool out_of_memory() const {
        for (size_t sequence_group_id = 0; sequence_group_id < m_requests.size(); ++sequence_group_id) {
            if (m_requests[sequence_group_id]->out_of_memory())
                return true;
        }
        return false;
    }

    std::vector<GenerationResult> generate(const std::vector<std::string> prompts, std::vector<GenerationConfig> sampling_params) {
        OPENVINO_ASSERT(!has_running_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
        OPENVINO_ASSERT(prompts.size() == sampling_params.size());

        for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
            add_request(request_id, prompts[request_id], sampling_params[request_id]);
        }

        std::vector<GenerationResult> results;
        results.reserve(m_requests.size());

        while (has_running_requests() && !out_of_memory()) {
            std::vector<GenerationResult> partial_results = step();
            if (partial_results.size() > 0)
                results.insert(results.end(), partial_results.begin(), partial_results.end());
        }

        OPENVINO_ASSERT(!out_of_memory(), "Not enough memory for processing the requests.");

        // sort results according to request_id to return results in order of initial prompts
        std::sort(results.begin(), results.end(), [] (const GenerationResult& r1, const GenerationResult& r2) -> bool {
            return r1.m_request_id < r2.m_request_id;
        });

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

void ContinuousBatchingPipeline::add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params) {
    return m_impl->add_request(request_id, prompt, sampling_params);
}

std::vector<GenerationResult> ContinuousBatchingPipeline::step() {
     return m_impl->step();
}

bool ContinuousBatchingPipeline::has_running_requests() const {
    return m_impl->has_running_requests();
}

std::vector<GenerationResult> ContinuousBatchingPipeline::generate(const std::vector<std::string>& prompts, std::vector<GenerationConfig> sampling_params) {
    return m_impl->generate(prompts, sampling_params);
}
