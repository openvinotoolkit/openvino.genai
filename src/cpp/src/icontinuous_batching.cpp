// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "icontinuous_batching.hpp"
#include "lora_helper.hpp"

namespace ov::genai {

template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

GenerationConfig ContinuousBatchingPipeline::IContinuousBatchingPipeline::get_config() const {
    return m_generation_config;
}

void ContinuousBatchingPipeline::IContinuousBatchingPipeline::set_config(const GenerationConfig& config) {
    m_generation_config = config;
}

PipelineMetrics ContinuousBatchingPipeline::IContinuousBatchingPipeline::get_metrics() const {
    return m_pipeline_metrics;
}

Tokenizer ContinuousBatchingPipeline::IContinuousBatchingPipeline::get_tokenizer() {
    return m_tokenizer;
}

void ContinuousBatchingPipeline::IContinuousBatchingPipeline::start_chat(const std::string& system_message) {
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        OPENVINO_THROW("Chat mode is not supported.");
    }
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void ContinuousBatchingPipeline::IContinuousBatchingPipeline::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
};

std::vector<GenerationResult>
ContinuousBatchingPipeline::IContinuousBatchingPipeline::generate(
    const std::vector<std::string>& prompts,
    std::vector<ov::genai::GenerationConfig> sampling_params,
    const StreamerVariant& streamer) {
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        // TODO: remove this code and within model runner add check: if sequence group type is tokens, 
        // but embedding model is available => compute embeddings first, then pass to LLM
        std::vector<std::vector<ov::Tensor>> images(prompts.size());
        return generate(prompts, images, sampling_params, streamer);
    }
    std::vector<ov::Tensor> input_ids;
    auto start_time =  std::chrono::steady_clock::now();

    std::vector<MicroSeconds> tokenization_durations;
    static ManualTimer timer("tokenize");
    if (m_is_chat_conversation) {
        OPENVINO_ASSERT(1 == prompts.size(), "Can't chat with multiple prompts");
        m_history.push_back({{"role", "user"}, {"content", prompts.at(0)}});
        constexpr bool add_generation_prompt = true;
        std::string history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        timer.start();
        const auto encode_start = std::chrono::steady_clock::now();
        // ov::genai::add_special_tokens(false) is aligned with stateful pipeline
        input_ids.push_back(m_tokenizer.encode(history, ov::genai::add_special_tokens(false)).input_ids);
        tokenization_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - encode_start));
        timer.end();
    } else {
        input_ids.reserve(prompts.size());
        timer.start();
        for (size_t i = 0; i < prompts.size(); i++) {
            const std::string& prompt = prompts.at(i);
            const auto encode_start = std::chrono::steady_clock::now();
            ov::Tensor encoded_inputs;
            if (sampling_params.at(i).apply_chat_template && !m_tokenizer.get_chat_template().empty()) {
                ChatHistory history({{{"role", "user"}, {"content", prompt}}});
                constexpr bool add_generation_prompt = true;
                auto templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
                encoded_inputs = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false)).input_ids;
            } else {
                // in case when chat_template was not found in tokenizer_config.json or set
                std::string input_str(prompt);
                encoded_inputs = m_tokenizer.encode(input_str, ov::genai::add_special_tokens(true)).input_ids;
            }
            input_ids.push_back(encoded_inputs);
            tokenization_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - encode_start));
        }
        timer.end();
    }

    std::vector<EncodedGenerationResult> encoded = generate(input_ids, sampling_params, streamer);

    std::vector<GenerationResult> decoded;
    decoded.reserve(encoded.size());
    for (size_t i = 0; i < encoded.size(); ++i) {
        EncodedGenerationResult res = encoded[i];
        auto& perf_metrics = res.perf_metrics;
        auto& raw_counters = perf_metrics.raw_metrics;
        raw_counters.tokenization_durations.emplace_back(tokenization_durations[i]);

        std::vector<std::string> generated;
        generated.reserve(res.m_generation_ids.size());
        for (size_t idx = 0; idx < res.m_generation_ids.size(); ++idx) {
            const auto decode_start = std::chrono::steady_clock::now();
            generated.push_back(m_tokenizer.decode(res.m_generation_ids.at(idx)));
            raw_counters.detokenization_durations.emplace_back(std::chrono::steady_clock::now() - decode_start);
            if (m_is_chat_conversation && 0 == idx && res.m_status != ov::genai::GenerationStatus::CANCEL) {
                m_history.push_back({{"role", "assistant"}, {"content", generated.back()}});
            }
        }

        // The same perf metrics for each sequence, only tokenization/detokenization will differ.
        perf_metrics.raw_metrics.generate_durations.clear();
        perf_metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time));
        // Reevaluate taking into accound tokenization/detokenization times.
        perf_metrics.m_evaluated = false;
        perf_metrics.evaluate_statistics(start_time);

        decoded.push_back(GenerationResult{
            res.m_request_id,
            std::move(generated),
            std::move(res.m_scores),
            res.m_status,
            perf_metrics,
        });
    }

    // if streaming was cancelled, prompt/answer of current step shouldn't be presented in history, so let's remove prompt from history
    if (m_is_chat_conversation && encoded[0].m_status == ov::genai::GenerationStatus::CANCEL)
        m_history.pop_back();

    return decoded;
}

std::vector<GenerationResult>
ContinuousBatchingPipeline::IContinuousBatchingPipeline::generate(
             const std::vector<std::string>& prompts,
             const std::vector<std::vector<ov::Tensor>>& rgbs_vector,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer)  {
    // TODO: Add performance metrics
    auto generate_start_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(m_model_input_type == ModelInputType::EMBEDDINGS);
    OPENVINO_ASSERT(!m_is_chat_conversation, "Chat mode is not supported.");

    OPENVINO_ASSERT(prompts.size() == sampling_params.size(), "Number of prompts should be equal to the number of generation configs.");
    OPENVINO_ASSERT(prompts.size() == rgbs_vector.size(), "Number of prompts should be equal to the number of images vectors.");

    std::vector<ov::Tensor> input_embeds_list;
    for (size_t i = 0; i < prompts.size(); i++) {
        auto prompt = prompts[i];
        auto rgbs = rgbs_vector[i];

        VLMPerfMetrics perf_metrics;
        input_embeds_list.emplace_back(m_inputs_embedder->get_inputs_embeds(prompt, rgbs, perf_metrics));
    }
    std::vector<GenerationResult> results;
    auto encoded_results = generate(input_embeds_list, sampling_params, streamer);
    for (const auto& result: encoded_results) {
        GenerationResult gen_result;
        for (size_t idx = 0; idx < result.m_generation_ids.size(); ++idx) {
            gen_result.m_generation_ids.push_back(m_tokenizer.decode(result.m_generation_ids.at(idx)));
            gen_result.m_scores.push_back(result.m_scores.at(idx));
            gen_result.m_status = result.m_status;
        }
        results.emplace_back(gen_result);
    }
    return results;
}

GenerationHandle 
ContinuousBatchingPipeline::IContinuousBatchingPipeline::add_request(uint64_t request_id,
                                        const std::string& prompt,
                                        const std::vector<ov::Tensor>& rgbs,
                                        GenerationConfig sampling_params) {
    OPENVINO_ASSERT(m_model_input_type == ModelInputType::EMBEDDINGS, "Model doesn't support embeddings.");
    ov::genai::VLMPerfMetrics metrics;
    ov::Tensor inputs;
    {
        const std::lock_guard<std::mutex> lock(m_inputs_embedder_mutex);
        inputs = m_inputs_embedder->get_inputs_embeds(prompt, rgbs, metrics);
    }
    return add_request(request_id, inputs, sampling_params);
}

void ContinuousBatchingPipeline::IContinuousBatchingPipeline::stream_tokens(
    const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr,
    const GenerationHandle& handle
) {
    if (!streamer_ptr->has_callback() || !handle->can_read()) {
        return;
    }

    const auto streaming_status = streamer_ptr->get_status();

    if (streaming_status == StreamingStatus::CANCEL) {
        handle->cancel();
        return;
    }

    if (streaming_status == StreamingStatus::STOP) {
        handle->stop();
        return;
    }

    std::unordered_map<uint64_t, GenerationOutput> generation_outputs = handle->read();
    OPENVINO_ASSERT(generation_outputs.size() <= 1);
    if (generation_outputs.empty()) {
        return;
    }

    const auto tokens = generation_outputs.begin()->second.generated_ids;
    streamer_ptr->write(tokens);
}

}
