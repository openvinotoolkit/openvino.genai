// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/pipeline_base.hpp"

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
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void ContinuousBatchingPipeline::IContinuousBatchingPipeline::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
    m_history_images.clear();
    m_history_videos.clear();
    m_history_image_ids.clear();
    m_history_video_ids.clear();
    if (m_inputs_embedder) {
        m_inputs_embedder->finish_chat();
    }
    m_image_id = 0;
    m_video_id = 0;
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
        auto results_vlm = generate(prompts, images, sampling_params, streamer);
        std::vector<GenerationResult> resutls;
        for (auto& vlm_result : results_vlm) {
            GenerationResult result;
            result.m_generation_ids = std::move(vlm_result.texts);
            result.m_scores = std::move(vlm_result.scores);
            result.perf_metrics = std::move(vlm_result.perf_metrics);
            resutls.push_back(result);
        }
        return resutls;
    }
    std::vector<ov::Tensor> input_ids;
    auto start_time = std::chrono::steady_clock::now();

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

    // TODO Consider moving to method and reuse
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
            res.extended_perf_metrics
        });
    }

    // if streaming was cancelled, prompt/answer of current step shouldn't be presented in history, so let's remove prompt from history
    if (m_is_chat_conversation && encoded[0].m_status == ov::genai::GenerationStatus::CANCEL)
        m_history.pop_back();

    return decoded;
}

std::vector<GenerationResult>
ContinuousBatchingPipeline::IContinuousBatchingPipeline::generate(
    const std::vector<ChatHistory>& histories,
    const std::vector<ov::genai::GenerationConfig>& sampling_params,
    const StreamerVariant& streamer
) {
    // TODO Enable chat history input for embeddings models.
    OPENVINO_ASSERT(m_model_input_type == ModelInputType::TOKENS, "Chat history input is not supported for embeddings models.");
    
    OPENVINO_ASSERT(histories.size() == sampling_params.size(), "Number of histories must match sampling params");
    OPENVINO_ASSERT(!m_tokenizer.get_chat_template().empty(), "Chat template must not be empty when using ChatHistory in generate method.");
    
    auto start_time = std::chrono::steady_clock::now();

    std::vector<ov::Tensor> input_ids;
    input_ids.reserve(histories.size());

    std::vector<MicroSeconds> tokenization_durations;
    static ManualTimer timer("tokenize");
    timer.start();

    for (size_t i = 0; i < histories.size(); i++) {
        OPENVINO_ASSERT(sampling_params[i].apply_chat_template, "Chat template must be applied when using ChatHistory in generate method.");
        OPENVINO_ASSERT(!histories[i].empty(), "Chat history must not be empty when using ChatHistory in generate method.");
        const auto encode_start = std::chrono::steady_clock::now();
        constexpr bool add_generation_prompt = true;
        std::string templated_history = m_tokenizer.apply_chat_template(histories[i], add_generation_prompt);
        input_ids.push_back(
            m_tokenizer.encode(templated_history, add_special_tokens(false)).input_ids
        );
        tokenization_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - encode_start));
    }
    
    timer.end();

    std::vector<EncodedGenerationResult> encoded_results = generate(input_ids, sampling_params, streamer);

    std::vector<GenerationResult> decoded_results;
    decoded_results.reserve(encoded_results.size());
    for (size_t i = 0; i < encoded_results.size(); ++i) {
        EncodedGenerationResult encoded_result = encoded_results[i];

        auto& perf_metrics = encoded_result.perf_metrics;
        auto& raw_counters = perf_metrics.raw_metrics;
        raw_counters.tokenization_durations.emplace_back(tokenization_durations[i]);

        std::vector<std::string> decoded_outputs;
        decoded_outputs.reserve(encoded_result.m_generation_ids.size());
        for (size_t idx = 0; idx < encoded_result.m_generation_ids.size(); ++idx) {
            const auto decode_start = std::chrono::steady_clock::now();
            decoded_outputs.push_back(m_tokenizer.decode(encoded_result.m_generation_ids.at(idx)));

            raw_counters.detokenization_durations.emplace_back(std::chrono::steady_clock::now() - decode_start);
        }

        // The same perf metrics for each sequence, only tokenization/detokenization will differ.
        perf_metrics.raw_metrics.generate_durations.clear();
        perf_metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time));
        // Reevaluate taking into accound tokenization/detokenization times.
        perf_metrics.m_evaluated = false;
        perf_metrics.evaluate_statistics(start_time);

        decoded_results.push_back(GenerationResult{
            encoded_result.m_request_id,
            std::move(decoded_outputs),
            std::move(encoded_result.m_scores),
            encoded_result.m_status,
            std::move(perf_metrics),
            std::move(encoded_result.extended_perf_metrics)
        });
    }

    return decoded_results;
}

std::vector<VLMDecodedResults>
ContinuousBatchingPipeline::IContinuousBatchingPipeline::generate(
             const std::vector<std::string>& prompts,
             const std::vector<std::vector<ov::Tensor>>& images_vector,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer) {
    return generate(prompts, images_vector, {{}}, sampling_params, streamer);
}

std::vector<VLMDecodedResults>
ContinuousBatchingPipeline::IContinuousBatchingPipeline::generate(
             const std::vector<std::string>& prompts,
             const std::vector<std::vector<ov::Tensor>>& images_vector,
             const std::vector<std::vector<ov::Tensor>>& videos_vector,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer)  {
    auto generate_start_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(m_model_input_type == ModelInputType::EMBEDDINGS);

    OPENVINO_ASSERT(prompts.size() == sampling_params.size(), "Number of prompts should be equal to the number of generation configs.");
    OPENVINO_ASSERT(prompts.size() == images_vector.size() && prompts.size() == videos_vector.size(), "Number of prompts should be equal to the number of images or video vectors.");

    std::vector<ov::Tensor> input_embeds_list;
    std::vector<ov::Tensor> token_type_ids_list;
    
    std::vector<VLMPerfMetrics> vlm_perf_metrics(prompts.size());
    std::vector<EncodedImage> encoded_images = {};
    std::vector<EncodedVideo> encoded_videos = {};

    if (m_is_chat_conversation) {
        OPENVINO_ASSERT(1 == prompts.size(), "Can't chat with multiple prompts");
        const auto& prompt = prompts[0];
        auto start_get_inputs_embeds = std::chrono::steady_clock::now();

        encoded_images = m_inputs_embedder->encode_images(images_vector[0]);
        m_history_images.insert(m_history_images.end(), encoded_images.begin(), encoded_images.end());

        encoded_videos = m_inputs_embedder->encode_videos(videos_vector[0]);
        m_history_videos.insert(m_history_videos.end(), encoded_videos.begin(), encoded_videos.end());

        auto [unified_prompt, image_sequence, video_sequence] = m_inputs_embedder->normalize_prompt(prompt, m_image_id, m_video_id, encoded_images, encoded_videos);

        m_history.push_back({{"role", "user"}, {"content", unified_prompt}});
        m_history_image_ids.insert(m_history_image_ids.end(), image_sequence.begin(), image_sequence.end());
        m_history_video_ids.insert(m_history_video_ids.end(), video_sequence.begin(), video_sequence.end());

        std::string templated_history = m_tokenizer.apply_chat_template(m_history, true);

        m_inputs_embedder->set_apply_chat_template_status(false);
        if (m_inputs_embedder->has_token_type_ids()) {
            auto [embeds, tt_ids] = m_inputs_embedder->get_inputs_embeds_with_token_type_ids(templated_history, m_history_images, vlm_perf_metrics[0], images_vector.size() > 0, m_history_image_ids);
            input_embeds_list.push_back(std::move(embeds));
            token_type_ids_list.push_back(std::move(tt_ids));
        } else {
            input_embeds_list.emplace_back(m_inputs_embedder->get_inputs_embeds(templated_history,
                                                                    m_history_images,
                                                                    m_history_videos,
                                                                    vlm_perf_metrics[0],
                                                                    true,
                                                                    m_history_image_ids,
                                                                    m_history_video_ids));
        }

        auto end_get_inputs_embeds = std::chrono::steady_clock::now();
        vlm_perf_metrics[0].vlm_raw_metrics.prepare_embeddings_durations.emplace_back(PerfMetrics::get_microsec(end_get_inputs_embeds - start_get_inputs_embeds));

    } else {
        for (size_t i = 0; i < prompts.size(); i++) {
            const auto& prompt = prompts[i];
            auto start_get_inputs_embeds = std::chrono::steady_clock::now();
            
            auto images_to_encode = images_vector.size() > 0 ? images_vector[i] : std::vector<ov::Tensor>{};
            auto videos_to_encode = videos_vector.size() > 0 ? videos_vector[i] : std::vector<ov::Tensor>{};
            const auto encoded_images = m_inputs_embedder->encode_images(images_to_encode);
            const auto encoded_videos = m_inputs_embedder->encode_videos(videos_to_encode);

            auto [unified_prompt, image_sequence, video_sequence] = m_inputs_embedder->normalize_prompt(prompt, m_image_id, m_video_id, encoded_images, encoded_videos);

            m_inputs_embedder->set_apply_chat_template_status(sampling_params[i].apply_chat_template);

            if (m_inputs_embedder->has_token_type_ids()) {
                auto [embeds, tt_ids] = m_inputs_embedder->get_inputs_embeds_with_token_type_ids(unified_prompt, encoded_images, vlm_perf_metrics[i], true, image_sequence);
                input_embeds_list.push_back(std::move(embeds));
                token_type_ids_list.push_back(std::move(tt_ids));
            } else {
                input_embeds_list.emplace_back(m_inputs_embedder->get_inputs_embeds(unified_prompt, encoded_images, encoded_videos, vlm_perf_metrics[i], true, image_sequence, video_sequence));
            }
        
            auto end_get_inputs_embeds = std::chrono::steady_clock::now();
            vlm_perf_metrics[i].vlm_raw_metrics.prepare_embeddings_durations.emplace_back(PerfMetrics::get_microsec(end_get_inputs_embeds - start_get_inputs_embeds));
        }
    }
    std::vector<VLMDecodedResults> results;
    std::vector<EncodedGenerationResult> encoded_results = generate(input_embeds_list, sampling_params, streamer, token_type_ids_list);
    for (size_t i = 0; i < prompts.size(); i++) {
        auto result = encoded_results[i];
        VLMDecodedResults gen_result;
        gen_result.perf_metrics = result.perf_metrics;

        gen_result.perf_metrics.vlm_raw_metrics = vlm_perf_metrics[i].vlm_raw_metrics;
        gen_result.perf_metrics.raw_metrics.tokenization_durations = vlm_perf_metrics[i].raw_metrics.tokenization_durations;
        gen_result.perf_metrics.raw_metrics.detokenization_durations = vlm_perf_metrics[i].raw_metrics.detokenization_durations;
        
        auto decode_start_time = std::chrono::steady_clock::now();
        for (size_t idx = 0; idx < result.m_generation_ids.size(); ++idx) {
            gen_result.texts.push_back(m_tokenizer.decode(result.m_generation_ids.at(idx)));
            gen_result.scores.push_back(result.m_scores.at(idx));
        }
        auto decode_end_time = std::chrono::steady_clock::now();
        gen_result.perf_metrics.raw_metrics.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_end_time - decode_start_time));
        
        gen_result.perf_metrics.m_evaluated = false;
        gen_result.perf_metrics.evaluate_statistics(generate_start_time);

        results.emplace_back(gen_result);
    }
    if (m_is_chat_conversation) {
        m_inputs_embedder->update_chat_history(results[0].texts[0], encoded_results[0].m_status);
        if (encoded_results[0].m_status != ov::genai::GenerationStatus::CANCEL) {
            m_image_id += encoded_images.size();
            m_video_id += encoded_videos.size();
            m_history.push_back({{"role", "assistant"}, {"content", results[0].texts[0]}});
        }
        else {
            m_history.pop_back();
            for (size_t idx = 0; idx < encoded_images.size(); idx++) {
                m_history_image_ids.pop_back();
                m_history_images.pop_back();
            }
        }
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
    std::optional<ov::Tensor> token_type_ids;
    {
        std::lock_guard<std::mutex> lock(m_embeddings_mutex);
        m_inputs_embedder->set_apply_chat_template_status(sampling_params.apply_chat_template);
        const auto encoded_images = m_inputs_embedder->encode_images(rgbs);

        const auto [unified_prompt, image_sequence, video_sequence] = m_inputs_embedder->normalize_prompt(prompt, 0, encoded_images);
        if (m_inputs_embedder->has_token_type_ids()) {
            std::tie(inputs, token_type_ids) = m_inputs_embedder->get_inputs_embeds_with_token_type_ids(unified_prompt, encoded_images, metrics, true, image_sequence);
        } else {
            inputs = m_inputs_embedder->get_inputs_embeds(unified_prompt, encoded_images, metrics, true, image_sequence);
        }
    }
    return add_request(request_id, inputs, sampling_params, token_type_ids);
}

GenerationHandle 
ContinuousBatchingPipeline::IContinuousBatchingPipeline::add_request(uint64_t request_id,
                                        const std::string& prompt,
                                        const std::vector<ov::Tensor>& images,
                                        const std::vector<ov::Tensor>& videos,
                                        GenerationConfig sampling_params) {
    OPENVINO_ASSERT(m_model_input_type == ModelInputType::EMBEDDINGS, "Model doesn't support embeddings.");
    ov::genai::VLMPerfMetrics metrics;
    ov::Tensor inputs;
    {
        std::lock_guard<std::mutex> lock(m_embeddings_mutex);
        m_inputs_embedder->set_apply_chat_template_status(sampling_params.apply_chat_template);
        const auto encoded_images = m_inputs_embedder->encode_images(images);
        const auto encoded_videos = m_inputs_embedder->encode_videos(videos);

        const auto [unified_prompt, image_sequence, video_sequence] = m_inputs_embedder->normalize_prompt(prompt, 0, 0, encoded_images, encoded_videos);
        inputs = m_inputs_embedder->get_inputs_embeds(unified_prompt, encoded_images, encoded_videos, metrics, true, image_sequence, video_sequence);
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

ContinuousBatchingPipeline::IContinuousBatchingPipeline::~IContinuousBatchingPipeline() {
    m_tokenizer = {};
    utils::release_core_plugin(m_device);
}

}
