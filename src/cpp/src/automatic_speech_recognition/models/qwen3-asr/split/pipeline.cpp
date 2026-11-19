// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

#include <algorithm>

#include "utils.hpp"

namespace {

ov::genai::ContinuousBatchingPipeline create_continuous_batching_pipeline(const std::filesystem::path& models_path,
                                                                          const ov::genai::Tokenizer& tokenizer,
                                                                          const std::string& device,
                                                                          const ov::AnyMap& properties) {
    auto [properties_without_backend, attention_backend] = ov::genai::utils::extract_attention_backend(properties);
    OPENVINO_ASSERT(attention_backend == ov::genai::PA_BACKEND,
                    "Qwen3ASRContinuousBatchingAdapter requires '",
                    ov::genai::PA_BACKEND,
                    "' attention backend, got '",
                    attention_backend,
                    "'");
    ov::genai::erase_allowed_asr_ctor_properties(properties_without_backend);
    auto [plugin_properties, scheduler_config] =
        ov::genai::utils::extract_scheduler_config(properties_without_backend,
                                                   ov::genai::utils::get_latency_oriented_scheduler_config());
    return ov::genai::ContinuousBatchingPipeline{models_path, tokenizer, scheduler_config, device, plugin_properties};
}

std::unique_ptr<ov::genai::VLMPipeline> create_sdpa_vlm_pipeline(const std::filesystem::path& models_path,
                                                                 const std::string& device,
                                                                 const ov::AnyMap& properties) {
    // VLMPipeline would create its own ContinuousBatchingPipeline for PagedAttention backend.
    OPENVINO_ASSERT(ov::genai::utils::extract_attention_backend(properties).second == ov::genai::SDPA_BACKEND,
                    "Qwen3ASRVLMPipelineAdapter requires '",
                    ov::genai::SDPA_BACKEND,
                    "' attention backend");
    ov::AnyMap vlm_properties = properties;
    ov::genai::erase_allowed_asr_ctor_properties(vlm_properties);
    return std::make_unique<ov::genai::VLMPipeline>(models_path, device, vlm_properties);
}

ov::genai::GenerationConfig create_chunk_generation_config(const ov::genai::ASRGenerationConfig& config) {
    ov::genai::GenerationConfig chunk_config = config;
    // The prompt is already formatted by Qwen3ASRPipelineBase::build_text_prompt().
    chunk_config.apply_chat_template = false;
    return chunk_config;
}

// Read-only tensor wraps chunk samples without copying.
ov::Tensor wrap_chunk_audio(const ov::genai::AudioChunk& chunk) {
    return ov::Tensor{ov::element::f32, {chunk.wav.size()}, chunk.wav.data()};
}

template <typename T>
void append(std::vector<T>& destination, const std::vector<T>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

/**
 * @brief Accumulate metrics of requests processed in one generate() call.
 * Generation counters are shared by all requests of the call, so they are taken from the first result.
 */
void append_perf_metrics(const std::vector<ov::genai::VLMDecodedResults>& results,
                         ov::genai::ASRPerfMetrics& perf_metrics) {
    ov::genai::RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    ov::genai::ASRRawPerfMetrics& asr_raw_metrics = perf_metrics.asr_raw_metrics;

    const ov::genai::RawPerfMetrics& generation_metrics = results.front().perf_metrics.raw_metrics;
    raw_metrics.m_inference_durations[0] += generation_metrics.m_inference_durations.at(0);
    append(raw_metrics.m_new_token_times, generation_metrics.m_new_token_times);
    append(raw_metrics.m_batch_sizes, generation_metrics.m_batch_sizes);
    append(raw_metrics.m_token_infer_durations, generation_metrics.m_token_infer_durations);
    append(raw_metrics.m_sampling_durations, generation_metrics.m_sampling_durations);
    append(asr_raw_metrics.decode_inference_durations, generation_metrics.m_token_infer_durations);

    for (const ov::genai::VLMDecodedResults& result : results) {
        append(raw_metrics.tokenization_durations, result.perf_metrics.raw_metrics.tokenization_durations);
        append(raw_metrics.detokenization_durations, result.perf_metrics.raw_metrics.detokenization_durations);
        for (const ov::genai::MicroSeconds& duration : result.perf_metrics.vlm_raw_metrics.audio_encoding_durations) {
            raw_metrics.m_inference_durations[0] += duration;
            asr_raw_metrics.encode_inference_durations.push_back(duration);
        }
    }
}

}  // namespace

namespace ov::genai {

Qwen3ASRContinuousBatchingAdapter::Qwen3ASRContinuousBatchingAdapter(const std::filesystem::path& models_path,
                                                                     const std::string& device,
                                                                     const ov::AnyMap& properties)
    : Qwen3ASRPipelineBase(models_path, properties),
      m_impl{create_continuous_batching_pipeline(models_path, m_tokenizer, device, properties)} {}

std::vector<std::string> Qwen3ASRContinuousBatchingAdapter::infer(const std::vector<AudioChunk>& chunks,
                                                                  const ASRGenerationConfig& config,
                                                                  ASRPerfMetrics& perf_metrics,
                                                                  const std::shared_ptr<StreamerBase>& streamer,
                                                                  const std::string& streaming_prefix) {
    const std::string prompt = build_text_prompt(config, streaming_prefix);
    const GenerationConfig chunk_config = create_chunk_generation_config(config);

    std::vector<GenerationHandle> handles;
    handles.reserve(chunks.size());
    for (const AudioChunk& chunk : chunks) {
        // add_request() encodes the audio synchronously, so the non-owning tensor is valid for the call.
        handles.push_back(m_impl.add_request(m_next_request_id++,
                                             prompt,
                                             std::vector<ov::Tensor>{},
                                             std::vector<ov::Tensor>{},
                                             std::vector<ov::Tensor>{wrap_chunk_audio(chunk)},
                                             chunk_config));
    }

    // Qwen3ASRPipelineBase::generate() allows a streamer only for a single chunk.
    const std::shared_ptr<StreamerBase> chunk_streamer = create_chunk_streamer(streamer, config);
    std::vector<int64_t> streamed_ids;
    const auto stream_available_tokens = [&]() {
        GenerationHandle& handle = handles.front();
        while (handle->can_read()) {
            for (const auto& [sequence_id, output] : handle->read()) {
                streamed_ids.insert(streamed_ids.end(), output.generated_ids.begin(), output.generated_ids.end());
                if (chunk_streamer->write(output.generated_ids) != StreamingStatus::RUNNING) {
                    handle->stop();
                }
            }
        }
    };
    const auto requests_running = [&handles]() {
        return std::any_of(handles.begin(), handles.end(), [](const GenerationHandle& handle) {
            return handle->get_status() == GenerationStatus::RUNNING;
        });
    };

    // Concurrent infer() calls take turns: one runs step(), which advances the requests of all calls, and the others
    // wait for that step to end and then re-check their own requests.
    std::unique_lock<std::mutex> lock{m_step_mutex};
    while (true) {
        if (chunk_streamer) {
            stream_available_tokens();
        }
        if (!requests_running()) {
            break;
        }
        if (m_stepping) {
            m_step_cv.wait(lock);
            continue;
        }
        m_stepping = true;
        lock.unlock();
        try {
            if (m_impl.has_non_finished_requests()) {
                m_impl.step();
            }
        } catch (...) {
            lock.lock();
            m_stepping = false;
            m_step_cv.notify_all();
            throw;
        }
        lock.lock();
        m_stepping = false;
        m_step_cv.notify_all();
    }
    lock.unlock();
    if (chunk_streamer) {
        stream_available_tokens();
        chunk_streamer->end();
    }

    std::vector<std::string> texts;
    texts.reserve(handles.size());
    for (GenerationHandle& handle : handles) {
        std::vector<int64_t> generated_ids;
        if (chunk_streamer) {
            generated_ids = std::move(streamed_ids);
        } else {
            const std::vector<GenerationOutput> outputs = handle->read_all();
            OPENVINO_ASSERT(!outputs.empty(), "Qwen3-ASR request produced no output.");
            generated_ids = outputs.front().generated_ids;
        }

        VLMDecodedResults result;
        result.texts.push_back(m_tokenizer.decode(generated_ids));
        result.perf_metrics = handle->get_vlm_perf_metrics();
        if (result.perf_metrics.raw_metrics.m_inference_durations.empty()) {
            result.perf_metrics.raw_metrics.m_inference_durations.emplace_back(0.0f);
        }
        texts.push_back(result.texts.front());
        append_perf_metrics({std::move(result)}, perf_metrics);
    }
    return texts;
}

Qwen3ASRVLMPipelineAdapter::Qwen3ASRVLMPipelineAdapter(const std::filesystem::path& models_path,
                                                       const std::string& device,
                                                       const ov::AnyMap& properties)
    : Qwen3ASRVLMPipelineAdapter(models_path, create_sdpa_vlm_pipeline(models_path, device, properties), properties) {}

Qwen3ASRVLMPipelineAdapter::Qwen3ASRVLMPipelineAdapter(const std::filesystem::path& models_path,
                                                       std::unique_ptr<VLMPipeline> impl,
                                                       const ov::AnyMap& properties)
    : Qwen3ASRPipelineBase(models_path, impl->get_tokenizer(), properties),
      m_impl{std::move(impl)} {}

std::vector<std::string> Qwen3ASRVLMPipelineAdapter::infer(const std::vector<AudioChunk>& chunks,
                                                           const ASRGenerationConfig& config,
                                                           ASRPerfMetrics& perf_metrics,
                                                           const std::shared_ptr<StreamerBase>& streamer,
                                                           const std::string& streaming_prefix) {
    const std::string prompt = build_text_prompt(config, streaming_prefix);
    const GenerationConfig chunk_config = create_chunk_generation_config(config);
    // Qwen3ASRPipelineBase::generate() allows a streamer only for a single chunk.
    const StreamerVariant chunk_streamer =
        streamer ? StreamerVariant{create_chunk_streamer(streamer, config)} : StreamerVariant{std::monostate{}};

    std::vector<std::string> texts;
    texts.reserve(chunks.size());
    for (const AudioChunk& chunk : chunks) {
        VLMDecodedResults result =
            m_impl->generate(prompt, {}, {}, {wrap_chunk_audio(chunk)}, {}, chunk_config, chunk_streamer);
        texts.push_back(result.texts.at(0));
        append_perf_metrics({std::move(result)}, perf_metrics);
    }
    return texts;
}

std::unique_ptr<ASRPipelineImplBase> create_qwen3_asr_split_pipeline(const std::filesystem::path& models_path,
                                                                     const std::string& device,
                                                                     const ov::AnyMap& properties) {
    const auto [properties_without_backend, attention_backend] = utils::extract_attention_backend(properties);

    if (utils::explicitly_requires_paged_attention(properties)) {
        return std::make_unique<Qwen3ASRContinuousBatchingAdapter>(models_path, device, properties);
    }

    if (attention_backend == PA_BACKEND) {
        try {
            // PagedAttention model can be created but not inferred on architectures other than x86 and arm64.
#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
            return std::make_unique<Qwen3ASRContinuousBatchingAdapter>(models_path, device, properties);
#endif
        } catch (const ov::Exception& exception) {
            utils::log_paged_attention_fallback(exception);
        }
    }

    ov::AnyMap sdpa_properties = properties_without_backend;
    sdpa_properties["ATTENTION_BACKEND"] = SDPA_BACKEND;
    return std::make_unique<Qwen3ASRVLMPipelineAdapter>(models_path, device, sdpa_properties);
}

}  // namespace ov::genai
