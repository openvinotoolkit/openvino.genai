// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>

#include "visual_language/pipeline_base.hpp"
#include "visual_language/chat_history_state.hpp"
#include "visual_language/qwen3_omni/speech_pipeline.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "logger.hpp"

using namespace ov::genai;

class ov::genai::VLMPipeline::VLMContinuousBatchingAdapter : public ov::genai::VLMPipeline::VLMPipelineBase {
public:
    ContinuousBatchingPipeline m_impl;
    // Optional speech pipeline for Qwen3-Omni
    std::unique_ptr<Qwen3OmniSpeechPipeline> m_speech_pipeline;

    VLMContinuousBatchingAdapter(
        const std::filesystem::path& models_dir,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties
    ): m_impl{
        models_dir,
        scheduler_config,
        device,
        properties} {
        try_init_speech_pipeline(models_dir, device, properties);
    }

    VLMContinuousBatchingAdapter(
        const std::shared_ptr<ov::Model>& language_model,
        const std::filesystem::path& models_dir,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties
    ): m_impl{
        language_model,
        models_dir,
        scheduler_config,
        device,
        properties} {
        try_init_speech_pipeline(models_dir, device, properties);
    }

    VLMContinuousBatchingAdapter(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties,
        const ov::genai::GenerationConfig& generation_config
    ): m_impl{
        models_map,
        tokenizer,
        scheduler_config,
        device,
        config_dir_path,
        properties,
        generation_config} {
        try_init_speech_pipeline(config_dir_path, device, properties);
    }

    VLMContinuousBatchingAdapter(
        const std::shared_ptr<ov::Model>& language_model,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties,
        const ov::genai::GenerationConfig& generation_config
    ): m_impl{
        language_model,
        models_map,
        tokenizer,
        scheduler_config,
        device,
        config_dir_path,
        properties,
        generation_config} {
        try_init_speech_pipeline(config_dir_path, device, properties);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(prompt, images, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        const auto speaker = generation_config.speaker;
        m_impl.encode_audios(m_pending_audios);
        auto start_time = std::chrono::steady_clock::now();
        auto result = m_impl.generate({prompt}, {images}, {videos}, {std::move(generation_config)}, streamer)[0];
        auto stop_time = std::chrono::steady_clock::now();
        return build_decoded_results(result, start_time, stop_time, speaker);
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(history, images, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        const auto speaker = generation_config.speaker;
        m_impl.encode_audios(m_pending_audios);
        auto start_time = std::chrono::steady_clock::now();
        ChatHistoryInternalState::get_or_create(history);
        auto result = m_impl.generate({history}, {images}, {videos}, {std::move(generation_config)}, streamer)[0];
        auto stop_time = std::chrono::steady_clock::now();
        return build_decoded_results(result, start_time, stop_time, speaker);
    }

    void start_chat(const std::string& system_message) override { m_impl.start_chat(system_message); }

    void finish_chat() override { m_impl.finish_chat(); }

    Tokenizer get_tokenizer() const override { return m_impl.get_tokenizer(); }

    void set_chat_template(const std::string& new_template) override { OPENVINO_THROW("Chat mode is not supported."); }

    GenerationConfig get_generation_config() const override { return m_impl.get_config(); }

    void set_generation_config(const GenerationConfig& new_config) override { m_impl.set_config(new_config); }

private:
    void try_init_speech_pipeline(const std::filesystem::path& models_dir,
                                  const std::string& device,
                                  const ov::AnyMap& properties) {
        if (!std::filesystem::exists(models_dir / "config.json")) {
            return;
        }
        auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(models_dir, "config.json");
        if (vlm_config.model_type == VLMModelType::QWEN3_OMNI && vlm_config.enable_audio_output) {
            m_speech_pipeline = std::make_unique<Qwen3OmniSpeechPipeline>(models_dir, vlm_config, device, properties);
        }
    }

    /// @brief Build VLMDecodedResults from CB generate output, including perf metrics and speech generation.
    VLMDecodedResults build_decoded_results(
        const VLMDecodedResults& result,
        std::chrono::steady_clock::time_point start_time,
        std::chrono::steady_clock::time_point stop_time,
        const std::string& speaker) {
        VLMDecodedResults decoded;
        decoded.perf_metrics = result.perf_metrics;
        decoded.perf_metrics.load_time = get_load_time();
        decoded.perf_metrics.raw_metrics.generate_durations.clear();
        decoded.perf_metrics.raw_metrics.generate_durations.emplace_back(
            PerfMetrics::get_microsec(stop_time - start_time));
        decoded.perf_metrics.m_evaluated = false;
        decoded.perf_metrics.evaluate_statistics(start_time);

        for (size_t idx = 0; idx < result.texts.size(); ++idx) {
            decoded.texts.push_back(result.texts.at(idx));
            decoded.scores.push_back(result.scores.at(idx));
        }
        decoded.finish_reasons = result.finish_reasons;

        run_speech_if_needed(result, decoded, speaker, m_pending_audio_streamer, m_pending_audio_chunk_frames);
        return decoded;
    }

    /// @brief Flatten per-step hidden states into per-token hidden states.
    /// Each step tensor has shape [num_tokens, 1, hidden_size]. We split multi-token
    /// tensors (from prefill) into individual [1, 1, hidden_size] slices.
    static std::vector<ov::Tensor> flatten_hidden_states(const std::vector<ov::Tensor>& per_step_hs) {
        std::vector<ov::Tensor> per_token;
        size_t total_tokens = 0;
        for (const auto& step_hs : per_step_hs) {
            total_tokens += step_hs.get_shape()[0];
        }
        per_token.reserve(total_tokens);
        for (const auto& step_hs : per_step_hs) {
            const auto shape = step_hs.get_shape();
            const size_t num_tokens = shape[0];
            const size_t hidden_size = shape.back();
            const size_t elem_size = step_hs.get_element_type().size();
            const auto* src = static_cast<const uint8_t*>(step_hs.data());
            const size_t token_bytes = hidden_size * elem_size;
            const size_t stride = (shape.size() == 3)
                ? shape[1] * hidden_size * elem_size
                : token_bytes;
            for (size_t t = 0; t < num_tokens; t++) {
                ov::Tensor token_hs(step_hs.get_element_type(), {1, 1, hidden_size});
                std::memcpy(token_hs.data(), src + t * stride, token_bytes);
                per_token.push_back(std::move(token_hs));
            }
        }
        return per_token;
    }

    /// @brief Run speech generation if hidden states are available and speech pipeline is initialized.
    void run_speech_if_needed(const VLMDecodedResults& cb_result, VLMDecodedResults& output,
                              const std::string& speaker,
                              const AudioStreamerVariant& audio_streamer,
                              size_t chunk_frames) {
        if (!cb_result.m_hidden_states_data) {
            return;
        }
        if (!m_speech_pipeline || !m_speech_pipeline->is_available()) {
            GENAI_WARN("return_audio was requested but no speech pipeline is available; audio output will be empty");
            return;
        }

        const auto& hs_data = cb_result.m_hidden_states_data;
        if (hs_data->hidden_states.empty()) {
            return;
        }

        const auto& full_token_ids = hs_data->prompt_ids;

        // Flatten per-step hidden states into per-token hidden states
        auto all_hs = flatten_hidden_states(hs_data->hidden_states[0]);
        auto all_ihs = hs_data->intermediate_hidden_states.empty()
            ? std::vector<ov::Tensor>{}
            : flatten_hidden_states(hs_data->intermediate_hidden_states[0]);

        GENAI_DEBUG("Speech: tokens=%zu, hidden_states=%zu, intermediate=%zu",
                    full_token_ids.size(), all_hs.size(), all_ihs.size());

        auto waveform = m_speech_pipeline->generate_speech(
            full_token_ids, all_hs, all_ihs, audio_streamer, chunk_frames, speaker);

        if (waveform && waveform.get_size() > 0) {
            output.speech_outputs.push_back(std::move(waveform));
        }
    }
};
