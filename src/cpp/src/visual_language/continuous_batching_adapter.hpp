// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "utils.hpp"
#include "visual_language/chat_history_state.hpp"
#include "visual_language/pipeline_base.hpp"
#include "visual_language/vlm_config.hpp"

using namespace ov::genai;

class ov::genai::VLMPipeline::VLMContinuousBatchingAdapter : public ov::genai::VLMPipeline::VLMBackend {
public:
    ContinuousBatchingPipeline m_impl;

    VLMContinuousBatchingAdapter(const std::filesystem::path& models_dir,
                                 const SchedulerConfig& scheduler_config,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
        : m_impl{models_dir, scheduler_config, device, properties} {
        set_attention_backend(PA_BACKEND);
        load_vlm_config(models_dir);
    }

    VLMContinuousBatchingAdapter(const std::shared_ptr<ov::Model>& language_model,
                                 const std::filesystem::path& models_dir,
                                 const SchedulerConfig& scheduler_config,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
        : m_impl{language_model, models_dir, scheduler_config, device, properties} {
        set_attention_backend(PA_BACKEND);
        load_vlm_config(models_dir);
    }

    VLMContinuousBatchingAdapter(const ModelsMap& models_map,
                                 const Tokenizer& tokenizer,
                                 const std::filesystem::path& config_dir_path,
                                 const SchedulerConfig& scheduler_config,
                                 const std::string& device,
                                 const ov::AnyMap& properties,
                                 const ov::genai::GenerationConfig& generation_config)
        : m_impl{models_map, tokenizer, scheduler_config, device, config_dir_path, properties, generation_config} {
        set_attention_backend(PA_BACKEND);
        load_vlm_config(config_dir_path);
    }

    VLMContinuousBatchingAdapter(const std::shared_ptr<ov::Model>& language_model,
                                 const ModelsMap& models_map,
                                 const Tokenizer& tokenizer,
                                 const std::filesystem::path& config_dir_path,
                                 const SchedulerConfig& scheduler_config,
                                 const std::string& device,
                                 const ov::AnyMap& properties,
                                 const ov::genai::GenerationConfig& generation_config)
        : m_impl{language_model,
                 models_map,
                 tokenizer,
                 scheduler_config,
                 device,
                 config_dir_path,
                 properties,
                 generation_config} {
        set_attention_backend(PA_BACKEND);
        load_vlm_config(config_dir_path);
    }

    VLMDecodedResults generate(const std::string& prompt,
                               const std::vector<ov::Tensor>& images,
                               const GenerationConfig& generation_config,
                               const StreamerVariant& streamer) override {
        return generate(prompt, images, {}, generation_config, streamer);
    }

    VLMDecodedResults generate(const std::string& prompt,
                               const std::vector<ov::Tensor>& images,
                               const std::vector<ov::Tensor>& videos,
                               const GenerationConfig& generation_config,
                               const StreamerVariant& streamer) override {
        return generate(prompt, images, videos, {}, {}, generation_config, streamer);
    }

    VLMDecodedResults generate(const std::string& prompt,
                               const std::vector<ov::Tensor>& images,
                               const std::vector<ov::Tensor>& videos,
                               const std::vector<ov::Tensor>& audios,
                               const std::vector<VideoMetadata>& videos_metadata,
                               const GenerationConfig& generation_config,
                               const StreamerVariant& streamer) override {
        auto start_time = std::chrono::steady_clock::now();
        std::vector<ov::genai::GenerationConfig> generation_configs = {generation_config};
        const auto decoded_results = m_impl.generate({prompt},
                                                     ov::genai::images_batches({images}),
                                                     ov::genai::videos_batches({videos}),
                                                     ov::genai::videos_metadata_batches({videos_metadata}),
                                                     ov::genai::audios_batches({audios}),
                                                     ov::genai::generation_config_batches(generation_configs),
                                                     ov::genai::streamer(streamer))[0];
        auto stop_time = std::chrono::steady_clock::now();
        return finalize_decoded_results(decoded_results, start_time, stop_time);
    }

    VLMDecodedResults generate(const ChatHistory& history,
                               const std::vector<ov::Tensor>& images,
                               const GenerationConfig& generation_config,
                               const StreamerVariant& streamer) override {
        return generate(history, images, {}, generation_config, streamer);
    }

    VLMDecodedResults generate(const ChatHistory& history,
                               const std::vector<ov::Tensor>& images,
                               const std::vector<ov::Tensor>& videos,
                               const GenerationConfig& generation_config,
                               const StreamerVariant& streamer) override {
        return generate(history, images, videos, {}, {}, generation_config, streamer);
    }

    VLMDecodedResults generate(const ChatHistory& history,
                               const std::vector<ov::Tensor>& images,
                               const std::vector<ov::Tensor>& videos,
                               const std::vector<ov::Tensor>& audios,
                               const std::vector<VideoMetadata>& videos_metadata,
                               const GenerationConfig& generation_config,
                               const StreamerVariant& streamer) override {
        auto start_time = std::chrono::steady_clock::now();
        ChatHistoryInternalState::get_or_create(history);
        std::vector<ov::genai::GenerationConfig> generation_configs = {generation_config};
        const auto decoded_results = m_impl.generate({history},
                                                     ov::genai::images_batches({images}),
                                                     ov::genai::videos_batches({videos}),
                                                     ov::genai::videos_metadata_batches({videos_metadata}),
                                                     ov::genai::audios_batches({audios}),
                                                     ov::genai::generation_config_batches(generation_configs),
                                                     ov::genai::streamer(streamer))[0];
        auto stop_time = std::chrono::steady_clock::now();
        return finalize_decoded_results(decoded_results, start_time, stop_time);
    }

    void start_chat(const std::string& system_message) override {
        m_impl.start_chat(system_message);
    }

    void finish_chat() override {
        m_impl.finish_chat();
    }

    Tokenizer get_tokenizer() const override {
        return m_impl.get_tokenizer();
    }

    void set_chat_template(const std::string& new_template) override {
        OPENVINO_THROW("Chat mode is not supported.");
    }

    GenerationConfig get_generation_config() const override {
        return m_impl.get_config();
    }

    void set_generation_config(const GenerationConfig& new_config) override {
        m_impl.set_config(new_config);
    }

    bool supports_hidden_states_collection() const override {
        return true;
    }

    bool is_audio_output_enabled() const override {
        return m_vlm_config.enable_audio_output;
    }

private:
    /// VLM config loaded once at construction so model-type/audio-output queries are cheap.
    /// Empty/default when no config.json is present (e.g., test scaffolding paths).
    VLMConfig m_vlm_config;

    void load_vlm_config(const std::filesystem::path& models_dir) {
        if (!std::filesystem::exists(models_dir / "config.json")) {
            return;
        }
        m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(models_dir, "config.json");
    }

    /// @brief Build VLMDecodedResults from CB generate output, attaching pipeline-level perf metrics.
    /// Speech generation is intentionally NOT performed here; the speech pipeline lives in
    /// `OmniPipelineImpl`.
    VLMDecodedResults finalize_decoded_results(const VLMDecodedResults& result,
                                               std::chrono::steady_clock::time_point start_time,
                                               std::chrono::steady_clock::time_point stop_time) {
        VLMDecodedResults decoded;
        decoded.perf_metrics = result.perf_metrics;
        decoded.perf_metrics.load_time = get_load_time();
        decoded.perf_metrics.raw_metrics.generate_durations.clear();
        decoded.perf_metrics.raw_metrics.generate_durations.emplace_back(
            PerfMetrics::get_microsec(stop_time - start_time));
        decoded.perf_metrics.m_evaluated = false;
        decoded.perf_metrics.evaluate_statistics(start_time);
        decoded.extended_perf_metrics = result.extended_perf_metrics;

        for (size_t idx = 0; idx < result.texts.size(); ++idx) {
            decoded.texts.push_back(result.texts.at(idx));
            decoded.scores.push_back(result.scores.at(idx));
        }
        decoded.finish_reasons = result.finish_reasons;
        // Forward hidden-states fields so OmniPipelineImpl / TalkerBase can drive speech
        // generation. Inner ov::Tensor copies are ref-counted handles, so the outer-vector
        // copies are O(n) in step count, not in tensor bytes.
        decoded.intermediate_hidden_states = result.intermediate_hidden_states;
        decoded.full_token_ids = result.full_token_ids;
        return decoded;
    }
};
