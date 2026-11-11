// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "continuous_batching/pipeline_impl.hpp"
#include "omni/speech_streamer_utils.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "utils.hpp"
#include "visual_language/chat_history_state.hpp"
#include "visual_language/lfm3_omni/speech_step_hook.hpp"
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
        install_speech_hook([models_dir, device, properties] {
            return std::make_unique<LFM3SpeechModels>(models_dir, device, properties);
        });
    }

    VLMContinuousBatchingAdapter(const std::shared_ptr<ov::Model>& language_model,
                                 const std::filesystem::path& models_dir,
                                 const SchedulerConfig& scheduler_config,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
        : m_impl{language_model, models_dir, scheduler_config, device, properties} {
        set_attention_backend(PA_BACKEND);
        load_vlm_config(models_dir);
        install_speech_hook([models_dir, device, properties] {
            return std::make_unique<LFM3SpeechModels>(models_dir, device, properties);
        });
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
        const ModelsMap speech_models =
            utils::select_models(models_map, {"depth_decoder", "audio_out_embeddings", "audio_decoder"});
        install_speech_hook([speech_models, config_dir_path, device, properties] {
            return std::make_unique<LFM3SpeechModels>(speech_models, config_dir_path, device, properties);
        });
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
        const ModelsMap speech_models =
            utils::select_models(models_map, {"depth_decoder", "audio_out_embeddings", "audio_decoder"});
        install_speech_hook([speech_models, config_dir_path, device, properties] {
            return std::make_unique<LFM3SpeechModels>(speech_models, config_dir_path, device, properties);
        });
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
        const auto run = [&](const GenerationConfig& config, const StreamerVariant& text_streamer) {
            return m_impl.generate({prompt},
                                   ov::genai::images_batches({images}),
                                   ov::genai::videos_batches({videos}),
                                   ov::genai::videos_metadata_batches({videos_metadata}),
                                   ov::genai::audios_batches({audios}),
                                   ov::genai::generation_config_batches(std::vector<GenerationConfig>{config}),
                                   ov::genai::streamer(text_streamer))[0];
        };
        if (m_pending_speech_config) {
            return generate_with_speech(generation_config, streamer, start_time, run);
        }
        const auto decoded_results = run(generation_config, streamer);
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
        const auto run = [&](const GenerationConfig& config, const StreamerVariant& text_streamer) {
            return m_impl.generate({history},
                                   ov::genai::images_batches({images}),
                                   ov::genai::videos_batches({videos}),
                                   ov::genai::videos_metadata_batches({videos_metadata}),
                                   ov::genai::audios_batches({audios}),
                                   ov::genai::generation_config_batches(std::vector<GenerationConfig>{config}),
                                   ov::genai::streamer(text_streamer))[0];
        };
        if (m_pending_speech_config) {
            return generate_with_speech(generation_config, streamer, start_time, run);
        }
        const auto decoded_results = run(generation_config, streamer);
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

    void check_speech_in_stream(const OmniTalkerSpeechConfig& speech_config) const override {
        if (!m_speech_hook) {
            VLMBackend::check_speech_in_stream(speech_config);
        }
        LFM3SpeechRequest::validate(speech_config);
    }

    bool is_audio_output_enabled() const override {
        return m_vlm_config.enable_audio_output;
    }

private:
    /// VLM config loaded once at construction so model-type/audio-output queries are cheap.
    /// Empty/default when no config.json is present (e.g., test scaffolding paths).
    VLMConfig m_vlm_config;

    /// LFM3-Omni speech output; null for other models and for pipelines without a plain continuous batching core.
    std::shared_ptr<LFM3SpeechStepHook> m_speech_hook;

    void install_speech_hook(std::function<std::unique_ptr<LFM3SpeechModels>()> make_models) {
        if (m_vlm_config.model_type != VLMModelType::LFM3_OMNI) {
            return;
        }
        // Speculative and prompt lookup pipelines wrap other cores; speech stays unavailable there.
        const auto impl = std::dynamic_pointer_cast<ContinuousBatchingPipeline::ContinuousBatchingImpl>(m_impl.m_impl);
        if (!impl) {
            return;
        }
        m_speech_hook = std::make_shared<LFM3SpeechStepHook>(std::move(make_models));
        impl->set_step_hook(m_speech_hook);
    }

    /// @brief LFM3-Omni speech on continuous batching: the step hook picks every slot, and the pipeline only
    /// records them, greedily and with the speech-aware stop tokens.
    template <typename Generate>
    VLMDecodedResults generate_with_speech(const GenerationConfig& generation_config,
                                           const StreamerVariant& streamer,
                                           std::chrono::steady_clock::time_point start_time,
                                           const Generate& run) {
        GenerationConfig text_config = generation_config;
        const GenerationConfig defaults = m_impl.get_config();
        if (text_config.eos_token_id == -1) {
            text_config.set_eos_token_id(defaults.eos_token_id);
        }
        if (text_config.stop_token_ids.empty()) {
            text_config.stop_token_ids = defaults.stop_token_ids;
        }
        // The pipeline only sees the greedy copy below, so the caller's settings are checked here, as on SDPA.
        text_config.validate();
        const LFM3SpeechRequest request(*m_pending_speech_config, text_config);

        struct ExpectedRequestGuard {
            LFM3SpeechStepHook& hook;
            ~ExpectedRequestGuard() {
                hook.forget_expected_request();
            }
        } expected_request_guard{*m_speech_hook};
        const auto text_streamer = utils::create_streamer(streamer, m_impl.get_tokenizer());
        m_speech_hook->expect_request(request, m_pending_speech_streamer, text_streamer);
        const LFM3SpeechModelConfig& speech_config = m_speech_hook->config();

        GenerationConfig pipeline_config = text_config;
        pipeline_config.do_sample = false;
        pipeline_config.repetition_penalty = 1.0f;
        // Inside an interleaved answer <|text_end|> starts the speech tail, so only the hook may end the turn on it.
        pipeline_config.stop_token_ids.erase(speech_config.text_end_token_id);
        OPENVINO_ASSERT(pipeline_config.eos_token_id != speech_config.text_end_token_id,
                        "LFM3-Omni speech output needs an eos_token_id other than <|text_end|>");

        // The hook writes the text streamer itself, so the pipeline streams nothing.
        const VLMDecodedResults decoded_results = run(pipeline_config, std::monostate{});
        m_speech_hook->rethrow_streamer_error();
        if (text_streamer) {
            text_streamer->end();
        }
        LFM3SpeechResult speech = m_speech_hook->take_result();
        // Ended here so an end() failure reaches the caller; SpeechPropertiesGuard swallows it while unwinding.
        const auto speech_streamer = std::exchange(m_pending_speech_streamer, std::monostate{});
        if (is_speech_streamer_active(speech_streamer)) {
            end_speech_streamer(speech_streamer);
        }
        const auto stop_time = std::chrono::steady_clock::now();
        VLMDecodedResults decoded = finalize_decoded_results(decoded_results, start_time, stop_time);
        // Speech and text come from one decode loop, so the speech time is the whole call.
        decoded.speech_result.perf_metrics.generation_time_ms =
            PerfMetrics::get_microsec(stop_time - start_time) / 1000.0f;
        if (speech.waveform) {
            decoded.speech_result.perf_metrics.num_generated_samples = speech.waveform.get_size();
            decoded.speech_result.waveforms.push_back(std::move(speech.waveform));
        }
        return decoded;
    }

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
