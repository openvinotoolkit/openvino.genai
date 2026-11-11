// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "utils.hpp"

namespace ov::genai {

void erase_allowed_asr_ctor_properties(ov::AnyMap& properties);

/// Owns the input queue and generation state of ASRStream; derived classes implement model inference.
class ASRStreamImplBase {
public:
    explicit ASRStreamImplBase(size_t max_queued_frames, float load_time_ms = 0.0f);
    virtual ~ASRStreamImplBase() = default;

    void push_audio(const std::vector<float>& audio_frame);
    void end_audio();
    size_t get_num_generated_tokens() const;
    virtual size_t get_kv_cache_size() const {
        OPENVINO_THROW("KV-cache size monitoring is not supported by this ASR model.");
    }
    virtual std::map<std::string, std::pair<size_t, size_t>> get_kv_cache_window_sizes() const {
        OPENVINO_THROW("KV-cache window monitoring is not supported by this ASR model.");
    }
    virtual std::map<std::string, double> get_stage_durations() const {
        OPENVINO_THROW("Stage timing is not supported by this ASR model.");
    }
    ASRDecodedResults generate(const ASRTextCallback& text_callback);

protected:
    /// Run inference on the next audio frame and return newly decoded text.
    virtual std::string process_audio(const std::vector<float>& audio_frame) = 0;
    /// Run inference on audio buffered after the last frame and return the remaining text.
    virtual std::string flush() = 0;
    void stop_generation();

    ASRPerfMetrics m_perf_metrics;

private:
    enum class GenerationState { NOT_STARTED, RUNNING, STOPPED };

    /// Return std::nullopt when the end of audio is reached.
    std::optional<std::vector<float>> pop_audio_frame();

    const size_t m_max_queued_frames;
    const float m_load_time_ms;
    std::atomic<size_t> m_num_generated_tokens{0};
    std::mutex m_mutex;
    std::condition_variable m_frame_pushed;
    std::condition_variable m_frame_popped;
    std::deque<std::vector<float>> m_queue;
    bool m_audio_ended = false;
    GenerationState m_state = GenerationState::NOT_STARTED;
};

using AudiosView = std::vector<std::reference_wrapper<const std::vector<float>>>;

class ASRPipelineImplBase {
public:
    ASRGenerationConfig m_generation_config;
    Tokenizer m_tokenizer;
    float m_load_time_ms = 0.0f;

    ASRPipelineImplBase(const std::filesystem::path& models_path)
        : ASRPipelineImplBase(models_path, Tokenizer{models_path}) {}

    ASRPipelineImplBase(const std::filesystem::path& models_path, const Tokenizer& tokenizer)
        : m_generation_config(utils::from_config_json_if_exists<ASRGenerationConfig>(models_path)),
          m_tokenizer{tokenizer} {}

    virtual ASRDecodedResults generate(const AudiosView& audios,
                                       const std::optional<ASRGenerationConfig>& generation_config,
                                       const std::shared_ptr<StreamerBase> streamer) = 0;

    virtual std::unique_ptr<ASRStreamImplBase> create_stream(
        const std::optional<ASRGenerationConfig>& generation_config) {
        OPENVINO_THROW("Streaming input is not supported by the selected ASR model.");
    }

    virtual void set_generation_config(const ASRGenerationConfig& config) {
        m_generation_config = config;
        m_generation_config.validate();
    }

    virtual ~ASRPipelineImplBase() = default;

protected:
    static const std::vector<float>& validate_single_audio_and_get(const AudiosView& audios) {
        OPENVINO_ASSERT(audios.size() == 1,
                        "This ASR model supports only one audio per generate, but received ",
                        audios.size(),
                        ".");
        return audios.front();
    }
};

}  // namespace ov::genai
