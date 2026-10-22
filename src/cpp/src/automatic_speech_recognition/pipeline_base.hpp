// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "utils.hpp"

namespace ov::genai {

void erase_allowed_asr_ctor_properties(ov::AnyMap& properties);

/// Owns the input queue and generation state of ASRStream; derived classes implement model inference.
class ASRStreamImplBase {
public:
    explicit ASRStreamImplBase(size_t max_queued_frames);
    virtual ~ASRStreamImplBase() = default;

    void push_audio(const std::vector<float>& audio_frame);
    void end_audio();
    ASRDecodedResults generate(const ASRTextCallback& text_callback);

protected:
    /// Run inference on the next audio frame and return newly decoded text.
    virtual std::string process_audio(const std::vector<float>& audio_frame) = 0;
    /// Run inference on audio buffered after the last frame and return the remaining text.
    virtual std::string flush() = 0;

private:
    enum class GenerationState { NOT_STARTED, RUNNING, STOPPED };

    /// Return std::nullopt when the end of audio is reached.
    std::optional<std::vector<float>> pop_audio_frame();
    void stop_generation();

    const size_t m_max_queued_frames;
    std::mutex m_mutex;
    std::condition_variable m_frame_pushed;
    std::condition_variable m_frame_popped;
    std::deque<std::vector<float>> m_queue;
    bool m_audio_ended = false;
    GenerationState m_state = GenerationState::NOT_STARTED;
};

class ASRPipelineImplBase {
public:
    ASRGenerationConfig m_generation_config;
    Tokenizer m_tokenizer;
    float m_load_time_ms = 0.0f;

    ASRPipelineImplBase(const std::filesystem::path& models_path)
        : m_generation_config(utils::from_config_json_if_exists<ASRGenerationConfig>(models_path)),
          m_tokenizer{models_path} {}

    virtual ASRDecodedResults generate(const AudioInputs& audio_inputs,
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
};

}  // namespace ov::genai
