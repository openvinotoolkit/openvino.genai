// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "automatic_speech_recognition/streaming_session_impl_base.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

namespace ov::genai {

class WhisperASRPipelineAdapter;

class WhisperASRStreamingSessionImpl final : public ASRStreamingSession::Impl {
public:
    WhisperASRStreamingSessionImpl(WhisperASRPipelineAdapter* pipeline,
                                   const ASRStreamingConfig& streaming_config,
                                   const ASRGenerationConfig& generation_config);

    std::optional<ASRPartialResult> push_chunk(const std::vector<float>& pcm16k) override;
    ASRPartialResult finish() override;

private:
    void decode_current_accum();
    std::string compute_committed_text() const;

    WhisperASRPipelineAdapter* m_pipeline;  // non-owning; lifetime guaranteed by ASRPipeline
    ASRStreamingConfig m_streaming_config;
    ASRGenerationConfig m_generation_config;

    std::vector<float> m_buffer;
    std::vector<float> m_audio_accum;
    size_t m_chunk_count = 0;
    size_t m_chunk_size_samples;
    bool m_window_full = false;  // set when audio_accum reaches the Whisper 30-second limit

    std::string m_current_language;
    std::string m_current_text;
    std::string m_current_committed_text;
    std::string m_current_new_committed_text;
    std::string m_current_partial_text;

    ASRPerfMetrics m_perf_metrics;
};

}  // namespace ov::genai
