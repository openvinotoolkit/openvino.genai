// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deque>
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
    // Trims context_rollback_tokens off the tail of an arbitrary raw string (retrying with a
    // larger rollback on a UTF-8 boundary corruption). Operates on a plain parameter rather than
    // a persistent member so callers can apply it to any candidate string without mutating state.
    std::string trim_rollback(const std::string& raw) const;
    // Tag-free concatenation of m_commit_history's deltas, in order -- this session's decoder
    // prefix is this text verbatim (Whisper needs no language-tag wrapping the way Qwen3-ASR's
    // text-prompt-based prefix does).
    std::string history_text() const;

    WhisperASRPipelineAdapter* m_pipeline;  // non-owning; lifetime guaranteed by ASRPipeline
    ASRStreamingConfig m_streaming_config;
    ASRGenerationConfig m_generation_config;

    std::vector<float> m_buffer;
    std::vector<float> m_audio_accum;

    // One entry per decode pass that advanced committed_text, holding just that pass's own
    // (rollback-trimmed) contribution. history_text() rebuilds the decoder's prefix by
    // concatenating these fresh every pass -- never a persistent chain that could silently erode.
    // Entries are evicted in decode_current_accum() once m_total_dropped_samples has advanced past
    // the absolute sample range that grounded that chunk's own audio, not a static chunk-count margin.
    struct CommitRecord {
        size_t chunk_index;
        std::string text_delta;
    };
    std::deque<CommitRecord> m_commit_history;

    size_t m_chunk_count = 0;
    size_t m_chunk_size_samples;
    // Cumulative samples ever dropped from the front of m_audio_accum by apply_sliding_window_drop().
    // Since m_audio_accum is always a contiguous suffix of the full session audio, this is exactly
    // the absolute sample offset of m_audio_accum's current front within that full stream.
    size_t m_total_dropped_samples = 0;
    bool m_window_full = false;  // set when audio_accum reaches the Whisper 30-second limit

    std::string m_current_language;
    std::string m_current_text;
    std::string m_current_committed_text;
    std::string m_current_new_committed_text;
    std::string m_current_partial_text;

    ASRPerfMetrics m_perf_metrics;
};

}  // namespace ov::genai
