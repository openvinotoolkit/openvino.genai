// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deque>
#include <string>
#include <vector>

#include "automatic_speech_recognition/streaming_session_impl_base.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

namespace ov::genai {

class Qwen3ASR;

class Qwen3ASRStreamingSessionImpl final : public ASRStreamingSession::Impl {
public:
    Qwen3ASRStreamingSessionImpl(Qwen3ASR* pipeline,
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
    // Rebuilds the decoder-facing prefix fresh from m_commit_history every pass -- see the comment
    // on m_commit_history for why this is a pure function of bounded history rather than a chain.
    std::string bounded_prefix() const;
    // Tag-free concatenation of m_commit_history's deltas -- the same text bounded_prefix() wraps
    // with a language tag, exposed separately so decode_current_accum() can use its length as the
    // split point for this pass's own delta (see the delta-computation comment there).
    std::string history_text() const;

    Qwen3ASR* m_pipeline;  // non-owning; lifetime guaranteed by ASRPipeline
    ASRStreamingConfig m_streaming_config;
    ASRGenerationConfig m_generation_config;

    std::vector<float> m_buffer;
    std::vector<float> m_audio_accum;

    // One entry per decode pass that advanced committed_text, holding just that pass's own
    // (parsed, tag-free) contribution. bounded_prefix() rebuilds the decoder's prefix by
    // concatenating these fresh every pass -- never a persistent chain that could silently erode.
    // Entries are evicted in decode_current_accum() once m_total_dropped_samples has advanced past
    // the absolute sample range that grounded that chunk's own audio -- i.e. tied to actual sliding
    // window rolls (mirroring SGLang's start_new_window() reset), not a static chunk-count margin.
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

    std::string m_current_language;
    std::string m_current_text;
    std::string m_current_committed_text;
    std::string m_current_new_committed_text;
    std::string m_current_partial_text;

    ASRPerfMetrics m_perf_metrics;
};

}  // namespace ov::genai
