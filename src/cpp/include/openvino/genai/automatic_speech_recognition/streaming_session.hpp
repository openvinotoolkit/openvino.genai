// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"
#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

class ASRPipeline;
class ASRStreamingSessionImpl;

/// Parameters of an ASRStreamingSession. Invalid values throw in the ASRStreamingSession constructor.
struct OPENVINO_GENAI_EXPORTS ASRStreamingConfig {
    /// Audio duration between two decode passes. Must be > 0.
    float chunk_size_sec = 2.0f;
    /// Re-decode models: tokens at the tail of each pass treated as unstable and excluded from committed text.
    size_t unfixed_token_num = 5;
    /// Re-decode models: chunks of already decoded audio kept in the window for re-decoding.
    size_t unfixed_chunk_num = 2;
    /// Re-decode models: audio window bound in chunks. 0 = unbounded. Must be 0 or greater than unfixed_chunk_num.
    size_t window_chunk_num = 8;
};

/// Result of one decode pass of an ASRStreamingSession.
struct OPENVINO_GENAI_EXPORTS ASRStreamingResult {
    std::string language;
    /// All stable text so far. Append-only across results.
    std::string committed_text;
    /// Stable text added by this result.
    std::string new_committed_text;
    /// Unstable tail that may change in the next result. Always empty for native streaming models.
    std::string partial_text;
    /// True for the last result after end(). partial_text is empty for the final result.
    bool is_final = false;
    /// Metrics of this decode pass.
    ASRPerfMetrics perf_metrics;
};

/**
 * @brief Streaming transcription of one audio stream.
 *
 * push() only buffers audio. read() returns the next result and runs the decode pass that produces it on the calling
 * thread. end() closes input; subsequent read() calls drain the remaining audio and the last result has
 * is_final == true.
 *
 * Threading: push() and end() may be called from one producer thread while read() is called from one consumer thread.
 * Sessions of one pipeline may be used from different threads. The constructor throws if the pipeline cannot serve
 * another concurrent session. The session co-owns the pipeline implementation and may outlive the ASRPipeline.
 */
class OPENVINO_GENAI_EXPORTS ASRStreamingSession {
public:
    ASRStreamingSession(ASRPipeline& pipeline,
                        const ASRStreamingConfig& streaming_config = {},
                        const std::optional<ASRGenerationConfig>& generation_config = std::nullopt);

    ~ASRStreamingSession();
    /// Move-only: a session is one live audio stream, so it can't be copied. Moving lets a session be returned from a
    /// function or kept in a container, for example one per server connection. The session slot moves with it; a
    /// moved-from session throws on use. Don't move a session while another thread is inside one of its calls.
    ASRStreamingSession(ASRStreamingSession&&) noexcept;
    ASRStreamingSession& operator=(ASRStreamingSession&&) noexcept;

    /// Append mono 16 kHz float32 PCM. Never runs inference. Throws after end().
    void push(const std::vector<float>& pcm16k);

    /// Close input. May be called once.
    void end();

    /// Return the next result, or std::nullopt if no decode pass is due (not enough buffered audio and end() not
    /// called) or the final result was already returned. Blocks while the decode pass runs.
    std::optional<ASRStreamingResult> read();

private:
    std::unique_ptr<ASRStreamingSessionImpl> m_impl;
};

}  // namespace ov::genai
