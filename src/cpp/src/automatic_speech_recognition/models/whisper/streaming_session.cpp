// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "streaming_session.hpp"

#include <algorithm>

#include "automatic_speech_recognition/debug_dump.hpp"
#include "automatic_speech_recognition/sliding_window.hpp"
#include "pipeline.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

namespace ov::genai {

namespace {

// Whisper encoder accepts at most 30 seconds of 16 kHz audio.
static constexpr size_t WHISPER_MAX_SAMPLES = 480000;

// UTF-8 encoding of U+FFFD REPLACEMENT CHARACTER — signals a corrupted decode boundary.
static constexpr const char* REPLACEMENT_CHAR_UTF8 = "\xef\xbf\xbd";

}  // namespace

WhisperASRStreamingSessionImpl::WhisperASRStreamingSessionImpl(WhisperASRPipelineAdapter* pipeline,
                                                               const ASRStreamingConfig& streaming_config,
                                                               const ASRGenerationConfig& generation_config)
    : m_pipeline{pipeline},
      m_streaming_config{streaming_config},
      m_generation_config{generation_config},
      m_chunk_size_samples{static_cast<size_t>(
          std::max(1.0f, streaming_config.chunk_size_sec) * 16000.0f)} {
    OPENVINO_ASSERT(m_pipeline != nullptr, "WhisperASRStreamingSessionImpl: pipeline pointer must not be null");
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

// Returns the stable committed text by tokenizing the current result and dropping the last
// context_rollback_tokens tokens. Increases rollback on UTF-8 boundary errors (same policy as Qwen3-ASR).
std::string WhisperASRStreamingSessionImpl::compute_committed_text() const {
    if (m_current_text.empty() || m_chunk_count < m_streaming_config.warmup_chunks) {
        return "";
    }

    const TokenizedInputs encoded = m_pipeline->m_tokenizer.encode(m_current_text);
    const ov::Tensor& ids_tensor = encoded.input_ids;
    const size_t n_tokens = ids_tensor.get_shape()[1];

    size_t rollback = m_streaming_config.context_rollback_tokens;

    while (true) {
        const size_t keep = (n_tokens > rollback) ? n_tokens - rollback : 0;
        if (keep == 0) {
            return "";
        }
        const int64_t* data = ids_tensor.data<const int64_t>();
        const std::vector<int64_t> kept_ids(data, data + keep);
        const std::string committed = m_pipeline->m_tokenizer.decode(kept_ids);
        if (committed.find(REPLACEMENT_CHAR_UTF8) == std::string::npos) {
            return committed;
        }
        ++rollback;
        if (rollback >= n_tokens) {
            return "";
        }
    }
}

void WhisperASRStreamingSessionImpl::decode_current_accum() {
    // Inject committed_text as the decoder prefix so the model is anchored to stable prior output.
    // With no prior committed text (warmup or rollback=0) the prefix is left unset.
    const bool inject_prefix = m_chunk_count >= m_streaming_config.warmup_chunks &&
                               !m_current_committed_text.empty();
    ASRGenerationConfig config_for_pass = m_generation_config;
    if (inject_prefix) {
        config_for_pass.prefix = m_current_committed_text;
    }

    if (asr_debug_dump_enabled()) {
        asr_debug_dump_chunk("whisper",
                             m_chunk_count,
                             m_audio_accum,
                             /*sample_rate=*/16000,
                             inject_prefix ? m_current_committed_text : std::string());
    }

    const ASRDecodedResults results = m_pipeline->generate(m_audio_accum, config_for_pass, nullptr);
    OPENVINO_ASSERT(!results.texts.empty(), "WhisperASRStreamingSessionImpl: generate returned empty results");

    m_current_language = results.languages.empty() ? "" : results.languages[0];

    // When a prefix was injected the decoder only generates the continuation; reconstruct the full
    // transcript by prepending the committed prefix so that compute_committed_text() can operate on it.
    m_current_text = (inject_prefix ? m_current_committed_text : "") + results.texts[0];
    ++m_chunk_count;

    const std::string prev_committed = m_current_committed_text;
    std::string candidate_committed = compute_committed_text();

    // committed_text is documented to grow monotonically. A decode pass can occasionally produce
    // a shorter (or empty) continuation than the audio/text context implies -- e.g. once the
    // ever-growing text prefix outpaces what a tight sliding audio window still supports -- so
    // never let it retreat below what was already committed.
    if (candidate_committed.size() >= prev_committed.size()) {
        m_current_committed_text = std::move(candidate_committed);
    }
    m_current_new_committed_text = m_current_committed_text.substr(prev_committed.size());
    m_current_partial_text = (m_current_text.size() >= m_current_committed_text.size())
                                 ? m_current_text.substr(m_current_committed_text.size())
                                 : m_current_text;
}

std::optional<ASRPartialResult> WhisperASRStreamingSessionImpl::push_chunk(const std::vector<float>& pcm16k) {
    if (m_window_full) {
        // The Whisper 30-second window is exhausted; callers should invoke finish().
        return std::nullopt;
    }

    m_buffer.insert(m_buffer.end(), pcm16k.begin(), pcm16k.end());

    if (m_buffer.size() < m_chunk_size_samples) {
        return std::nullopt;
    }

    // m_audio_accum has been fully decoded by the end of every prior push_chunk()/finish()
    // call, so its size right before this drain is exactly how much is safe to drop from.
    const size_t already_inferred_samples = m_audio_accum.size();

    // Drain the buffer, capping at the Whisper maximum window.
    const size_t remaining_capacity = WHISPER_MAX_SAMPLES - m_audio_accum.size();
    const size_t drain = std::min(m_buffer.size(), remaining_capacity);
    m_audio_accum.insert(m_audio_accum.end(), m_buffer.begin(), m_buffer.begin() + drain);
    m_buffer.erase(m_buffer.begin(), m_buffer.begin() + drain);

    // Drop already-decoded audio outside the sliding window now that the drain is done. This
    // frees capacity against WHISPER_MAX_SAMPLES for the *next* call, so a window configured
    // to stay under 30s means m_window_full is rarely (if ever) reached and streaming is
    // effectively unbounded, not just single-utterance.
    apply_sliding_window_drop(m_audio_accum,
                              already_inferred_samples,
                              m_chunk_size_samples,
                              m_streaming_config.window_chunk_num,
                              m_streaming_config.window_rollback_chunk_num);

    if (m_audio_accum.size() >= WHISPER_MAX_SAMPLES) {
        m_window_full = true;
        m_buffer.clear();  // discard audio that would exceed the window
    }

    decode_current_accum();
    return ASRPartialResult{m_current_language, m_current_committed_text,
                            m_current_new_committed_text, m_current_partial_text};
}

ASRPartialResult WhisperASRStreamingSessionImpl::finish() {
    m_current_new_committed_text = "";

    if (!m_buffer.empty() && !m_window_full) {
        const size_t already_inferred_samples = m_audio_accum.size();

        const size_t remaining_capacity = WHISPER_MAX_SAMPLES - m_audio_accum.size();
        const size_t drain = std::min(m_buffer.size(), remaining_capacity);
        m_audio_accum.insert(m_audio_accum.end(), m_buffer.begin(), m_buffer.begin() + drain);
        m_buffer.clear();

        apply_sliding_window_drop(m_audio_accum,
                                  already_inferred_samples,
                                  m_chunk_size_samples,
                                  m_streaming_config.window_chunk_num,
                                  m_streaming_config.window_rollback_chunk_num);

        if (!m_audio_accum.empty()) {
            decode_current_accum();
        }
    }

    // Commit any remaining partial tail; final result always has partial_text == "".
    m_current_committed_text += m_current_partial_text;
    m_current_new_committed_text += std::move(m_current_partial_text);
    m_current_partial_text = "";

    return {m_current_language, m_current_committed_text,
            m_current_new_committed_text, m_current_partial_text};
}

}  // namespace ov::genai
