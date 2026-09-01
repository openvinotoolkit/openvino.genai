// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "streaming_session.hpp"

#include <algorithm>
#include <chrono>
#include <iterator>

#include "automatic_speech_recognition/debug_dump.hpp"
#include "automatic_speech_recognition/sliding_window.hpp"
#include "pipeline.hpp"

namespace ov::genai {

namespace {

// UTF-8 encoding of U+FFFD REPLACEMENT CHARACTER — signals a corrupted decode boundary.
static constexpr const char* REPLACEMENT_CHAR_UTF8 = "\xef\xbf\xbd";

}  // namespace

Qwen3ASRStreamingSessionImpl::Qwen3ASRStreamingSessionImpl(Qwen3ASR* pipeline,
                                                           const ASRStreamingConfig& streaming_config,
                                                           const ASRGenerationConfig& generation_config)
    : m_pipeline{pipeline},
      m_streaming_config{streaming_config},
      m_generation_config{generation_config},
      m_chunk_size_samples{static_cast<size_t>(
          std::max(1.0f, streaming_config.chunk_size_sec) * m_pipeline->m_feature_extractor.sampling_rate)} {
    OPENVINO_ASSERT(m_pipeline != nullptr, "Qwen3ASRStreamingSessionImpl: pipeline pointer must not be null");
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

std::string Qwen3ASRStreamingSessionImpl::trim_rollback(const std::string& raw) const {
    if (m_chunk_count < m_streaming_config.warmup_chunks || raw.empty()) {
        return "";
    }

    const TokenizedInputs encoded = m_pipeline->m_tokenizer.encode(raw);
    const ov::Tensor& ids_tensor = encoded.input_ids;
    const size_t n_tokens = ids_tensor.get_shape()[1];

    size_t rollback = m_streaming_config.context_rollback_tokens;

    // Increase rollback until the decoded prefix is free of replacement characters.
    while (true) {
        const size_t keep = (n_tokens > rollback) ? n_tokens - rollback : 0;
        if (keep == 0) {
            return "";
        }
        const int64_t* data = ids_tensor.data<const int64_t>();
        const std::vector<int64_t> kept_ids(data, data + keep);
        const std::string trimmed = m_pipeline->m_tokenizer.decode(kept_ids);
        if (trimmed.find(REPLACEMENT_CHAR_UTF8) == std::string::npos) {
            return trimmed;
        }
        ++rollback;
        if (rollback >= n_tokens) {
            return "";
        }
    }
}

std::string Qwen3ASRStreamingSessionImpl::history_text() const {
    std::string text;
    for (const auto& rec : m_commit_history) {
        text += rec.text_delta;
    }
    return text;
}

std::string Qwen3ASRStreamingSessionImpl::bounded_prefix() const {
    const std::string text = history_text();
    if (text.empty()) {
        return "";
    }

    const bool language_forced =
        m_generation_config.language.has_value() && !m_generation_config.language.value().empty();
    if (!language_forced && !m_current_language.empty()) {
        // Auto-detect mode: build_text_prompt() has no other source for the leading language tag
        // (it only injects one itself when the language is forced), so reconstruct the same tag
        // the model generated on its own first pass.
        return "language " + m_current_language + "<asr_text>" + text;
    }
    return text;
}

void Qwen3ASRStreamingSessionImpl::decode_current_accum() {
    // Evict commit_history entries whose grounding audio no longer exists in m_audio_accum, tied
    // directly to how far the sliding window has actually rolled (m_total_dropped_samples) rather
    // than a fixed chunk-count margin -- mirroring SGLang's start_new_window() reset (which discards
    // everything not yet safely emitted whenever the window rolls), but at the finer per-chunk
    // granularity our commit_history already tracks instead of a single blob reset.
    //
    // Chunk `i`'s own audio spans absolute samples [i * chunk_size, (i+1) * chunk_size) in the full
    // session stream (each successful decode pass consumes exactly one new chunk in steady state).
    // Once m_total_dropped_samples has advanced past that chunk's end, its audio is entirely gone
    // from the window, and the model can no longer be expected to reconcile a prefix built from it.
    //
    // window_chunk_num == 0 means the audio window is itself unbounded, so nothing is ever dropped
    // (m_total_dropped_samples stays 0) and nothing is evicted here either -- matches the audio
    // window's own semantics.
    while (!m_commit_history.empty() &&
           (m_commit_history.front().chunk_index + 1) * m_chunk_size_samples <= m_total_dropped_samples) {
        m_commit_history.pop_front();
    }

    // Captured before wrapping with the language tag: the exact tag-free text this pass's prefix
    // was built from, used below as the split point for this pass's own delta. Using this (rather
    // than the full historical m_current_committed_text) is what makes delta computation immune to
    // how aggressively m_commit_history has just been evicted above -- the model was only ever
    // asked to continue from prefix_text, so anything trim_rollback() finds stable beyond it is
    // genuinely new content, regardless of how much *earlier* history no longer appears in the prompt.
    const std::string prefix_text = history_text();
    const std::string prefix = bounded_prefix();

    if (asr_debug_dump_enabled()) {
        asr_debug_dump_chunk("qwen3-asr",
                             m_chunk_count,
                             m_audio_accum,
                             m_pipeline->m_feature_extractor.sampling_rate,
                             prefix);
    }

    const std::string raw_out =
        m_pipeline->infer_streaming_chunk(m_audio_accum, prefix, m_generation_config, m_perf_metrics);

    const std::string full_raw = prefix + raw_out;
    auto [language, text] = Qwen3ASR::parse_asr_output(full_raw, m_generation_config.language);
    m_current_language = std::move(language);
    m_current_text = std::move(text);
    const size_t this_chunk_index = m_chunk_count;
    ++m_chunk_count;

    // candidate_full_text = tokens from THIS pass stable enough to survive into the next decode
    // pass, still including the (tag-free) prefix_text this pass started from -- trim_rollback()
    // only removes trailing tokens, so candidate_full_text always begins with prefix_text verbatim.
    const std::string next_prefix_raw = trim_rollback(full_raw);
    std::string candidate_full_text;
    if (!next_prefix_raw.empty()) {
        auto [next_lang, next_text] = Qwen3ASR::parse_asr_output(next_prefix_raw, m_generation_config.language);
        candidate_full_text = std::move(next_text);
    }

    // This pass's own contribution, relative to the prefix it actually saw -- not a length
    // comparison against the historical total, which broke once eviction made m_commit_history's
    // reconstructed prefix diverge from the ever-growing m_current_committed_text (a shorter,
    // differently-worded candidate_full_text could still out-length the old total and silently
    // clobber it). Appending a prefix-relative delta is safe by construction: it can only grow.
    std::string delta;
    size_t committed_len_this_pass = prefix_text.size();
    if (candidate_full_text.size() > prefix_text.size()) {
        delta = candidate_full_text.substr(prefix_text.size());
        committed_len_this_pass = candidate_full_text.size();
    }
    if (!delta.empty()) {
        m_commit_history.push_back({this_chunk_index, delta});
        m_current_committed_text += delta;
    }

    // Local to this pass -- independent of m_current_committed_text's (unbounded) length, which
    // can legitimately outgrow m_current_text once older commit_history entries are evicted.
    m_current_partial_text = m_current_text.size() >= committed_len_this_pass
                                 ? m_current_text.substr(committed_len_this_pass)
                                 : m_current_text;
    m_current_new_committed_text = delta;
}

std::optional<ASRPartialResult> Qwen3ASRStreamingSessionImpl::push_chunk(const std::vector<float>& pcm16k) {
    m_buffer.insert(m_buffer.end(), pcm16k.begin(), pcm16k.end());

    if (m_buffer.size() < m_chunk_size_samples) {
        return std::nullopt;
    }

    // m_audio_accum has been fully decoded by the end of every prior push_chunk()/finish()
    // call, so its size right before this merge is exactly how much is safe to drop from.
    const size_t already_inferred_samples = m_audio_accum.size();

    // Drain the entire buffer in one pass (single decode per push_chunk call).
    m_audio_accum.insert(m_audio_accum.end(), m_buffer.begin(), m_buffer.end());
    m_buffer.clear();

    m_total_dropped_samples += apply_sliding_window_drop(m_audio_accum,
                                                         already_inferred_samples,
                                                         m_chunk_size_samples,
                                                         m_streaming_config.window_chunk_num,
                                                         m_streaming_config.window_rollback_chunk_num);

    decode_current_accum();
    return ASRPartialResult{m_current_language, m_current_committed_text,
                            m_current_new_committed_text, m_current_partial_text};
}

ASRPartialResult Qwen3ASRStreamingSessionImpl::finish() {
    m_current_new_committed_text = "";

    if (!m_buffer.empty()) {
        const size_t already_inferred_samples = m_audio_accum.size();
        m_audio_accum.insert(m_audio_accum.end(), m_buffer.begin(), m_buffer.end());
        m_buffer.clear();
        m_total_dropped_samples += apply_sliding_window_drop(m_audio_accum,
                                                             already_inferred_samples,
                                                             m_chunk_size_samples,
                                                             m_streaming_config.window_chunk_num,
                                                             m_streaming_config.window_rollback_chunk_num);
        decode_current_accum();
    }

    // Commit any remaining partial tail; final result always has partial_text == "".
    m_current_committed_text += m_current_partial_text;
    m_current_new_committed_text += std::move(m_current_partial_text);
    m_current_partial_text = "";

    return {m_current_language, m_current_committed_text,
            m_current_new_committed_text, m_current_partial_text};
}

}  // namespace ov::genai
