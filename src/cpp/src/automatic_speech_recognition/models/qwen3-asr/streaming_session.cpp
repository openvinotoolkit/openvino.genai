// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "streaming_session.hpp"

#include <algorithm>
#include <chrono>
#include <iterator>

#include "automatic_speech_recognition/sliding_window.hpp"
#include "pipeline_base.hpp"

namespace ov::genai {

namespace {

// UTF-8 encoding of U+FFFD REPLACEMENT CHARACTER — signals a corrupted decode boundary.
static constexpr const char* REPLACEMENT_CHAR_UTF8 = "\xef\xbf\xbd";

// Marks the end of the "language <Lang><asr_text>" header that precedes the actual transcript in
// auto-detect mode. Rollback must never trim back into it -- see trim_rollback().
static const std::string ASR_TEXT_TAG = "<asr_text>";

// Number of initial decode passes whose output is never committed and never used as a prefix.
constexpr size_t WARMUP_PASSES = 2;

}  // namespace

Qwen3ASRStreamingSessionImpl::Qwen3ASRStreamingSessionImpl(Qwen3ASRPipelineBase* pipeline,
                                                           const ASRStreamingConfig& streaming_config,
                                                           const ASRGenerationConfig& generation_config)
    : ASRStreamingSessionImpl{static_cast<size_t>(streaming_config.chunk_size_sec *
                                                    pipeline->m_feature_extractor.sampling_rate)},
      m_pipeline{pipeline},
      m_streaming_config{streaming_config},
      m_generation_config{generation_config},
      m_chunk_size_samples{static_cast<size_t>(streaming_config.chunk_size_sec *
                                               pipeline->m_feature_extractor.sampling_rate)} {}

std::string Qwen3ASRStreamingSessionImpl::trim_rollback(const std::string& raw) const {
    if (m_chunk_count < WARMUP_PASSES || raw.empty()) {
        return "";
    }

    const TokenizedInputs encoded = m_pipeline->m_tokenizer.encode(raw);
    const ov::Tensor& ids_tensor = encoded.input_ids;
    const size_t n_tokens = ids_tensor.get_shape()[1];
    const int64_t* data = ids_tensor.data<const int64_t>();

    // In auto-detect mode, `raw` is prefixed with a "language <Lang><asr_text>" header the model
    // generated itself (rather than a fixed part of the prompt template), so it's just as much a
    // target for rollback trimming as the real transcript that follows it. If rollback ever cuts
    // into the header, parse_asr_output() can no longer find "<asr_text>" and falls back to
    // returning the (now header-fragment-containing) text unstripped, leaking literal template text
    // into the committed transcript. Find the smallest token count whose decode still reproduces
    // the header in full, and never trim below it.
    size_t min_keep_tokens = 0;
    const size_t tag_pos = raw.find(ASR_TEXT_TAG);
    if (tag_pos != std::string::npos) {
        const size_t tag_end = tag_pos + ASR_TEXT_TAG.size();
        for (size_t keep = 1; keep <= n_tokens; ++keep) {
            const std::vector<int64_t> kept_ids(data, data + keep);
            const std::string decoded = m_pipeline->m_tokenizer.decode(kept_ids);
            if (decoded.size() >= tag_end && decoded.find(ASR_TEXT_TAG) != std::string::npos) {
                min_keep_tokens = keep;
                break;
            }
        }
        if (min_keep_tokens == 0) {
            // No token count reproduced the header intact (tokenizer re-encoding drift); refuse to
            // commit anything this pass rather than risk emitting a truncated header.
            return "";
        }
    }

    size_t rollback = m_streaming_config.unfixed_token_num;

    // Increase rollback until the decoded prefix is free of replacement characters, but never trim
    // past min_keep_tokens (the end of the "<asr_text>" header, when present).
    while (true) {
        const size_t keep = (n_tokens > rollback) ? n_tokens - rollback : 0;
        if (keep == 0 || keep < min_keep_tokens) {
            return "";
        }
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

void Qwen3ASRStreamingSessionImpl::decode_current_accum(ASRPerfMetrics& perf_metrics) {
    // Evict commit_history entries whose grounding audio no longer exists in m_audio_accum, tied
    // directly to how far the sliding window has actually rolled (m_total_dropped_samples) rather
    // than a fixed chunk-count margin.
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

    const std::string raw_out =
        m_pipeline->infer_streaming_chunk(m_audio_accum, prefix, m_generation_config, perf_metrics);

    const std::string full_raw = prefix + raw_out;
    auto [language, text] = m_pipeline->parse_asr_output(full_raw, m_generation_config.language);
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
        auto [next_lang, next_text] = m_pipeline->parse_asr_output(next_prefix_raw, m_generation_config.language);
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

ASRStreamingResult Qwen3ASRStreamingSessionImpl::process(const std::vector<float>& audio, bool is_last) {
    const auto start_time = std::chrono::steady_clock::now();
    ASRPerfMetrics perf_metrics;
    perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    m_current_new_committed_text.clear();
    if (!audio.empty()) {
        // m_audio_accum has been fully decoded by every previous pass, so its size before the merge is exactly
        // how much audio is safe to drop.
        const size_t already_inferred_samples = m_audio_accum.size();
        m_audio_accum.insert(m_audio_accum.end(), audio.begin(), audio.end());
        m_total_dropped_samples += apply_sliding_window_drop(m_audio_accum,
                                                             already_inferred_samples,
                                                             m_chunk_size_samples,
                                                             m_streaming_config.window_chunk_num,
                                                             m_streaming_config.unfixed_chunk_num);
        decode_current_accum(perf_metrics);
    }

    if (is_last) {
        // The final result commits the remaining unstable tail.
        m_current_committed_text += m_current_partial_text;
        m_current_new_committed_text += m_current_partial_text;
        m_current_partial_text.clear();
    }

    PerfMetrics::emplace_duration(perf_metrics.raw_metrics.generate_durations, start_time);
    perf_metrics.evaluate_statistics(start_time);
    return {m_current_language,
            m_current_committed_text,
            m_current_new_committed_text,
            m_current_partial_text,
            is_last,
            std::move(perf_metrics)};
}

}  // namespace ov::genai
