// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "streaming_session.hpp"

#include <algorithm>
#include <chrono>
#include <iterator>

#include "pipeline.hpp"

namespace ov::genai {

namespace {

// UTF-8 encoding of U+FFFD REPLACEMENT CHARACTER — signals a corrupted decode boundary.
static constexpr const char* REPLACEMENT_CHAR_UTF8 = "\xef\xbf\xbd";

}  // namespace

Qwen3ASRStreamingSessionImpl::Qwen3ASRStreamingSessionImpl(Qwen3ASR* pipeline,
                                                           const ASRStreamingConfig& streaming_config,
                                                           const ASRGenerationConfig& generation_config,
                                                           ASRPartialResultCallback callback)
    : m_pipeline{pipeline},
      m_streaming_config{streaming_config},
      m_generation_config{generation_config},
      m_callback{std::move(callback)},
      m_chunk_size_samples{static_cast<size_t>(
          std::max(1.0f, streaming_config.chunk_size_sec) * m_pipeline->m_feature_extractor.sampling_rate)} {
    OPENVINO_ASSERT(m_pipeline != nullptr, "Qwen3ASRStreamingSessionImpl: pipeline pointer must not be null");
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

std::string Qwen3ASRStreamingSessionImpl::compute_prefix() const {
    if (m_chunk_count < m_streaming_config.unfixed_chunk_num || m_accumulated_raw.empty()) {
        return "";
    }

    const TokenizedInputs encoded = m_pipeline->m_tokenizer.encode(m_accumulated_raw);
    const ov::Tensor& ids_tensor = encoded.input_ids;
    const size_t n_tokens = ids_tensor.get_shape()[1];

    size_t rollback = m_streaming_config.unfixed_token_num;

    // Increase rollback until the decoded prefix is free of replacement characters.
    while (true) {
        const size_t keep = (n_tokens > rollback) ? n_tokens - rollback : 0;
        if (keep == 0) {
            return "";
        }
        const int64_t* data = ids_tensor.data<const int64_t>();
        const std::vector<int64_t> kept_ids(data, data + keep);
        const std::string prefix = m_pipeline->m_tokenizer.decode(kept_ids);
        if (prefix.find(REPLACEMENT_CHAR_UTF8) == std::string::npos) {
            return prefix;
        }
        ++rollback;
        if (rollback >= n_tokens) {
            return "";
        }
    }
}

void Qwen3ASRStreamingSessionImpl::decode_current_accum() {
    const std::string prefix = compute_prefix();

    const std::string raw_out =
        m_pipeline->infer_streaming_chunk(m_audio_accum, prefix, m_generation_config, m_perf_metrics);

    m_accumulated_raw = prefix + raw_out;
    auto [language, text] = Qwen3ASR::parse_asr_output(m_accumulated_raw, m_generation_config.language);
    m_current_language = std::move(language);
    m_current_text = std::move(text);
    ++m_chunk_count;

    if (m_callback) {
        m_callback({m_current_language, m_current_text});
    }
}

void Qwen3ASRStreamingSessionImpl::push_chunk(const std::vector<float>& pcm16k) {
    m_buffer.insert(m_buffer.end(), pcm16k.begin(), pcm16k.end());

    while (m_buffer.size() >= m_chunk_size_samples) {
        m_audio_accum.insert(m_audio_accum.end(), m_buffer.begin(), m_buffer.begin() + m_chunk_size_samples);
        m_buffer.erase(m_buffer.begin(), m_buffer.begin() + m_chunk_size_samples);
        decode_current_accum();
    }
}

ASRDecodedResults Qwen3ASRStreamingSessionImpl::finish() {
    if (!m_buffer.empty()) {
        m_audio_accum.insert(m_audio_accum.end(), m_buffer.begin(), m_buffer.end());
        m_buffer.clear();
        decode_current_accum();
    }

    ASRDecodedResults results;
    results.texts = {m_current_text};
    results.scores = {0.0f};
    results.languages = {m_current_language};
    results.perf_metrics = m_perf_metrics;
    return results;
}

ASRPartialResult Qwen3ASRStreamingSessionImpl::get_partial_result() const {
    return {m_current_language, m_current_text};
}

}  // namespace ov::genai
