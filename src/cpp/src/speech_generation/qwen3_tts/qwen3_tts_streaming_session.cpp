// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_streaming_session.hpp"

#include "openvino/genai/perf_metrics.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"

namespace ov {
namespace genai {

std::unique_ptr<Text2SpeechStreamingSessionImpl> Qwen3TTSImpl::create_streaming_session_impl(
    std::shared_ptr<Text2SpeechPipelineImpl> owner,
    const ov::Tensor& speaker_embedding,
    const SpeechGenerationConfig& generation_config,
    const Text2SpeechStreamingConfig& streaming_config) {
    auto self = std::static_pointer_cast<Qwen3TTSImpl>(std::move(owner));
    OPENVINO_ASSERT(self.get() == this, "The streaming session owner must be this pipeline implementation");
    return std::make_unique<Qwen3TTSStreamingSessionImpl>(std::move(self), acquire_streaming_session_slot(),
                                                          speaker_embedding, generation_config, streaming_config);
}

Qwen3TTSStreamingSessionImpl::Qwen3TTSStreamingSessionImpl(std::shared_ptr<Qwen3TTSImpl> owner,
                                                           Text2SpeechStreamingSessionSlot slot,
                                                           const ov::Tensor& speaker_embedding,
                                                           const SpeechGenerationConfig& generation_config,
                                                           const Text2SpeechStreamingConfig& streaming_config)
    : m_owner(std::move(owner)),
      m_slot(std::move(slot)),
      m_engine(m_owner->m_engine),
      m_config(generation_config),
      m_committer(streaming_config.max_holdback_chars) {
    // The text arrives over time, so the whole of it cannot go into the prefill.
    m_config.non_streaming_mode = false;
    m_voice = m_owner->resolve_voice(speaker_embedding, m_config);
    m_suppress_tokens = m_engine.compute_suppress_tokens();
    m_hidden = m_engine.talker_hidden_size();

    // The three role tokens of "<|im_start|>assistant\n" open every turn.
    const std::vector<int64_t> role_ids = encode("<|im_start|>assistant\n");
    OPENVINO_ASSERT(role_ids.size() == 3, "Expected the Qwen3-TTS role prefix to be 3 tokens, got ", role_ids.size());

    if (m_voice.icl) {
        // Reference transcript: the tokens between the role prefix and the end-of-turn markers.
        const auto ref_ids = encode("<|im_start|>assistant\n" + m_voice.ref_text + "<|im_end|>\n");
        OPENVINO_ASSERT(ref_ids.size() > 5, "Qwen3 voice-clone reference transcript tokenized too short");
        const std::vector<int64_t> ref_content_ids(ref_ids.begin() + 3, ref_ids.end() - 2);
        m_icl = m_engine.prepare_voice_clone(role_ids, m_engine.embed_text_tokens(ref_content_ids),
                                             m_voice.speaker_embedding, m_voice.ref_codec_ids, m_config);
    } else {
        m_head = m_engine.build_prefill_head(role_ids, m_voice.speaker_embedding, m_voice.language, m_voice.speaker,
                                             m_voice.base_model);
    }
}

std::vector<int64_t> Qwen3TTSStreamingSessionImpl::encode(const std::string& text) {
    const ov::Tensor ids = m_engine.tokenizer().encode(text).input_ids;
    OPENVINO_ASSERT(ids.get_element_type() == ov::element::i64 && ids.get_shape().size() == 2 && ids.get_shape()[0] == 1,
                    "Qwen3-TTS tokenizer must produce an i64 input_ids tensor of shape [1, T]");
    const int64_t* data = ids.data<const int64_t>();
    return std::vector<int64_t>(data, data + ids.get_shape()[1]);
}

void Qwen3TTSStreamingSessionImpl::push(const std::string& text) {
    std::lock_guard<std::mutex> lock{m_input_mutex};
    OPENVINO_ASSERT(!m_end_requested, "Text2SpeechStreamingSession::push() called after end().");
    m_incoming += text;
}

void Qwen3TTSStreamingSessionImpl::end() {
    std::lock_guard<std::mutex> lock{m_input_mutex};
    OPENVINO_ASSERT(!m_end_requested, "Text2SpeechStreamingSession::end() may only be called once.");
    m_end_requested = true;
}

ov::Tensor Qwen3TTSStreamingSessionImpl::rows_tensor(size_t first, size_t count) const {
    ov::Tensor rows(ov::element::f32, ov::Shape{1, count, m_hidden});
    std::copy_n(m_rows.data() + first * m_hidden, count * m_hidden, rows.data<float>());
    return rows;
}

// Tokenizes and embeds the text the committer has released since the last call.
void Qwen3TTSStreamingSessionImpl::absorb_committed_text() {
    const std::string segment = m_committer.take();
    if (segment.empty()) {
        return;
    }
    const auto ids = qwen3_tts::tokenize_target_segment(
        [this](const std::string& text) { return encode(text); }, segment);
    if (ids.empty()) {
        return;
    }
    const ov::Tensor rows = m_engine.embed_text_tokens(ids);
    OPENVINO_ASSERT(rows.get_shape() == ov::Shape({1, ids.size(), m_hidden}), "Unexpected text embedding shape ",
                    rows.get_shape());
    m_rows.insert(m_rows.end(), rows.data<const float>(), rows.data<const float>() + rows.get_size());
    m_num_rows += ids.size();
}

// Builds the prefill and the run once there is enough text. False: not yet.
bool Qwen3TTSStreamingSessionImpl::start_run() {
    const bool ended = m_committer.ended();
    if (!ended) {
        // Plain voices need the first text token. ICL needs enough target tokens to cover the reference, and at least one.
        const size_t needed = m_voice.icl ? std::max<size_t>(m_icl->target_rows_needed(), 1) : 1;
        if (m_num_rows < needed) {
            return false;
        }
    }
    OPENVINO_ASSERT(m_num_rows > 0, "The text of a Qwen3-TTS streaming session ended without any speakable text.");

    if (m_voice.icl) {
        const auto prefill = m_engine.assemble_icl_prefill(*m_icl, rows_tensor(0, m_num_rows), ended, m_config.instruct);
        m_run.emplace(*m_owner, prefill.talker_prefill, prefill.tts_pad, ov::Tensor(), m_config, m_suppress_tokens,
                      m_icl->ref_codec_ids, true, true);
        m_rows_given = m_num_rows;
        // The text rows past the reference (and tts_eos, when the text has ended) are the text of the frames to come.
        if (prefill.trailing_text_hidden && prefill.trailing_text_hidden.get_shape()[1] > 0) {
            m_run->text().append(prefill.trailing_text_hidden.data<const float>(),
                                 prefill.trailing_text_hidden.get_shape()[1]);
        }
        if (ended) {
            m_run->text().finish();
            m_text_closed = true;
        }
    } else {
        const ov::Tensor prefill = m_engine.prepend_instruct_embed(
            m_engine.build_streaming_prefill(*m_head, rows_tensor(0, 1)), m_config.instruct);
        m_run.emplace(*m_owner, prefill, m_head->tts_pad, ov::Tensor(), m_config, m_suppress_tokens, ov::Tensor(),
                      true, true);
        m_rows_given = 1;  // the first token is in the prefill
    }
    return true;
}

// Hands the rows committed since the last call to the run, and closes its text after end().
void Qwen3TTSStreamingSessionImpl::feed_text() {
    if (!m_run || m_text_closed) {
        return;
    }
    if (m_rows_given < m_num_rows) {
        m_run->text().append(m_rows.data() + m_rows_given * m_hidden, m_num_rows - m_rows_given);
        m_rows_given = m_num_rows;
    }
    if (m_committer.ended()) {
        // tts_eos tells the talker the text is over. It is the last row, so the talker may end the utterance after it.
        const ov::Tensor& tts_eos = m_voice.icl ? m_icl->tts_eos : m_head->tts_eos;
        m_run->text().append(tts_eos.data<const float>(), 1);
        m_run->text().finish();
        m_text_closed = true;
    }
}

Text2SpeechStreamingResult Qwen3TTSStreamingSessionImpl::make_result(std::vector<float> samples,
                                                                     bool is_final,
                                                                     TimePoint start) {
    Text2SpeechStreamingResult result;
    result.speech = ov::Tensor(ov::element::f32, ov::Shape{samples.size()});
    if (!samples.empty()) {
        std::copy(samples.begin(), samples.end(), result.speech.data<float>());
    }
    result.output_sample_rate = m_engine.output_sample_rate();
    result.start_sample = m_samples_returned;
    result.is_final = is_final;
    if (is_final) {
        result.finish_reason = m_run->ended_by_eos() ? GenerationFinishReason::STOP : GenerationFinishReason::LENGTH;
    }
    m_samples_returned += samples.size();
    result.perf_metrics.num_generated_samples = samples.size();
    result.perf_metrics.raw_metrics.generate_durations.emplace_back(
        PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start));
    result.perf_metrics.evaluate_statistics();
    return result;
}

std::optional<Text2SpeechStreamingResult> Qwen3TTSStreamingSessionImpl::read() {
    std::lock_guard<std::mutex> read_lock{m_read_mutex};
    if (m_final_returned) {
        return std::nullopt;
    }
    const TimePoint start = std::chrono::steady_clock::now();

    std::string incoming;
    bool end_requested = false;
    {
        std::lock_guard<std::mutex> lock{m_input_mutex};
        incoming = std::move(m_incoming);
        m_incoming.clear();
        end_requested = m_end_requested;
    }
    if (!incoming.empty()) {
        m_committer.push(incoming);
    }
    if (end_requested && !m_committer.ended()) {
        m_committer.end();
    }
    absorb_committed_text();

    if (!m_run && !start_run()) {
        return std::nullopt;
    }
    feed_text();

    std::vector<float> chunk;
    while (true) {
        switch (m_run->step(chunk)) {
        case Qwen3TTSImpl::TalkerRun::Status::FRAME:
            if (!chunk.empty()) {
                return make_result(std::move(chunk), false, start);
            }
            break;
        case Qwen3TTSImpl::TalkerRun::Status::NEED_TEXT: {
            // The next frame waits for text. Audio that is already decoded does not have to.
            auto decoded = m_run->take_decoded();
            if (decoded.empty()) {
                return std::nullopt;
            }
            return make_result(std::move(decoded), false, start);
        }
        case Qwen3TTSImpl::TalkerRun::Status::DONE: {
            m_final_returned = true;
            return make_result(m_run->finish(), true, start);
        }
        }
    }
}

}  // namespace genai
}  // namespace ov
