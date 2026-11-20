// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm3_omni/speech_step_hook.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "logger.hpp"

namespace ov::genai {

namespace {

// A frame limit hit before the STOP on the final flush keeps LENGTH, like on SDPA.
GenerationFinishReason stop_reason(GenerationFinishReason reason) {
    return reason == GenerationFinishReason::TOOL_CALL || reason == GenerationFinishReason::LENGTH
               ? reason
               : GenerationFinishReason::STOP;
}

void set_finish_reason(const std::vector<Sequence::Ptr>& sequences, GenerationFinishReason reason) {
    for (const Sequence::Ptr& sequence : sequences) {
        sequence->set_finish_reason(reason);
    }
}

}  // namespace

LFM3SpeechStepHook::ActiveRequest::ActiveRequest(uint64_t id,
                                                 const LFM3SpeechModelConfig& config,
                                                 const PendingRequest& pending)
    : id(id),
      state(config, pending.request, *pending.speech_streamer),
      text_streamer(pending.text_streamer) {}

LFM3SpeechStepHook::LFM3SpeechStepHook(std::unique_ptr<LFM3SpeechModels> models) : m_models(std::move(models)) {}

void LFM3SpeechStepHook::expect_request(const LFM3SpeechRequest& request,
                                        const OmniSpeechStreamerVariant& speech_streamer,
                                        std::shared_ptr<StreamerBase> text_streamer) {
    OPENVINO_ASSERT(!m_pending_request && !m_active, "LFM3-Omni decodes one speech request at a time");
    m_pending_request = PendingRequest{request, &speech_streamer, std::move(text_streamer)};
    m_result.reset();
    m_request_error = nullptr;
    m_text_streamer_failed = false;
}

void LFM3SpeechStepHook::rethrow_request_error() {
    if (m_request_error) {
        std::rethrow_exception(std::exchange(m_request_error, nullptr));
    }
}

void LFM3SpeechStepHook::keep_request_error(const char* source) {
    std::string message = "unknown exception";
    try {
        throw;
    } catch (const std::exception& error) {
        message = error.what();
    } catch (...) {
        // Not an std::exception: the exception itself is still kept below.
    }
    GENAI_WARN("LFM3-Omni speech request ends after a %s failure: %s", source, message.c_str());
    if (!m_request_error) {
        m_request_error = std::current_exception();
    }
}

void LFM3SpeechStepHook::finish_request() {
    try {
        m_result = m_active->state.finish(*m_models);
    } catch (...) {
        keep_request_error("final speech decode");
        m_result = LFM3SpeechResult{};
        m_result->status = GenerationStatus::CANCEL;
    }
    m_active.reset();
}

LFM3SpeechResult LFM3SpeechStepHook::take_result() {
    OPENVINO_ASSERT(m_result, "LFM3-Omni speech request has no result");
    LFM3SpeechResult result = std::move(*m_result);
    m_result.reset();
    return result;
}

bool LFM3SpeechStepHook::on_add_request(uint64_t request_id,
                                        ov::Tensor& inputs_embeds,
                                        std::optional<ov::Tensor>& prompt_ids,
                                        InputsEmbedder& inputs_embedder) {
    if (!m_pending_request) {
        return false;
    }
    OPENVINO_ASSERT(!m_active, "LFM3-Omni speech request ", m_active->id, " is still running");
    const PendingRequest pending = std::move(*m_pending_request);
    m_pending_request.reset();
    m_active.emplace(request_id, m_models->config(), pending);
    LFM3SpeechState& state = m_active->state;

    int64_t last_prompt_token = -1;
    if (prompt_ids && prompt_ids->get_size() > 0) {
        last_prompt_token = prompt_ids->data<const int64_t>()[prompt_ids->get_size() - 1];
    }
    const int64_t marker = state.take_prompt_marker(last_prompt_token);
    if (marker >= 0) {
        inputs_embeds =
            append_lfm3_prompt_row(inputs_embeds, embed_lfm3_token(*inputs_embedder.get_embedding_model(), marker));
        if (prompt_ids) {
            // Prompt ids are [1, num_tokens]; the marker joins them as the last token.
            const size_t num_tokens = prompt_ids->get_size();
            ov::Tensor extended(ov::element::i64, {1, num_tokens + 1});
            std::copy_n(prompt_ids->data<const int64_t>(), num_tokens, extended.data<int64_t>());
            extended.data<int64_t>()[num_tokens] = marker;
            prompt_ids = extended;
        }
    }
    state.start(inputs_embeds.get_shape().at(1));
    return true;
}

void LFM3SpeechStepHook::before_sampling(const std::vector<SequenceGroup::Ptr>& requests, const ov::Tensor& logits) {
    if (!m_active) {
        return;
    }
    OPENVINO_ASSERT(!m_active->frame, "LFM3-Omni speech frame of the previous step was not fed back");
    // The same row walk as Sampler::sample: each scheduled group owns num_running_seqs * output_seq_len rows.
    size_t processed_rows = 0;
    for (const SequenceGroup::Ptr& group : requests) {
        if (!group->is_scheduled()) {
            continue;
        }
        const size_t first_row = processed_rows;
        processed_rows += group->num_running_seqs() * group->get_output_seq_len();
        if (group->get_request_id() != m_active->id) {
            continue;
        }
        // A stopped or cancelled handle frees the request at the end of this step: no further slot is streamed.
        if (!group->requires_sampling() || group->handle_stopped() || group->handle_cancelled()) {
            return;
        }
        const Sequence::Ptr sequence = group->get_running_sequences().front();
        if (m_active->state.next_is_frame()) {
            sample_frame_slot(group, sequence);
        } else {
            sample_text_slot(sequence, logits, first_row + group->get_output_seq_len() - 1);
        }
        return;
    }
}

void LFM3SpeechStepHook::sample_text_slot(const Sequence::Ptr& sequence, const ov::Tensor& logits, size_t row) {
    OPENVINO_ASSERT(logits.get_shape().size() == 3, "LFM3-Omni expects logits [tokens, 1, vocab]");
    const size_t vocab_size = logits.get_shape()[2];
    const float* row_logits = logits.data<const float>() + row * vocab_size;
    std::vector<float> scores(row_logits, row_logits + vocab_size);
    LFM3SpeechState& state = m_active->state;
    const int64_t token = state.sample_text(scores);
    StreamingStatus status = StreamingStatus::RUNNING;
    if (m_active->text_streamer) {
        // An exception of the text streamer ends only this request, as one of the speech streamer does.
        try {
            status = m_active->text_streamer->write(token);
        } catch (...) {
            keep_request_error("text streamer");
            m_text_streamer_failed = true;
            status = StreamingStatus::CANCEL;
        }
    }
    state.accept_text(token, status);
    sequence->set_forced_next_token(token);
}

void LFM3SpeechStepHook::sample_frame_slot(const SequenceGroup::Ptr& group, const Sequence::Ptr& sequence) {
    ov::Tensor hidden_state = sequence->get_hidden_state();
    OPENVINO_ASSERT(hidden_state && hidden_state.is_continuous() && hidden_state.get_element_type() == ov::element::f32,
                    "LFM3-Omni speech needs the f32 hidden states of the language model");
    const size_t hidden_size = hidden_state.get_shape().back();
    // The last row belongs to the last scheduled token, the one the next slot follows.
    const ov::Tensor last_row(ov::element::f32,
                              {1, hidden_size},
                              hidden_state.data<float>() + hidden_state.get_size() - hidden_size);
    LFM3SpeechState& state = m_active->state;
    std::vector<int64_t> codes;
    // A depth decoder, codec or speech streamer exception ends only this request: it must not leave the step.
    try {
        codes = state.accept_frame(m_models->sample_frame(last_row, state), *m_models);
    } catch (...) {
        keep_request_error("speech frame");
        state.end_with(GenerationStatus::CANCEL);
    }
    m_active->frame = PendingFrame{group, sequence, std::move(codes)};
    sequence->set_forced_next_token(m_models->config().audio_out_token_id);
}

std::vector<IStepHook::GeneratedEmbedding> LFM3SpeechStepHook::after_sampling(
    const std::vector<SequenceGroup::Ptr>& requests,
    SamplerOutput& sampler_output) {
    std::vector<GeneratedEmbedding> generated;
    if (!m_active) {
        return generated;
    }
    // Only a request that goes on needs its frame as the next input; one that ended this step is finished below.
    const std::optional<PendingFrame> frame = std::exchange(m_active->frame, std::nullopt);
    if (frame && !m_active->state.finished()) {
        try {
            generated.push_back({frame->sequence_group, frame->sequence, m_models->embed_frame(frame->codes)});
        } catch (...) {
            keep_request_error("audio_out_embeddings");
            m_active->state.end_with(GenerationStatus::CANCEL);
        }
    }
    // Sampling has already cleared the scheduled flag, so every request is checked.
    const auto group = std::find_if(requests.begin(), requests.end(), [&](const SequenceGroup::Ptr& request) {
        return request->get_request_id() == m_active->id;
    });
    if (group != requests.end()) {
        finish_if_ended(*group, sampler_output);
    }
    return generated;
}

void LFM3SpeechStepHook::finish_if_ended(const SequenceGroup::Ptr& group, SamplerOutput& sampler_output) {
    // The speech state ends a request on <|text_end|> rules, the frame limit and streamer statuses, the sampler at
    // max_new_tokens and stop tokens. Either way the request is finished here, before its handle is notified, so a
    // STOP or CANCEL, the final flush included, reaches the handle like on SDPA.
    const bool sampler_finished = group->has_finished();
    const bool state_finished = m_active->state.finished();
    if (!state_finished && !sampler_finished) {
        return;
    }
    finish_request();
    const LFM3SpeechResult& result = *m_result;
    if (result.status == GenerationStatus::CANCEL) {
        group->get_generation_stream()->cancel();
    } else if (result.status == GenerationStatus::STOP) {
        const GenerationFinishReason reason = stop_reason(result.finish_reason);
        group->get_generation_stream()->stop(reason);
        if (sampler_finished) {
            // A stop on the last allowed token: results read the sequence reason first, so LENGTH must not stay.
            set_finish_reason(group->get_sequences(), reason);
        }
    } else if (!sampler_finished) {
        for (const Sequence::Ptr& sequence : group->get_running_sequences()) {
            sequence->set_status(SequenceStatus::FINISHED);
            sequence->set_finish_reason(result.finish_reason);
            sampler_output.m_dropped_sequences.push_back(sequence->get_id());
        }
    } else if (state_finished) {
        // <|text_end|> on the last allowed slot: the sampler, which does not stop on it, has set LENGTH.
        set_finish_reason(group->get_sequences(), result.finish_reason);
    }
}

void LFM3SpeechStepHook::on_request_freed(const SequenceGroup::Ptr& request) {
    if (!m_active || m_active->id != request->get_request_id()) {
        return;
    }
    // Only a request whose handle was stopped or cancelled from outside is still here: after_sampling() finishes the
    // others.
    if (request->handle_cancelled()) {
        m_active->state.end_with(GenerationStatus::CANCEL);
    } else if (request->handle_stopped()) {
        m_active->state.end_with(GenerationStatus::STOP);
    }
    finish_request();
}

void LFM3SpeechStepHook::on_request_dropped(uint64_t request_id) {
    if (m_active && m_active->id == request_id) {
        m_active.reset();
    }
}

}  // namespace ov::genai
