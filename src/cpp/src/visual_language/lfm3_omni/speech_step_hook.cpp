// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm3_omni/speech_step_hook.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

#include "circular_buffer_queue.hpp"

namespace ov::genai {

namespace {

ov::Tensor embed_token(InputsEmbedder& inputs_embedder, int64_t token_id) {
    ov::Tensor input_ids(ov::element::i64, {1, 1});
    input_ids.data<int64_t>()[0] = token_id;
    const EmbeddingsModel::Ptr embedding = inputs_embedder.get_embedding_model();
    CircularBufferQueueElementGuard<EmbeddingsRequest> guard(embedding->get_request_queue().get());
    const ov::Tensor embeds = embedding->infer(guard.get(), input_ids);
    ov::Tensor result(embeds.get_element_type(), embeds.get_shape());
    std::memcpy(result.data(), embeds.data(), embeds.get_byte_size());
    return result;
}

// Embeddings [1, n, hidden] with one more row: the rows of source, then row [1, 1, hidden].
ov::Tensor append_row(const ov::Tensor& source, const ov::Tensor& row) {
    ov::Shape shape = source.get_shape();
    OPENVINO_ASSERT(shape.size() >= 2 && row.get_byte_size() * shape[shape.size() - 2] == source.get_byte_size(),
                    "LFM3-Omni: the appended row does not match the prompt");
    shape[shape.size() - 2] += 1;
    ov::Tensor result(source.get_element_type(), shape);
    std::memcpy(result.data(), source.data(), source.get_byte_size());
    std::memcpy(static_cast<uint8_t*>(result.data()) + source.get_byte_size(), row.data(), row.get_byte_size());
    return result;
}

}  // namespace

LFM3SpeechStepHook::LFM3SpeechStepHook(std::function<std::unique_ptr<LFM3SpeechModels>()> make_models)
    : m_make_models(std::move(make_models)) {}

void LFM3SpeechStepHook::expect_request(const LFM3SpeechRequest& request,
                                        const OmniSpeechStreamerVariant& speech_streamer,
                                        std::shared_ptr<StreamerBase> text_streamer) {
    OPENVINO_ASSERT(!m_pending_request && m_states.empty(), "LFM3-Omni decodes one speech request at a time");
    if (!m_models) {
        m_models = m_make_models();
    }
    m_pending_request = PendingRequest{request, &speech_streamer, std::move(text_streamer)};
    m_result.reset();
    m_streamer_error = nullptr;
}

void LFM3SpeechStepHook::rethrow_streamer_error() {
    if (m_streamer_error) {
        std::rethrow_exception(std::exchange(m_streamer_error, nullptr));
    }
}

void LFM3SpeechStepHook::finish_request(std::map<uint64_t, LFM3SpeechState>::iterator it) {
    try {
        m_result = it->second.finish(*m_models);
    } catch (...) {
        m_streamer_error = std::current_exception();
        m_result = LFM3SpeechResult{};
        m_result->status = GenerationStatus::CANCEL;
    }
    m_states.erase(it);
    m_text_streamer.reset();
}

LFM3SpeechResult LFM3SpeechStepHook::take_result() {
    OPENVINO_ASSERT(m_result, "LFM3-Omni speech request has no result");
    LFM3SpeechResult result = std::move(*m_result);
    m_result.reset();
    return result;
}

void LFM3SpeechStepHook::on_add_request(uint64_t request_id,
                                        ov::Tensor& inputs_embeds,
                                        std::optional<ov::Tensor>& prompt_ids,
                                        InputsEmbedder& inputs_embedder) {
    if (!m_pending_request) {
        return;
    }
    const PendingRequest pending = std::move(*m_pending_request);
    m_pending_request.reset();
    auto [it, inserted] =
        m_states.try_emplace(request_id, m_models->config(), pending.request, *pending.speech_streamer);
    OPENVINO_ASSERT(inserted, "LFM3-Omni speech request ", request_id, " is already running");
    m_text_streamer = pending.text_streamer;
    LFM3SpeechState& state = it->second;

    int64_t last_prompt_token = -1;
    if (prompt_ids && prompt_ids->get_size() > 0) {
        last_prompt_token = prompt_ids->data<const int64_t>()[prompt_ids->get_size() - 1];
    }
    const int64_t marker = state.take_prompt_marker(last_prompt_token);
    if (marker >= 0) {
        inputs_embeds = append_row(inputs_embeds, embed_token(inputs_embedder, marker));
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
}

void LFM3SpeechStepHook::before_sampling(const std::vector<SequenceGroup::Ptr>& requests, const ov::Tensor& logits) {
    OPENVINO_ASSERT(m_pending_frames.empty(), "LFM3-Omni speech frames of the previous step were not fed back");
    if (m_states.empty()) {
        return;
    }
    OPENVINO_ASSERT(logits.get_shape().size() == 3, "LFM3-Omni expects logits [tokens, 1, vocab]");
    const size_t vocab_size = logits.get_shape()[2];
    std::vector<LFM3SpeechState*> frame_states;
    std::vector<const float*> hidden_rows;
    size_t hidden_size = 0;
    // The same row walk as Sampler::sample: each scheduled group owns num_running_seqs * output_seq_len rows.
    size_t processed_rows = 0;
    for (const SequenceGroup::Ptr& group : requests) {
        if (!group->is_scheduled()) {
            continue;
        }
        const size_t first_row = processed_rows;
        processed_rows += group->num_running_seqs() * group->get_output_seq_len();
        const auto it = m_states.find(group->get_request_id());
        // A stopped or cancelled handle frees the request at the end of this step: no further slot is streamed.
        if (it == m_states.end() || !group->requires_sampling() || group->handle_stopped() ||
            group->handle_cancelled()) {
            continue;
        }
        LFM3SpeechState& state = it->second;
        const Sequence::Ptr sequence = group->get_running_sequences().front();
        if (state.next_is_frame()) {
            const ov::Tensor hidden_state = sequence->get_hidden_state();
            OPENVINO_ASSERT(
                hidden_state && hidden_state.is_continuous() && hidden_state.get_element_type() == ov::element::f32,
                "LFM3-Omni speech needs the f32 hidden states of the language model");
            hidden_size = hidden_state.get_shape().back();
            // The last row belongs to the last scheduled token, the one the next slot follows.
            hidden_rows.push_back(hidden_state.data<const float>() + hidden_state.get_size() - hidden_size);
            frame_states.push_back(&state);
            m_pending_frames.push_back({group, sequence, {}});
        } else {
            const float* row = logits.data<const float>() + (first_row + group->get_output_seq_len() - 1) * vocab_size;
            std::vector<float> scores(row, row + vocab_size);
            const int64_t token = state.sample_text(scores);
            StreamingStatus status = StreamingStatus::RUNNING;
            if (m_text_streamer) {
                // An exception of the text streamer ends only this request, as one of the speech streamer does.
                try {
                    status = m_text_streamer->write(token);
                } catch (...) {
                    m_streamer_error = std::current_exception();
                    status = StreamingStatus::CANCEL;
                }
            }
            state.accept_text(token, status);
            sequence->set_forced_next_token(token);
        }
    }
    if (frame_states.empty()) {
        return;
    }
    ov::Tensor hidden_states(ov::element::f32, {frame_states.size(), hidden_size});
    for (size_t row = 0; row < hidden_rows.size(); ++row) {
        std::copy_n(hidden_rows[row], hidden_size, hidden_states.data<float>() + row * hidden_size);
    }
    std::vector<std::vector<int64_t>> codes = m_models->sample_frames(hidden_states, frame_states);
    for (size_t row = 0; row < frame_states.size(); ++row) {
        // accept_frame() may stream a chunk; an exception of the speech streamer ends only this request.
        m_pending_frames[row].codes = codes[row];
        try {
            m_pending_frames[row].codes = frame_states[row]->accept_frame(std::move(codes[row]), *m_models);
        } catch (...) {
            m_streamer_error = std::current_exception();
            frame_states[row]->end_with(GenerationStatus::CANCEL);
        }
        m_pending_frames[row].sequence->set_forced_next_token(m_models->config().audio_out_token_id);
    }
}

std::vector<IStepHook::GeneratedEmbedding> LFM3SpeechStepHook::after_sampling(
    const std::vector<SequenceGroup::Ptr>& requests,
    SamplerOutput& sampler_output) {
    std::vector<GeneratedEmbedding> generated;
    if (!m_pending_frames.empty()) {
        std::vector<std::vector<int64_t>> frames;
        for (const PendingFrame& frame : m_pending_frames) {
            frames.push_back(frame.codes);
        }
        const ov::Tensor embeds = m_models->embed_frames(frames);
        const size_t hidden_size = embeds.get_shape().at(1);
        for (size_t row = 0; row < m_pending_frames.size(); ++row) {
            ov::Tensor embedding(ov::element::f32, {1, 1, hidden_size});
            std::copy_n(embeds.data<const float>() + row * hidden_size, hidden_size, embedding.data<float>());
            generated.push_back({m_pending_frames[row].sequence_group, m_pending_frames[row].sequence, embedding});
        }
        m_pending_frames.clear();
    }
    // The speech state ends a request on <|text_end|> rules, the frame limit and streamer statuses, the sampler at
    // max_new_tokens and stop tokens. Either way the request is finished here, before its handle is notified, so a
    // STOP or CANCEL, the final flush included, reaches the handle like on SDPA. Sampling has already cleared the
    // scheduled flag, so every request is checked.
    for (const SequenceGroup::Ptr& group : requests) {
        const auto it = m_states.find(group->get_request_id());
        const bool sampler_finished = group->has_finished();
        if (it == m_states.end() || (!it->second.finished() && !sampler_finished)) {
            continue;
        }
        finish_request(it);
        if (m_result->status == GenerationStatus::CANCEL) {
            group->get_generation_stream()->cancel();
        } else if (m_result->status == GenerationStatus::STOP) {
            group->get_generation_stream()->stop(m_result->finish_reason == GenerationFinishReason::TOOL_CALL
                                                     ? GenerationFinishReason::TOOL_CALL
                                                     : GenerationFinishReason::STOP);
        } else if (!sampler_finished) {
            for (const Sequence::Ptr& sequence : group->get_running_sequences()) {
                sequence->set_status(SequenceStatus::FINISHED);
                sequence->set_finish_reason(m_result->finish_reason);
                sampler_output.m_dropped_sequences.push_back(sequence->get_id());
            }
        }
    }
    return generated;
}

void LFM3SpeechStepHook::on_request_freed(const SequenceGroup::Ptr& request) {
    const auto it = m_states.find(request->get_request_id());
    if (it == m_states.end()) {
        return;
    }
    // Only a request whose handle was stopped or cancelled from outside is still here: after_sampling() finishes the
    // others.
    if (request->handle_cancelled()) {
        it->second.end_with(GenerationStatus::CANCEL);
    } else if (request->handle_stopped()) {
        it->second.end_with(GenerationStatus::STOP);
    }
    finish_request(it);
}

void LFM3SpeechStepHook::on_request_dropped(uint64_t request_id) {
    m_states.erase(request_id);
    m_text_streamer.reset();
    m_pending_frames.clear();
}

}  // namespace ov::genai
