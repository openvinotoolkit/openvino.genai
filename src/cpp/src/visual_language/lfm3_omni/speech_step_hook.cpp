// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm3_omni/speech_step_hook.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

#include "logger.hpp"

namespace ov::genai {

namespace {

/// @brief The message of the exception being handled.
std::string current_exception_message() {
    try {
        throw;
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
        return "unknown exception";
    }
}

StreamingStatus text_stream_status(const LFM3SpeechPayload& payload) {
    return payload.text_streamer ? payload.text_streamer->get_status() : StreamingStatus::RUNNING;
}

/// @brief The hidden state [1, hidden_size] of the last scheduled token, the one the next slot follows; a view that
/// lives as long as the hidden state of sequence.
ov::Tensor last_hidden_row(const Sequence& sequence) {
    const ov::Tensor hidden_state = sequence.get_hidden_state();
    OPENVINO_ASSERT(hidden_state && hidden_state.is_continuous() && hidden_state.get_element_type() == ov::element::f32,
                    "LFM3-Omni speech needs the f32 hidden states of the language model");
    const size_t hidden_size = hidden_state.get_shape().back();
    return ov::Tensor(ov::element::f32,
                      {1, hidden_size},
                      hidden_state.data<float>() + hidden_state.get_size() - hidden_size);
}

}  // namespace

LFM3SpeechPayload::LFM3SpeechPayload(LFM3SpeechRequest request,
                                     const OmniSpeechStreamerVariant& speech_streamer,
                                     std::shared_ptr<ThreadedStreamerWrapper> text_streamer)
    : request(std::move(request)),
      speech_streamer(speech_streamer),
      text_streamer(std::move(text_streamer)) {}

struct LFM3SpeechStepHook::RequestState final : StepRequestState {
    RequestState(const LFM3SpeechModelConfig& config, std::shared_ptr<LFM3SpeechPayload> payload)
        : payload(std::move(payload)),
          speech(config, this->payload->request, this->payload->speech_streamer) {}

    std::shared_ptr<LFM3SpeechPayload> payload;
    LFM3SpeechState speech;
    /// @brief Set once the speech decode ended; on_request_done() hands it to the payload.
    std::optional<LFM3SpeechResult> result;
};

LFM3SpeechStepHook::LFM3SpeechStepHook(std::unique_ptr<LFM3SpeechModels> models) : m_models(std::move(models)) {}

LFM3SpeechStepHook::RequestState& LFM3SpeechStepHook::as_request_state(StepRequestState& state) {
    // Only create_request_state() of the hook installed on a pipeline creates the step states of its requests.
    return static_cast<RequestState&>(state);
}

std::unique_ptr<StepRequestState> LFM3SpeechStepHook::create_request_state(const Admission& admission) {
    if (!admission.payload) {
        return nullptr;
    }
    auto speech_payload = std::dynamic_pointer_cast<LFM3SpeechPayload>(admission.payload);
    OPENVINO_ASSERT(speech_payload, "The LFM3-Omni speech hook got a step payload that is not LFM3-Omni speech");
    OPENVINO_ASSERT(admission.inputs_embedder, "LFM3-Omni speech needs a language model with embedding inputs");
    auto state = std::make_unique<RequestState>(m_models->config(), std::move(speech_payload));
    auto& speech = state->speech;
    ov::Tensor& inputs_embeds = admission.inputs;
    std::optional<ov::Tensor>& prompt_ids = admission.prompt_ids;

    int64_t last_prompt_token = -1;
    if (prompt_ids && prompt_ids->get_size() > 0) {
        last_prompt_token = prompt_ids->data<const int64_t>()[prompt_ids->get_size() - 1];
    }
    const int64_t marker = speech.take_prompt_marker(last_prompt_token);
    if (marker >= 0) {
        inputs_embeds = append_lfm3_prompt_row(
            inputs_embeds,
            embed_lfm3_token(*admission.inputs_embedder->get_embedding_model(), marker));
        if (prompt_ids) {
            // Prompt ids are [1, num_tokens]; the marker joins them as the last token.
            const size_t num_tokens = prompt_ids->get_size();
            ov::Tensor extended(ov::element::i64, {1, num_tokens + 1});
            std::copy_n(prompt_ids->data<const int64_t>(), num_tokens, extended.data<int64_t>());
            extended.data<int64_t>()[num_tokens] = marker;
            prompt_ids = extended;
        }
    }
    speech.start(inputs_embeds.get_shape().at(1));
    admission.sampling_overrides = lfm3_text_sampling_overrides(m_models->config());
    return state;
}

void LFM3SpeechStepHook::on_request_done(StepRequestState& step_state, const RequestDone& done) {
    auto& state = as_request_state(step_state);
    if (done.status == GenerationStatus::FAILED || done.status == GenerationStatus::IGNORED) {
        // No speech model or streamer may run for a dropped request, and its caller reads no result.
        GENAI_DEBUG("LFM3-Omni speech request ends without a result, status %d", static_cast<int>(done.status));
        return;
    }
    if (!state.result) {
        // Only a request whose handle was stopped or cancelled from outside is still running: after_sampling() finishes
        // the others.
        if (done.status == GenerationStatus::STOP || done.status == GenerationStatus::CANCEL) {
            state.speech.end_with(done.status);
        }
        state.result = state.speech.finish(*m_models);
    }
    state.payload->result = std::move(state.result);
}

std::vector<std::optional<int64_t>> LFM3SpeechStepHook::force_tokens(const std::vector<Row>& rows) {
    std::vector<std::optional<int64_t>> tokens;
    tokens.reserve(rows.size());
    for (const Row& row : rows) {
        const auto& state = as_request_state(row.state);
        // A request is freed in the step its speech ends, so it never samples again.
        OPENVINO_ASSERT(!state.result,
                        "LFM3-Omni speech request ",
                        row.sequence_group->get_request_id(),
                        " samples after its speech ended");
        OPENVINO_ASSERT(state.speech.generated_slots() == row.sequence->get_generated_len(),
                        "LFM3-Omni speech state is out of step with its sequence: ",
                        state.speech.generated_slots(),
                        " slots, ",
                        row.sequence->get_generated_len(),
                        " tokens");
        // A frame slot feeds back <|audio_out|> with the frame embedding instead of a sampled token.
        tokens.push_back(state.speech.next_is_frame() ? std::optional<int64_t>{m_models->config().audio_out_token_id}
                                                      : std::nullopt);
    }
    return tokens;
}

std::vector<IStepHook::RowOutcome> LFM3SpeechStepHook::after_sampling(const std::vector<Row>& rows) {
    std::vector<RowOutcome> outcomes;
    outcomes.reserve(rows.size());
    // One row at a time: each speech model has a single infer request, and a batch > 1 export is unverified.
    for (const Row& row : rows) {
        outcomes.push_back(after_slot(row));
    }
    return outcomes;
}

IStepHook::RowOutcome LFM3SpeechStepHook::after_slot(const Row& row) {
    auto& state = as_request_state(row.state);
    auto& speech = state.speech;
    // The sampler already appended the token of this slot.
    OPENVINO_ASSERT(speech.generated_slots() + 1 == row.sequence->get_generated_len(),
                    "LFM3-Omni speech state is out of step with its sequence: ",
                    speech.generated_slots() + 1,
                    " slots, ",
                    row.sequence->get_generated_len(),
                    " tokens");
    // A depth decoder, codec or streamer exception fails only this request: it must not leave the step.
    try {
        std::optional<std::vector<int64_t>> frame_codes;
        if (speech.next_is_frame()) {
            frame_codes = accept_frame_slot(state, *row.sequence);
        } else {
            accept_text_slot(state, *row.sequence);
        }
        RowOutcome outcome;
        if (!speech.finished() && !row.sequence->has_finished()) {
            if (frame_codes) {
                outcome.next_embedding = m_models->embed_frame(*frame_codes);
            }
            return outcome;
        }
        // The speech state ends a request on <|text_end|> rules, the frame limit and streamer statuses, the sampler
        // at max_new_tokens and stop tokens. Either way the result is final here, before the handle is notified.
        if (const auto& text_streamer = state.payload->text_streamer) {
            // Callbacks of queued tokens run before the result is final, so a status or error of the last one counts.
            text_streamer->end();
            speech.accept_text_stream_status(text_streamer->get_status());
        }
        state.result = speech.finish(*m_models);
        // A request only the sampler ended keeps the sampler's reason, unless the final flush stopped or cancelled it.
        if (state.result->status != GenerationStatus::FINISHED || speech.finished()) {
            outcome.ending = Ending{state.result->status, state.result->finish_reason};
        }
        return outcome;
    } catch (...) {
        GENAI_WARN("LFM3-Omni speech request %lu fails: %s",
                   static_cast<unsigned long>(row.sequence_group->get_request_id()),
                   current_exception_message().c_str());
        RowOutcome outcome;
        outcome.error = std::current_exception();
        return outcome;
    }
}

std::vector<int64_t> LFM3SpeechStepHook::accept_frame_slot(RequestState& state, const Sequence& sequence) {
    auto& speech = state.speech;
    auto codes = speech.accept_frame(m_models->sample_frame(last_hidden_row(sequence), speech), *m_models);
    // The stop of an asynchronous text streamer may arrive in a frame slot, after its last text token.
    speech.accept_text_stream_status(text_stream_status(*state.payload));
    return codes;
}

void LFM3SpeechStepHook::accept_text_slot(RequestState& state, const Sequence& sequence) {
    const auto status = text_stream_status(*state.payload);
    const auto token = sequence.get_generated_ids().back();
    if (state.payload->text_streamer) {
        state.payload->text_streamer->write(token);
    }
    state.speech.accept_text(token, sequence.get_generated_log_probs().back(), status);
}

}  // namespace ov::genai
