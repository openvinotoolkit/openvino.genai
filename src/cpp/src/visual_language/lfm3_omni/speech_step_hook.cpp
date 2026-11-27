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
    /// @brief Set once the speech decode ended; after_remove_request() hands it to the payload.
    std::optional<LFM3SpeechResult> result;
};

LFM3SpeechStepHook::LFM3SpeechStepHook(std::unique_ptr<LFM3SpeechModels> models, EmbeddingsModel::Ptr embedding)
    : m_models(std::move(models)),
      m_embedding(std::move(embedding)) {
    OPENVINO_ASSERT(m_embedding, "LFM3-Omni speech needs a language model with embedding inputs");
}

LFM3SpeechStepHook::RequestState& LFM3SpeechStepHook::as_request_state(StepRequestState& state) {
    // Only before_add_request() of the hook installed on a pipeline creates the step states of its requests.
    return static_cast<RequestState&>(state);
}

LFM3SpeechStepHook::RequestState& LFM3SpeechStepHook::as_request_state(const SequenceGroup& request) {
    auto* const state = request.get_step_state();
    OPENVINO_ASSERT(state, "LFM3-Omni speech request ", request.get_request_id(), " has no step state");
    return as_request_state(*state);
}

std::unique_ptr<StepRequestState> LFM3SpeechStepHook::before_add_request(
    ov::Tensor& inputs_embeds,
    std::optional<ov::Tensor>& prompt_ids,
    SamplingOverrides& sampling_overrides,
    const std::shared_ptr<StepRequestPayload>& payload) {
    if (!payload) {
        return nullptr;
    }
    auto speech_payload = std::dynamic_pointer_cast<LFM3SpeechPayload>(payload);
    OPENVINO_ASSERT(speech_payload, "The LFM3-Omni speech hook got a step payload that is not LFM3-Omni speech");
    auto state = std::make_unique<RequestState>(m_models->config(), std::move(speech_payload));
    auto& speech = state->speech;

    int64_t last_prompt_token = -1;
    if (prompt_ids && prompt_ids->get_size() > 0) {
        last_prompt_token = prompt_ids->data<const int64_t>()[prompt_ids->get_size() - 1];
    }
    const int64_t marker = speech.take_prompt_marker(last_prompt_token);
    if (marker >= 0) {
        inputs_embeds = append_lfm3_prompt_row(inputs_embeds, embed_lfm3_token(*m_embedding, marker));
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
    sampling_overrides = lfm3_text_sampling_overrides(m_models->config());
    return state;
}

void LFM3SpeechStepHook::after_remove_request(StepRequestState& step_state, GenerationStatus status) {
    auto& state = as_request_state(step_state);
    if (status == GenerationStatus::FAILED || status == GenerationStatus::IGNORED) {
        // No speech model or streamer may run for a dropped request, and its caller reads no result.
        GENAI_DEBUG("LFM3-Omni speech request ends without a result, status %d", static_cast<int>(status));
        return;
    }
    if (!state.result) {
        // Only a request whose handle was stopped or cancelled from outside is still running: after_sampling() finishes
        // the others.
        if (status == GenerationStatus::STOP || status == GenerationStatus::CANCEL) {
            state.speech.end_with(status);
        }
        state.result = state.speech.finish(*m_models);
    }
    state.payload->result = std::move(state.result);
}

std::vector<std::optional<int64_t>> LFM3SpeechStepHook::before_sampling(
    const std::vector<SequenceGroup::Ptr>& requests) {
    std::vector<std::optional<int64_t>> tokens;
    tokens.reserve(requests.size());
    for (const auto& request : requests) {
        const auto& state = as_request_state(*request);
        const auto& sequence = request->get_sequences().front();
        // A request is freed in the step its speech ends, so it never samples again.
        OPENVINO_ASSERT(!state.result,
                        "LFM3-Omni speech request ",
                        request->get_request_id(),
                        " samples after its speech ended");
        OPENVINO_ASSERT(state.speech.generated_slots() == sequence->get_generated_len(),
                        "LFM3-Omni speech state is out of step with its sequence: ",
                        state.speech.generated_slots(),
                        " slots, ",
                        sequence->get_generated_len(),
                        " tokens");
        // A frame slot feeds back <|audio_out|> with the frame embedding instead of a sampled token.
        tokens.push_back(state.speech.next_is_frame() ? std::optional<int64_t>{m_models->config().audio_out_token_id}
                                                      : std::nullopt);
    }
    return tokens;
}

std::vector<IStepHook::RequestOutcome> LFM3SpeechStepHook::after_sampling(
    const std::vector<SequenceGroup::Ptr>& requests) {
    std::vector<RequestOutcome> outcomes;
    outcomes.reserve(requests.size());
    // One request at a time: each speech model has a single infer request, and a batch > 1 export is unverified.
    for (const auto& request : requests) {
        outcomes.push_back(after_slot(*request));
    }
    return outcomes;
}

IStepHook::RequestOutcome LFM3SpeechStepHook::after_slot(const SequenceGroup& request) {
    auto& state = as_request_state(request);
    auto& speech = state.speech;
    const auto& sequence = request.get_sequences().front();
    // The sampler already appended the token of this slot.
    OPENVINO_ASSERT(speech.generated_slots() + 1 == sequence->get_generated_len(),
                    "LFM3-Omni speech state is out of step with its sequence: ",
                    speech.generated_slots() + 1,
                    " slots, ",
                    sequence->get_generated_len(),
                    " tokens");
    // A depth decoder, codec or streamer exception fails only this request: it must not leave the step.
    try {
        std::optional<std::vector<int64_t>> frame_codes;
        if (speech.next_is_frame()) {
            frame_codes = accept_frame_slot(state, *sequence);
        } else {
            accept_text_slot(state, *sequence);
        }
        RequestOutcome outcome;
        if (!speech.finished() && !sequence->has_finished()) {
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
                   static_cast<unsigned long>(request.get_request_id()),
                   current_exception_message().c_str());
        RequestOutcome outcome;
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
