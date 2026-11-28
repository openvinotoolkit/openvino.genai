// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "continuous_batching/step_hook.hpp"
#include "continuous_batching/threaded_streamer.hpp"
#include "visual_language/lfm3_omni/speech_core.hpp"

namespace ov::genai {

/// @brief Speech settings of one request, built by the caller of generate(); after_remove_request() stores the result.
struct LFM3SpeechPayload final : StepRequestPayload {
    LFM3SpeechPayload(LFM3SpeechRequest request,
                      const OmniSpeechStreamerVariant& speech_streamer,
                      std::shared_ptr<ThreadedStreamerWrapper> text_streamer);

    const LFM3SpeechRequest request;
    /// @brief Must outlive the request.
    const OmniSpeechStreamerVariant& speech_streamer;
    /// @brief Optional, started by the caller and ended by the hook in the step the request ends. Its callback runs on
    /// its own thread, so a mid-answer stop or cancel lands a step later and keeps the slots generated meanwhile.
    const std::shared_ptr<ThreadedStreamerWrapper> text_streamer;
    /// @brief Set once the request finished, stopped or was cancelled.
    std::optional<LFM3SpeechResult> result;
};

/// @brief LFM3-Omni speech on continuous batching: the sampler draws text tokens, the hook makes speech frames
/// with the depth decoder and feeds them back as <|audio_out|>. Requests without an LFM3SpeechPayload are untouched.
class LFM3SpeechStepHook : public IStepHook {
public:
    /// @param embedding The language model's token embedding, for the prompt mode marker.
    LFM3SpeechStepHook(std::unique_ptr<LFM3SpeechModels> models, EmbeddingsModel::Ptr embedding);

    /// @brief Speech settings of the model.
    const LFM3SpeechModelConfig& config() const {
        return m_models->config();
    }

    std::unique_ptr<StepRequestState> before_add_request(ov::Tensor& inputs_embeds,
                                                         std::optional<ov::Tensor>& prompt_ids,
                                                         SamplingOverrides& sampling_overrides,
                                                         const std::shared_ptr<StepRequestPayload>& payload) override;
    void after_remove_request(StepRequestState& state, GenerationStatus status) override;
    std::vector<RequestOutcome> after_sampling(const std::vector<SequenceGroup::Ptr>& requests) override;

private:
    struct RequestState;

    static RequestState& as_request_state(StepRequestState& state);
    static RequestState& as_request_state(const SequenceGroup& request);

    /// @brief Records the slot of request sampled this step, and ends the request once its speech state or the
    /// sampler ended it. A speech model or streamer exception fails only this request.
    RequestOutcome after_slot(const SequenceGroup& request);

    /// @brief Records a speech frame sampled from the last hidden state of sequence.
    /// @return The codes of the frame, or nothing when a text streamer status already ended the request.
    std::optional<std::vector<int64_t>> accept_frame_slot(RequestState& state, const Sequence& sequence);

    /// @brief Records the text token the sampler drew this step.
    void accept_text_slot(RequestState& state, const Sequence& sequence);

    std::unique_ptr<LFM3SpeechModels> m_models;
    EmbeddingsModel::Ptr m_embedding;
};

}  // namespace ov::genai
