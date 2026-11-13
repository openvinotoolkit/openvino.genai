// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "continuous_batching/step_hook.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "visual_language/lfm3_omni/speech_core.hpp"

namespace ov::genai {

/// @brief LFM3-Omni speech output on continuous batching. The hook picks every slot of a speech request: text tokens
/// with the LFM3-Omni sampler, and speech frames with the depth decoder, fed back as <|audio_out|> with the frame
/// embedding. Requests that were not announced with expect_request() are left to the regular sampler.
class LFM3SpeechStepHook : public IStepHook {
public:
    /// @param make_models Builds the speech models; called on the first speech request.
    explicit LFM3SpeechStepHook(std::function<std::unique_ptr<LFM3SpeechModels>()> make_models);

    /// @brief Announces the speech settings of the next request added to the pipeline.
    /// @param speech_streamer Optional; must outlive the request.
    /// @param text_streamer Optional. Written in the step that samples each text token, so its stop or cancel ends
    /// the speech before the next frame, as on SDPA. The caller ends it.
    void expect_request(const LFM3SpeechRequest& request,
                        const OmniSpeechStreamerVariant& speech_streamer,
                        std::shared_ptr<StreamerBase> text_streamer);

    /// @brief Result of the speech request that ended last; one speech request runs at a time.
    LFM3SpeechResult take_result();

    /// @brief Rethrows an exception of the text or speech streamer. It ended only its request, so the pipeline stays
    /// usable.
    void rethrow_streamer_error();

    /// @brief True if the text streamer threw in the last request, so the caller must not call it again.
    bool text_streamer_failed() const {
        return m_text_streamer_failed;
    }

    /// @brief Forgets an announced request that was never added, e.g. after a failure before add_request().
    void forget_expected_request() {
        m_pending_request.reset();
    }

    /// @brief Speech settings of the model; valid after the first expect_request().
    const LFM3SpeechModelConfig& config() const {
        return m_models->config();
    }

    void on_add_request(uint64_t request_id,
                        ov::Tensor& inputs_embeds,
                        std::optional<ov::Tensor>& prompt_ids,
                        InputsEmbedder& inputs_embedder) override;
    void before_sampling(const std::vector<SequenceGroup::Ptr>& requests, const ov::Tensor& logits) override;
    std::vector<GeneratedEmbedding> after_sampling(const std::vector<SequenceGroup::Ptr>& requests,
                                                   SamplerOutput& sampler_output) override;
    void on_request_freed(const SequenceGroup::Ptr& request) override;
    void on_request_dropped(uint64_t request_id) override;

private:
    struct PendingRequest {
        LFM3SpeechRequest request;
        const OmniSpeechStreamerVariant* speech_streamer;
        std::shared_ptr<StreamerBase> text_streamer;
    };

    struct PendingFrame {
        SequenceGroup::Ptr sequence_group;
        Sequence::Ptr sequence;
        std::vector<int64_t> codes;
    };

    std::function<std::unique_ptr<LFM3SpeechModels>()> m_make_models;
    std::unique_ptr<LFM3SpeechModels> m_models;
    std::optional<PendingRequest> m_pending_request;
    std::map<uint64_t, LFM3SpeechState> m_states;
    std::shared_ptr<StreamerBase> m_text_streamer;
    std::vector<PendingFrame> m_pending_frames;
    std::optional<LFM3SpeechResult> m_result;
    std::exception_ptr m_streamer_error;
    bool m_text_streamer_failed = false;

    /// @brief Flushes the codec for the request of it, keeps the result and forgets the state. An exception of the
    /// speech streamer cancels the request and is kept for rethrow_streamer_error().
    void finish_request(std::map<uint64_t, LFM3SpeechState>::iterator it);
};

}  // namespace ov::genai
