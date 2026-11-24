// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "speech_generation/qwen3_tts/qwen3_tts_model.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_text_committer.hpp"
#include "speech_generation/text2speech_streaming_session_impl.hpp"

namespace ov {
namespace genai {

// Text2SpeechStreamingSession for the SDPA Qwen3-TTS backend: text that arrives in pieces drives one Qwen3TTSImpl::
// TalkerRun in the streaming prompt layout.
//
//   push() / end()  buffer the text under a mutex.
//   read()          commits the text that is safe to tokenize (TextCommitter), embeds it, and as soon as there is
//                   enough (one token for CustomVoice / VoiceDesign / Base x-vector; enough rows to cover the
//                   reference for ICL) builds the prefill and starts the run. From then on every committed text
//                   token is one more row of the run's text, the text of the frames to come. The run is stepped until
//                   a chunk of audio is ready, the next frame needs text that has not arrived (the decoded audio is
//                   returned, if any), or the utterance is complete.
class Qwen3TTSStreamingSessionImpl : public Text2SpeechStreamingSessionImpl {
public:
    Qwen3TTSStreamingSessionImpl(std::shared_ptr<Qwen3TTSImpl> owner,
                                 Text2SpeechStreamingSessionSlot slot,
                                 const ov::Tensor& speaker_embedding,
                                 const SpeechGenerationConfig& generation_config,
                                 const Text2SpeechStreamingConfig& streaming_config);

    void push(const std::string& text) override;
    void end() override;
    std::optional<Text2SpeechStreamingResult> read() override;

private:
    using TimePoint = std::chrono::steady_clock::time_point;

    std::vector<int64_t> encode(const std::string& text);
    ov::Tensor rows_tensor(size_t first, size_t count) const;
    void absorb_committed_text();
    bool start_run();
    void feed_text();
    Text2SpeechStreamingResult make_result(std::vector<float> samples, bool is_final, TimePoint start);

    // The members are destroyed in reverse order: the run (which uses the owner's talker) first, then the slot, then
    // the owner.
    std::shared_ptr<Qwen3TTSImpl> m_owner;
    Text2SpeechStreamingSessionSlot m_slot;
    Qwen3TTSCodecEngine& m_engine;
    SpeechGenerationConfig m_config;
    Qwen3TTSImpl::VoiceSetup m_voice;
    std::vector<bool> m_suppress_tokens;
    size_t m_hidden = 0;

    // Input side: push() / end() from the producer thread.
    std::mutex m_input_mutex;
    std::string m_incoming;
    bool m_end_requested = false;

    // Consumer side: everything below is used under m_read_mutex only.
    std::mutex m_read_mutex;
    qwen3_tts::TextCommitter m_committer;
    std::vector<float> m_rows;  // embeddings of all committed target text tokens, [rows, H]
    size_t m_num_rows = 0;
    size_t m_rows_given = 0;    // rows that went into the prefill or the run's text
    std::optional<Qwen3TTSCodecEngine::PrefillHead> m_head;  // CustomVoice / VoiceDesign / Base x-vector
    std::optional<Qwen3TTSCodecEngine::IclContext> m_icl;    // Base ICL
    std::optional<Qwen3TTSImpl::TalkerRun> m_run;
    bool m_text_closed = false;  // the run's text is complete
    size_t m_samples_returned = 0;
    bool m_final_returned = false;
};

}  // namespace genai
}  // namespace ov
