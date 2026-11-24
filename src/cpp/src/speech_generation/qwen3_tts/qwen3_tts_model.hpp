// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_audio_streaming.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_codec_engine.hpp"
#include "speech_generation/text2speech_pipeline_impl.hpp"

namespace ov {
namespace genai {

// SDPA/stateful driver for the Qwen3-TTS talker. Everything except the talker backbone's own
// inference mechanism and the outer per-session decode loop lives in Qwen3TTSCodecEngine, shared
// with the continuous-batching driver (Qwen3TTSContinuousBatchingImpl).
class Qwen3TTSImpl : public Text2SpeechPipelineImpl {
public:
    // Internal voice-clone prompt used by the generate() dispatch path (ICL mode only).
    struct Qwen3VoiceClonePrompt {
        ov::Tensor ref_spk_embedding;
        ov::Tensor ref_code;
        std::string ref_text;
    };
    Qwen3TTSImpl(const std::filesystem::path& models_path,
                 const std::string& device,
                 const ov::AnyMap& properties,
                 const Tokenizer& tokenizer);

    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding,
                                       const SpeechGenerationConfig& generation_config) override {
        return generate(texts, speaker_embedding, generation_config, std::monostate{});
    }

    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding,
                                       const SpeechGenerationConfig& generation_config,
                                       const OmniSpeechStreamerVariant& streamer) override;

    ov::Shape get_speaker_embedding_shape() const override;

    // The generation of ONE utterance on the talker, advanced one codec frame per step(). All the state that lives
    // between two frames (the talker's last logits and hidden states, the sampling RNGs and history, the text term
    // source, the frames and their audio) is held here, so a run can be paused after any frame and continued later.
    // The talker's KV cache lives in the pipeline's talker request: one run at a time, and no other talker call
    // (another generate(), a second run) while a run is unfinished.
    class TalkerRun {
    public:
        enum class Status {
            FRAME,      // a frame was generated
            NEED_TEXT,  // the next frame needs a text row that has not been appended yet (incremental text only)
            DONE,       // the utterance is complete
        };

        // Prefills the talker with `talker_prefill`. `stream_audio` decodes the frames to audio while they arrive
        // (see qwen3_tts::UtteranceFrames); otherwise finish() returns the waveform of the whole utterance.
        //
        // `incremental_text` ignores `trailing_text_hidden` and starts from an empty incremental text that the
        // caller feeds through text() while the run waits (NEED_TEXT). Codec EOS is then not allowed until the
        // text is finished and its last row has been consumed, so the talker cannot end the utterance while text is
        // still to come. With fixed text (the default) EOS is allowed at every step, as before.
        TalkerRun(Qwen3TTSImpl& impl,
                  const ov::Tensor& talker_prefill,
                  const ov::Tensor& tts_pad,
                  const ov::Tensor& trailing_text_hidden,
                  const SpeechGenerationConfig& generation_config,
                  const std::vector<bool>& suppress_tokens,
                  const ov::Tensor& ref_codec_ids,
                  bool stream_audio,
                  bool incremental_text = false);

        // Generates the next codec frame. After FRAME, `chunk` holds the audio of a chunk that became ready (empty
        // if none). The talker is advanced to the following frame lazily, at the start of the next call, so a caller
        // that stops after a frame does not pay for it; that advance is where the frame's text row is needed, and
        // NEED_TEXT leaves the run untouched so it can be retried after append()ing text.
        Status step(std::vector<float>& chunk);

        // The text source, to append rows to when the run was made with `incremental_text`.
        qwen3_tts::TrailingText& text() {
            return m_text;
        }

        // End of the utterance: streaming, the audio not yet handed over; otherwise, the whole waveform.
        std::vector<float> finish() {
            return m_frames.finish();
        }

        bool streaming() const {
            return m_frames.streaming();
        }

    private:
        // Feeds the frame generated last (its codec embeddings and the text term of its step) to the talker.
        void advance();

        Qwen3TTSImpl& m_impl;
        Qwen3TTSCodecEngine& m_engine;
        SpeechGenerationConfig m_config;
        std::vector<bool> m_suppress_tokens;
        std::vector<bool> m_suppress_tokens_and_eos;  // same plus codec EOS; only set when EOS is gated
        bool m_gate_eos = false;
        qwen3_tts::TrailingText m_text;
        qwen3_tts::UtteranceFrames m_frames;
        std::mt19937 m_talker_rng;
        std::mt19937 m_subtalker_rng;
        size_t m_absolute_pos;  // position of the next talker input
        ov::Tensor m_logits;
        ov::Tensor m_hidden_states;
        size_t m_hidden = 0;
        size_t m_max_steps = 0;
        size_t m_step = 0;  // frames generated so far
        std::vector<int64_t> m_generated_main;
        std::vector<int64_t> m_groups;  // codes of the frame generated last
        bool m_pending = false;         // that frame has not been fed to the talker yet
        bool m_done = false;
    };

private:
    ov::Tensor infer_talker(const ov::Tensor& inputs_embeds,
                            const ov::Tensor& attention_mask,
                            const ov::Tensor& position_ids,
                            bool reset_state);

    Text2SpeechDecodedResults generate_voice_clone(const std::string& text,
                                                   const Qwen3VoiceClonePrompt& prompt,
                                                   const SpeechGenerationConfig& generation_config,
                                                   const OmniSpeechStreamerVariant& streamer);

    Text2SpeechDecodedResults decode_from_prefill(const ov::Tensor& talker_prefill,
                                                  const ov::Tensor& tts_pad,
                                                  const ov::Tensor& trailing_text_hidden,
                                                  const SpeechGenerationConfig& generation_config,
                                                  const std::vector<bool>& suppress_tokens,
                                                  const OmniSpeechStreamerVariant& streamer,
                                                  const ov::Tensor& ref_codec_ids = ov::Tensor{},
                                                  std::optional<MicroSeconds> tokenization_duration = std::nullopt);

    Qwen3TTSCodecEngine m_engine;
    ov::InferRequest m_talker;

    mutable int64_t m_talker_prefill_tokens = 0;
};

}  // namespace genai
}  // namespace ov
