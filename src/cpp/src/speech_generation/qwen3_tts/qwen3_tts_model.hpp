// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
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
                                       const SpeechGenerationConfig& generation_config) override;

    ov::Shape get_speaker_embedding_shape() const override;

private:
    ov::Tensor infer_talker(const ov::Tensor& inputs_embeds,
                            const ov::Tensor& attention_mask,
                            const ov::Tensor& position_ids,
                            bool reset_state);

    Text2SpeechDecodedResults generate_voice_clone(const std::string& text,
                                                   const Qwen3VoiceClonePrompt& prompt,
                                                   const SpeechGenerationConfig& generation_config);

    Text2SpeechDecodedResults decode_from_prefill(const ov::Tensor& talker_prefill,
                                                  const ov::Tensor& tts_pad,
                                                  const ov::Tensor& trailing_text_hidden,
                                                  const SpeechGenerationConfig& generation_config,
                                                  const std::vector<bool>& suppress_tokens,
                                                  const ov::Tensor& ref_codec_ids = ov::Tensor{});

    Qwen3TTSCodecEngine m_engine;
    ov::InferRequest m_talker;

    mutable int64_t m_talker_prefill_tokens = 0;
};

}  // namespace genai
}  // namespace ov
