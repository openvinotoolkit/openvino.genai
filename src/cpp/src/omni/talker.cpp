// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/talker.hpp"

#include <cstring>

#include "openvino/core/except.hpp"
#include "visual_language/qwen3_omni/speech_pipeline.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

namespace {

/// @brief Flatten per-step hidden states into per-token hidden states.
/// Each step tensor has shape [num_tokens, 1, hidden_size]. Multi-token tensors
/// (from prefill) are split into individual [1, 1, hidden_size] slices.
std::vector<ov::Tensor> flatten_hidden_states(const std::vector<ov::Tensor>& per_step_hs) {
    std::vector<ov::Tensor> per_token;
    size_t total_tokens = 0;
    for (const auto& step_hs : per_step_hs) {
        total_tokens += step_hs.get_shape()[0];
    }
    per_token.reserve(total_tokens);
    for (const auto& step_hs : per_step_hs) {
        const auto shape = step_hs.get_shape();
        const size_t num_tokens = shape[0];
        const size_t hidden_size = shape.back();
        const size_t elem_size = step_hs.get_element_type().size();
        const auto* src = static_cast<const uint8_t*>(step_hs.data());
        const size_t token_bytes = hidden_size * elem_size;
        const size_t stride = (shape.size() == 3) ? shape[1] * hidden_size * elem_size : token_bytes;
        for (size_t t = 0; t < num_tokens; t++) {
            ov::Tensor token_hs(step_hs.get_element_type(), {1, 1, hidden_size});
            std::memcpy(token_hs.data(), src + t * stride, token_bytes);
            per_token.push_back(std::move(token_hs));
        }
    }
    return per_token;
}

}  // namespace

class Qwen3OmniTalker::Impl {
public:
    Impl(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap& properties) {
        VLMConfig vlm_config(model_dir / "config.json");
        m_speech = std::make_unique<Qwen3OmniSpeechPipeline>(model_dir, vlm_config, device, properties);
    }

    OmniDecodedResults generate(VLMDecodedResults vlm_result,
                                const OmniSpeechGenerationConfig& speech_config,
                                const OmniSpeechStreamerVariant& speech_streamer) {
        OPENVINO_ASSERT(!vlm_result.hidden_states.empty(),
                        "Qwen3OmniTalker: hidden states missing on VLMDecodedResults; the VLM must run with "
                        "speech_config.return_audio=true so it accumulates thinker hidden states.");

        auto all_hidden_states = flatten_hidden_states(vlm_result.hidden_states[0]);
        auto all_intermediate_hidden_states = vlm_result.intermediate_hidden_states.empty()
                                                  ? std::vector<ov::Tensor>{}
                                                  : flatten_hidden_states(vlm_result.intermediate_hidden_states[0]);

        // speech_config carries audio_chunk_frames, speaker / speaker_embedding,
        // max_new_tokens, rng_seed, plus the talker_* / cp_* sampling overrides. The pipeline
        // resolves each std::optional<...> against the JSON-loaded checkpoint defaults at the
        // top of generate_speech(); unset overrides keep the checkpoint values.
        ov::Tensor waveform = m_speech->generate_speech(vlm_result.prompt_ids,
                                                        all_hidden_states,
                                                        all_intermediate_hidden_states,
                                                        speech_streamer,
                                                        speech_config);

        OmniDecodedResults omni_result;
        static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
        if (waveform && waveform.get_size() > 0) {
            omni_result.speech_outputs = {std::move(waveform)};
        }
        return omni_result;
    }

    std::vector<std::string> list_speakers() const {
        return m_speech->list_speakers();
    }

    ov::Tensor get_speaker_embedding(const std::string& name) const {
        return m_speech->get_speaker_embedding(name);
    }

    bool is_available() const {
        return m_speech && m_speech->is_available();
    }

private:
    std::unique_ptr<Qwen3OmniSpeechPipeline> m_speech;
};

Qwen3OmniTalker::Qwen3OmniTalker(const std::filesystem::path& model_dir,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(model_dir, device, properties)) {}

Qwen3OmniTalker::~Qwen3OmniTalker() = default;
Qwen3OmniTalker::Qwen3OmniTalker(Qwen3OmniTalker&&) noexcept = default;
Qwen3OmniTalker& Qwen3OmniTalker::operator=(Qwen3OmniTalker&&) noexcept = default;

OmniDecodedResults Qwen3OmniTalker::generate(VLMDecodedResults vlm_result,
                                              const OmniSpeechGenerationConfig& speech_config,
                                              const OmniSpeechStreamerVariant& speech_streamer) {
    return m_impl->generate(std::move(vlm_result), speech_config, speech_streamer);
}

std::vector<std::string> Qwen3OmniTalker::list_speakers() const {
    return m_impl->list_speakers();
}

ov::Tensor Qwen3OmniTalker::get_speaker_embedding(const std::string& name) const {
    return m_impl->get_speaker_embedding(name);
}

bool Qwen3OmniTalker::is_available() const {
    return m_impl->is_available();
}

}  // namespace ov::genai
