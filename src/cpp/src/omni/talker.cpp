// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/talker.hpp"

#include "openvino/core/except.hpp"
#include "utils.hpp"
#include "visual_language/qwen3_omni/speech_pipeline.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

namespace {

/// @brief Flatten per-step hidden states into per-token hidden states.
/// Each step tensor has shape [num_tokens, 1, hidden_size]. Multi-token tensors
/// (from prefill) are split into individual [1, 1, hidden_size] slices.
std::vector<ov::Tensor> split_hidden_states(const std::vector<ov::Tensor>& per_step_hidden_states) {
    std::vector<ov::Tensor> per_token;
    size_t total_tokens = 0;
    for (const auto& step_hs : per_step_hidden_states) {
        total_tokens += step_hs.get_shape()[0];
    }
    per_token.reserve(total_tokens);
    for (const auto& step_hs : per_step_hidden_states) {
        const auto shape = step_hs.get_shape();
        const size_t num_tokens = shape[0];
        for (size_t t = 0; t < num_tokens; t++) {
            const auto [begin, end] = ov::genai::utils::make_roi(shape, 0, t, t + 1);
            ov::Tensor token_hs(step_hs, begin, end);
            per_token.push_back(std::move(token_hs));
        }
    }
    return per_token;
}

}  // namespace

class Talker::Impl {
public:
    Impl(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap& properties) {
        VLMConfig vlm_config(model_dir / "config.json");
        m_speech = std::make_unique<Qwen3OmniSpeechPipeline>(model_dir, vlm_config, device, properties);

        OPENVINO_ASSERT(m_speech->is_available(),
                        "Talker: speech generation submodels are missing from ", model_dir.string(),
                        ". If your model has no speech generation capability, use VLMPipeline directly instead of OmniPipeline.");
    }

    TalkerResults generate(const VLMDecodedResults& vlm_result,
                          const OmniTalkerSpeechConfig& talker_speech_config,
                          const OmniSpeechStreamerVariant& speech_streamer) {
        OPENVINO_ASSERT(!vlm_result.intermediate_hidden_states.empty(),
                        "Talker: intermediate hidden states missing on VLMDecodedResults; the VLM must run "
                        "with talker_speech_config.return_audio=true so it accumulates thinker hidden states.");
        OPENVINO_ASSERT(!vlm_result.full_token_ids.empty(),
                        "Talker: full_token_ids missing on VLMDecodedResults; the VLM must run "
                        "with talker_speech_config.return_audio=true.");
        OPENVINO_ASSERT(vlm_result.intermediate_hidden_states.size() == 1 && vlm_result.full_token_ids.size() == 1,
                        "Talker: expected a single return sequence, got ",
                        vlm_result.full_token_ids.size(),
                        "; speech generation consumes one hidden-state stream.");

        auto all_intermediate_hidden_states = split_hidden_states(vlm_result.intermediate_hidden_states[0]);

        // talker_speech_config carries audio_chunk_frames, speaker / speaker_embedding,
        // max_new_tokens, rng_seed, plus the talker_* / cp_* sampling overrides. The pipeline
        // resolves each std::optional<...> against the JSON-loaded checkpoint defaults at the
        // top of generate_speech(); unset overrides keep the checkpoint values.
        return m_speech->generate_speech(vlm_result.full_token_ids[0],
                                         all_intermediate_hidden_states,
                                         speech_streamer,
                                         talker_speech_config);
    }

    std::vector<std::string> list_speakers() const {
        return m_speech->list_speakers();
    }

    ov::Tensor get_speaker_embedding(const std::string& name) const {
        return m_speech->get_speaker_embedding(name);
    }

private:
    std::unique_ptr<Qwen3OmniSpeechPipeline> m_speech;
};

Talker::Talker(const std::filesystem::path& model_dir,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(model_dir, device, properties)) {}

Talker::~Talker() = default;

TalkerResults Talker::generate(const VLMDecodedResults& vlm_result,
                                        const OmniTalkerSpeechConfig& talker_speech_config,
                                        const OmniSpeechStreamerVariant& speech_streamer) {
    return m_impl->generate(vlm_result, talker_speech_config, speech_streamer);
}

std::vector<std::string> Talker::list_speakers() const {
    return m_impl->list_speakers();
}

ov::Tensor Talker::get_speaker_embedding(const std::string& name) const {
    return m_impl->get_speaker_embedding(name);
}

}  // namespace ov::genai
