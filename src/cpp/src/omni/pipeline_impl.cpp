// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "omni/pipeline_impl.hpp"

#include <cstring>
#include <utility>

#include "openvino/core/except.hpp"
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

OmniPipeline::OmniPipelineImpl::OmniPipelineImpl(const std::filesystem::path& models_path,
                                                  const std::string& device,
                                                  const ov::AnyMap& properties) {
    m_vlm = VLMPipeline::create_base(models_path, device, properties);
    assert_omni_capable();

    VLMConfig speech_vlm_config(models_path / "config.json");
    m_speech = std::make_unique<Qwen3OmniSpeechPipeline>(models_path, speech_vlm_config, device, properties);
}

OmniPipeline::OmniPipelineImpl::OmniPipelineImpl(const std::shared_ptr<VLMPipeline::VLMPipelineBase>& vlm,
                                                  const std::filesystem::path& speech_models_path,
                                                  const std::string& device,
                                                  const ov::AnyMap& properties) {
    OPENVINO_ASSERT(vlm != nullptr, "OmniPipeline: shared VLM pointer is null");
    m_vlm = vlm;
    assert_omni_capable();

    VLMConfig speech_vlm_config(speech_models_path / "config.json");
    m_speech =
        std::make_unique<Qwen3OmniSpeechPipeline>(speech_models_path, speech_vlm_config, device, properties);
}

void OmniPipeline::OmniPipelineImpl::assert_omni_capable() const {
    OPENVINO_ASSERT(
        m_vlm->is_audio_output_enabled(),
        "OmniPipeline requires a Qwen3-Omni model with audio output enabled (config.json: enable_audio_output=true)");
    OPENVINO_ASSERT(
        m_vlm->supports_hidden_states_collection(),
        "OmniPipeline speech output requires the continuous-batching backend, but the loaded VLM uses the SDPA "
        "fallback path. Load the model with attention_backend=PA on a CPU or GPU device (NPU is not supported "
        "for Qwen3-Omni speech output).");
}

OmniDecodedResults OmniPipeline::OmniPipelineImpl::generate(const std::string& prompt,
                                                             const std::vector<ov::Tensor>& images,
                                                             const std::vector<ov::Tensor>& videos,
                                                             const std::vector<ov::Tensor>& audios,
                                                             const OmniSpeechGenerationConfig& speech_config,
                                                             const StreamerVariant& text_streamer,
                                                             const OmniSpeechStreamerVariant& speech_streamer) {
    speech_config.validate();

    PendingAudiosGuard audios_guard(*m_vlm, audios);
    GenerationConfig text_cfg = static_cast<GenerationConfig>(speech_config);

    if (speech_config.return_audio) {
        VLMPipeline::VLMPipelineBase::HiddenStatesCollectionScope hs_scope(*m_vlm);
        VLMDecodedResults vlm_result =
            m_vlm->generate(prompt, images, videos, /*videos_metadata=*/{}, text_cfg, text_streamer);
        return run_with_speech(std::move(vlm_result), speech_config, speech_streamer);
    }

    // Text-only path: convert VLMDecodedResults to OmniDecodedResults with empty speech_outputs.
    VLMDecodedResults vlm_result = m_vlm->generate(prompt, images, videos, /*videos_metadata=*/{}, text_cfg, text_streamer);
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    return omni_result;
}

OmniDecodedResults OmniPipeline::OmniPipelineImpl::generate(const ChatHistory& history,
                                                             const std::vector<ov::Tensor>& images,
                                                             const std::vector<ov::Tensor>& videos,
                                                             const std::vector<ov::Tensor>& audios,
                                                             const OmniSpeechGenerationConfig& speech_config,
                                                             const StreamerVariant& text_streamer,
                                                             const OmniSpeechStreamerVariant& speech_streamer) {
    speech_config.validate();

    PendingAudiosGuard audios_guard(*m_vlm, audios);
    GenerationConfig text_cfg = static_cast<GenerationConfig>(speech_config);

    if (speech_config.return_audio) {
        // Speech output requires the prompts overload's per-prompt loop, which captures the
        // prompt-id slice into VLMDecodedResults::m_hidden_states_data. The histories overload
        // doesn't yet wire that capture (CB pipeline_base.cpp leaves original_prompt_ids_list
        // empty for the ChatHistory path — see the FIXME there). Apply the chat template here
        // and route through the prompts path so speech generation has its prompt_ids.
        const std::string templated_prompt = m_vlm->get_tokenizer().apply_chat_template(history, true);
        text_cfg.apply_chat_template = false;
        VLMPipeline::VLMPipelineBase::HiddenStatesCollectionScope hs_scope(*m_vlm);
        VLMDecodedResults vlm_result =
            m_vlm->generate(templated_prompt, images, videos, /*videos_metadata=*/{}, text_cfg, text_streamer);
        return run_with_speech(std::move(vlm_result), speech_config, speech_streamer);
    }

    // Text-only path: convert VLMDecodedResults to OmniDecodedResults with empty speech_outputs.
    VLMDecodedResults vlm_result = m_vlm->generate(history, images, videos, /*videos_metadata=*/{}, text_cfg, text_streamer);
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    return omni_result;
}

OmniDecodedResults OmniPipeline::OmniPipelineImpl::run_with_speech(VLMDecodedResults vlm_result,
                                                                    const OmniSpeechGenerationConfig& speech_config,
                                                                    const OmniSpeechStreamerVariant& speech_streamer) {
    OPENVINO_ASSERT(vlm_result.m_hidden_states_data != nullptr,
                    "OmniPipeline: hidden states missing despite return_audio=true; check that the model exposes "
                    "thinker hidden states");

    const auto& hs_data = *vlm_result.m_hidden_states_data;
    OPENVINO_ASSERT(!hs_data.hidden_states.empty(),
                    "OmniPipeline: collected hidden states are empty; speech pipeline cannot run");

    auto all_hidden_states = flatten_hidden_states(hs_data.hidden_states[0]);
    auto all_intermediate_hidden_states = hs_data.intermediate_hidden_states.empty()
                                              ? std::vector<ov::Tensor>{}
                                              : flatten_hidden_states(hs_data.intermediate_hidden_states[0]);

    // The user's speech_config.max_new_tokens (inherited from GenerationConfig, default
    // SIZE_MAX) is forwarded to the talker; Qwen3OmniSpeechPipeline takes
    // min(max_new_tokens, m_config.talker_max_new_tokens) internally so the checkpoint's
    // talker_max_new_tokens (loaded from generation_config.json) always caps the result.
    // The thinker's max_new_tokens drove the VLM call above and is independent.
    ov::Tensor waveform = m_speech->generate_speech(hs_data.prompt_ids,
                                                    all_hidden_states,
                                                    all_intermediate_hidden_states,
                                                    speech_streamer,
                                                    speech_config.audio_chunk_frames,
                                                    speech_config.speaker,
                                                    speech_config.max_new_tokens,
                                                    speech_config.rng_seed);

    // Convert VLMDecodedResults to OmniDecodedResults and populate speech_outputs.
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    omni_result.speech_outputs = {std::move(waveform)};
    return omni_result;
}

}  // namespace ov::genai
