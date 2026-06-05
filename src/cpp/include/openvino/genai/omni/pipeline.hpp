// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/omni/decoded_results.hpp"
#include "openvino/genai/omni/speech_generation_config.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/pipeline_base.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * @brief Public Qwen3-Omni pipeline supporting text + speech output.
 *
 * OmniPipeline composes a VLM pipeline (text generation with hidden-state collection)
 * with a Qwen3-Omni speech pipeline (Talker + CodePredictor + Code2Wav). Speech generation
 * is gated per-call by OmniSpeechGenerationConfig::return_audio.
 *
 * Constructors:
 *   - Path-based: loads VLM and speech models from a single models_path.
 *   - Shared-VLM: reuses an externally-loaded VLM (pass VLMPipeline::get_base()).
 *
 * Both ctors enforce that the loaded model is Qwen3-Omni capable (model_type == QWEN3_OMNI
 * and enable_audio_output) — non-Omni models throw at construction time.
 */
class OPENVINO_GENAI_EXPORTS OmniPipeline {
public:
    /// @brief Owning ctor: loads VLM and speech models from a single models_path.
    OmniPipeline(const std::filesystem::path& models_path,
                 const std::string& device,
                 const ov::AnyMap& properties = {});

    /// @brief Shared-VLM ctor: reuses an externally-loaded VLM. Useful when an app already
    /// owns a VLMPipeline for text-only inference and wants to add speech without re-loading
    /// multi-GB weights. Get the base from VLMPipeline::get_base().
    OmniPipeline(const std::shared_ptr<VLMPipeline::VLMPipelineBase>& vlm,
                 const std::filesystem::path& speech_models_path,
                 const std::string& device,
                 const ov::AnyMap& properties = {});

    ~OmniPipeline();
    OmniPipeline(OmniPipeline&&) noexcept;
    OmniPipeline& operator=(OmniPipeline&&) noexcept;
    OmniPipeline(const OmniPipeline&) = delete;
    OmniPipeline& operator=(const OmniPipeline&) = delete;

    /// @brief Generate text + (optionally) speech from a flat prompt.
    /// @param prompt The user prompt.
    /// @param images Image tensors to be prepended to the prompt.
    /// @param videos Video tensors to be prepended to the prompt.
    /// @param audios Audio tensors to be prepended to the prompt.
    /// @param speech_config Generation config (inherits GenerationConfig fields plus the three Omni fields).
    /// @param text_streamer Optional streamer for text tokens.
    /// @param speech_streamer Optional streamer for audio chunks.
    /// @return OmniDecodedResults with `speech_outputs` populated when speech_config.return_audio is true.
    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<ov::Tensor>& audios,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Generate text + (optionally) speech from a chat history.
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<ov::Tensor>& audios,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

private:
    class OmniPipelineImpl;
    std::unique_ptr<OmniPipelineImpl> m_pimpl;
};

}  // namespace ov::genai
