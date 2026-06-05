// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/llm_pipeline.hpp"  // util::EnableIfAllStringAny
#include "openvino/genai/omni/decoded_results.hpp"
#include "openvino/genai/omni/speech_generation_config.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker.hpp"
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
    /// @brief Sugar ctor: loads the VLM and the default Qwen3-Omni talker from a single
    /// `models_path`, both compiled to `device` with the same `properties`. Equivalent to
    /// constructing a `VLMPipeline::create_base(models_path, device, properties)` plus a
    /// `Qwen3OmniTalker(models_path, device, properties)` and passing them to the DI ctor.
    OmniPipeline(const std::filesystem::path& models_path,
                 const std::string& device,
                 const ov::AnyMap& properties = {});

    /// @brief Pure-DI ctor: takes pre-built VLM and talker components. Use when you want
    /// independent device/property choices for the two stages (e.g. VLM on GPU, talker on
    /// CPU), or to inject a custom `TalkerBase` subclass.
    /// @param vlm Backing VLM. Get from `VLMPipeline::get_base()` on an existing pipeline,
    ///            or build fresh via `VLMPipeline::create_base(...)`.
    /// @param talker Backing speech generator. The default impl is `Qwen3OmniTalker`.
    OmniPipeline(const std::shared_ptr<VLMPipeline::VLMPipelineBase>& vlm,
                 const std::shared_ptr<TalkerBase>& talker);

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

    /// @brief Images-only flat-prompt overload (no videos / no audios).
    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images + videos flat-prompt overload (no audios).
    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Single-image flat-prompt overload.
    OmniDecodedResults generate(const std::string& prompt,
                                const ov::Tensor& image,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Property-bag overload for flat-prompt generation. Recognized keys:
    ///        `images`, `videos`, `audios`, `speech_config`, `streamer`, `speech_streamer`,
    ///        plus any field of `OmniSpeechGenerationConfig`.
    OmniDecodedResults generate(const std::string& prompt, const ov::AnyMap& config_map);

    /// @brief Variadic-properties flat-prompt overload.
    /// Example: `pipe.generate("text", ov::genai::images(rgbs), ov::genai::audios(audios), ov::genai::speech_config(cfg));`
    template <typename... Properties>
    util::EnableIfAllStringAny<OmniDecodedResults, Properties...> generate(const std::string& prompt,
                                                                           Properties&&... properties) {
        return generate(prompt, AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief Generate text + (optionally) speech from a chat history.
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<ov::Tensor>& audios,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images-only chat-history overload (no videos / no audios).
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images + videos chat-history overload (no audios).
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Single-image chat-history overload.
    OmniDecodedResults generate(const ChatHistory& history,
                                const ov::Tensor& image,
                                const OmniSpeechGenerationConfig& speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Property-bag overload for chat-history generation. Recognized keys mirror the prompt overload.
    OmniDecodedResults generate(const ChatHistory& history, const ov::AnyMap& config_map);

    /// @brief Variadic-properties chat-history overload.
    template <typename... Properties>
    util::EnableIfAllStringAny<OmniDecodedResults, Properties...> generate(const ChatHistory& history,
                                                                           Properties&&... properties) {
        return generate(history, AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief Return precomputed talker speaker embedding for the named speaker.
    /// Tensor shape is `[1, 1, talker_hidden_size]`, f32.
    /// Use to blend voices: weight-sum two named-speaker embeddings and pass the
    /// result via `OmniSpeechGenerationConfig::speaker_embedding`.
    /// @throws Exception when the model exposes no named speakers, or `name` doesn't match.
    ov::Tensor get_speaker_embedding(const std::string& name) const;

    /// @brief List names of speakers available in the loaded model's `talker_config.speaker_id`.
    /// Returns an empty vector when the model exposes no named speakers.
    std::vector<std::string> list_speakers() const;

private:
    class OmniPipelineImpl;
    std::unique_ptr<OmniPipelineImpl> m_pimpl;
};

/// @brief Property bag keys for OmniPipeline::generate(prompt|history, AnyMap).
/// `images`, `videos`, `audios` are reused from `openvino/genai/visual_language/pipeline.hpp`.
static constexpr ov::Property<OmniSpeechGenerationConfig> speech_config{"speech_config"};
static constexpr ov::Property<OmniSpeechStreamerVariant> speech_streamer{"speech_streamer"};

}  // namespace ov::genai
