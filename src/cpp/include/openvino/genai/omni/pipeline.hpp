// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"  // util::EnableIfAllStringAny
#include "openvino/genai/omni/decoded_results.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
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
 * with a Qwen3-Omni speech pipeline (Talker + CodePredictor + Code2Wav). Each call to
 * `generate` takes two configs:
 *
 *   - `GenerationConfig text_config`     — drives the thinker text decode
 *     (max_new_tokens, do_sample, temperature, top_p, top_k, repetition_penalty,
 *      num_beams, eos_token_id, ...).
 *   - `OmniTalkerSpeechConfig talker_speech_config` — drives the talker + speech output
 *     (return_audio, speaker, speaker_embedding, audio_chunk_frames, max_new_tokens,
 *      rng_seed, talker_x / cp_x sampling overrides). Speech is gated per-call by
 *      `talker_speech_config.return_audio`.
 *
 * Cross-config validation (e.g. return_audio incompatible with beam search) runs inside
 * generate() before any work begins.
 *
 * Constructors:
 *   - Path-based: loads VLM and speech models from a single models_path.
 *   - Pure-DI: takes pre-built VLM and talker components for independent device choices
 *              or a custom TalkerBase subclass.
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
    /// @param text_config Thinker text-decode config.
    /// @param talker_speech_config Talker + speech-output config.
    /// @param text_streamer Optional streamer for text tokens.
    /// @param speech_streamer Optional streamer for audio chunks.
    /// @return OmniDecodedResults with `speech_outputs` populated when
    ///         `talker_speech_config.return_audio` is true.
    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<ov::Tensor>& audios,
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images-only flat-prompt overload (no videos / no audios).
    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images + videos flat-prompt overload (no audios).
    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Single-image flat-prompt overload.
    OmniDecodedResults generate(const std::string& prompt,
                                const ov::Tensor& image,
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Property-bag overload for flat-prompt generation. Recognized keys:
    ///        `images`, `videos`, `audios`, `text_config`, `talker_speech_config`, `streamer`,
    ///        `speech_streamer`, plus any field of `GenerationConfig` or `OmniTalkerSpeechConfig`.
    /// @note Leftover keys are forwarded to BOTH configs' `update_generation_config`. Shared
    ///       field names (e.g. `max_new_tokens`, `rng_seed`) end up on both — this is the
    ///       intended broadcast behavior. Pass typed configs explicitly to set different values.
    OmniDecodedResults generate(const std::string& prompt, const ov::AnyMap& config_map);

    /// @brief Variadic-properties flat-prompt overload.
    /// Example:
    ///   `pipe.generate("text", ov::genai::images(rgbs), ov::genai::audios(audios),
    ///                   ov::genai::text_config(g), ov::genai::talker_speech_config(t));`
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
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images-only chat-history overload (no videos / no audios).
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images + videos chat-history overload (no audios).
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& text_streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Single-image chat-history overload.
    OmniDecodedResults generate(const ChatHistory& history,
                                const ov::Tensor& image,
                                const GenerationConfig& text_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
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
    /// result via `OmniTalkerSpeechConfig::speaker_embedding`.
    /// @throws Exception when the model exposes no named speakers, or `name` doesn't match.
    ov::Tensor get_speaker_embedding(const std::string& name) const;

    /// @brief List names of speakers available in the loaded model's `talker_config.speaker_id`.
    /// Returns an empty vector when the model exposes no named speakers.
    std::vector<std::string> list_speakers() const;

    /// @brief Return the VLM's loaded GenerationConfig (parsed from generation_config.json).
    /// Used as the default `text_config` when callers don't pass one explicitly.
    GenerationConfig get_text_generation_config() const;

    /// @brief Replace the VLM's GenerationConfig. Subsequent generate() calls that don't pass an
    /// explicit `text_config` will use the new value.
    void set_text_generation_config(const GenerationConfig& new_config);

    /// @brief Return the pipeline's default OmniTalkerSpeechConfig. The path-based ctor seeds it
    /// from `<models_path>/config.json -> talker_config.speaker_id`; the DI ctor leaves it
    /// default-constructed. Used as the default when generate() is called without an explicit
    /// `talker_speech_config`.
    OmniTalkerSpeechConfig get_talker_speech_config() const;

    /// @brief Replace the pipeline's default OmniTalkerSpeechConfig. The new value is `validate()`-d
    /// before being stored so a misconfigured config can never silently take effect on a later call.
    void set_talker_speech_config(const OmniTalkerSpeechConfig& new_config);

    /// @brief Return the underlying VLM base (the same instance that backs the path-based ctor or
    /// was passed to the DI ctor). Useful for inspecting model metadata or sharing the loaded VLM
    /// with another pipeline without reloading multi-GB weights.
    std::shared_ptr<VLMPipeline::VLMPipelineBase> get_vlm() const;

    /// @brief Return the underlying TalkerBase. Useful for direct talker introspection (e.g.
    /// `is_available`, `list_speakers` on a custom subclass) when the OmniPipeline-level helpers
    /// don't expose what the caller needs.
    std::shared_ptr<TalkerBase> get_talker() const;

private:
    class OmniPipelineImpl;
    std::unique_ptr<OmniPipelineImpl> m_pimpl;
};

/// @brief Property bag keys for OmniPipeline::generate(prompt|history, AnyMap).
/// `images`, `videos`, `audios` are reused from `openvino/genai/visual_language/pipeline.hpp`.
static constexpr ov::Property<GenerationConfig> text_config{"text_config"};
static constexpr ov::Property<OmniTalkerSpeechConfig> talker_speech_config{"talker_speech_config"};
static constexpr ov::Property<OmniSpeechStreamerVariant> speech_streamer{"speech_streamer"};

}  // namespace ov::genai
