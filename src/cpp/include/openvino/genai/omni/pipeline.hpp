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
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/omni/decoded_results.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * @brief Public Qwen3-Omni pipeline supporting text + speech outputs.
 *
 * OmniPipeline composes a VLM pipeline (text generation with hidden-state collection)
 * with a Qwen3-Omni speech pipeline (Talker + CodePredictor + Code2Wav). Each call to
 * `generate` takes two configs:
 *
 *   - `GenerationConfig text_config`     — drives the thinker text decode
 *     (max_new_tokens, do_sample, temperature, top_p, top_k, repetition_penalty,
 *      num_beams, eos_token_id, ...).
 *   - `OmniTalkerSpeechConfig talker_speech_config` — drives the talker + speech output
 *     (return_audio, speaker (name or embedding tensor), audio_chunk_frames, max_new_tokens,
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
 *
 * @note This is a preview API and is subject to change.
 */
class OPENVINO_GENAI_EXPORTS OmniPipeline {
public:
    /// @brief Sugar ctor: loads the VLM and the default Qwen3-Omni talker from a single
    /// `models_path`, both compiled to `device` with the same `properties`. Equivalent to
    /// constructing a `VLMPipeline(models_path, device, properties)` plus a
    /// `Talker(models_path, device, properties)` and passing them to the DI ctor.
    OmniPipeline(const std::filesystem::path& models_path,
                 const std::string& device,
                 const ov::AnyMap& properties = {});

    /// @brief Pure-DI ctor: takes pre-built VLM and talker components. Use when you want
    /// independent device/property choices for the two stages (e.g. VLM on GPU, talker on
    /// CPU), or to inject a custom `TalkerBase` subclass. The VLM must be Qwen3-Omni-capable
    /// (audio output enabled) and use the continuous-batching backend; the ctor asserts on both.
    /// @param vlm Backing VLM pipeline. Shared ownership allows independent lifetime management.
    /// @param talker Backing speech generator. The default impl is `Talker`.
    OmniPipeline(const std::shared_ptr<VLMPipelineBase>& vlm, const std::shared_ptr<TalkerBase>& talker);

    ~OmniPipeline();

    /// @brief Generate text + (optionally) speech from a flat prompt.
    /// @param prompt The user prompt.
    /// @param images Image tensors to be prepended to the prompt.
    /// @param videos Video tensors to be prepended to the prompt.
    /// @param audios Audio tensors to be prepended to the prompt.
    /// @param text_generation_config Thinker text-decode config.
    /// @param talker_speech_config Talker + speech-output config.
    /// @param streamer Optional streamer for text tokens.
    /// @param speech_streamer Optional streamer for audio chunks.
    /// @return OmniDecodedResults with `speech_result.waveforms` populated when
    ///         `talker_speech_config.return_audio` is true.
    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<VideoMetadata>& videos_metadata,
                                const std::vector<ov::Tensor>& audios,
                                const GenerationConfig& text_generation_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images-only flat-prompt overload (no videos / no audios).
    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const GenerationConfig& text_generation_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images + videos flat-prompt overload (no audios).
    OmniDecodedResults generate(const std::string& prompt,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<VideoMetadata>& videos_metadata,
                                const GenerationConfig& text_generation_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Single-image flat-prompt overload.
    OmniDecodedResults generate(const std::string& prompt,
                                const ov::Tensor& image,
                                const GenerationConfig& text_generation_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Property-bag overload for flat-prompt generation. Recognized keys:
    ///        `images`, `videos`, `videos_metadata`, `audios`, `text_config`,
    ///        `talker_speech_config`, `streamer`, `speech_streamer`, plus any field of
    ///        `GenerationConfig` or `OmniTalkerSpeechConfig`.
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
                                const std::vector<VideoMetadata>& videos_metadata,
                                const std::vector<ov::Tensor>& audios,
                                const GenerationConfig& text_generation_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images-only chat-history overload (no videos / no audios).
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const GenerationConfig& text_generation_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Images + videos chat-history overload (no audios).
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<VideoMetadata>& videos_metadata,
                                const GenerationConfig& text_generation_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Single-image chat-history overload.
    OmniDecodedResults generate(const ChatHistory& history,
                                const ov::Tensor& image,
                                const GenerationConfig& text_generation_config,
                                const OmniTalkerSpeechConfig& talker_speech_config,
                                const StreamerVariant& streamer = std::monostate{},
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{});

    /// @brief Property-bag overload for chat-history generation. Recognized keys mirror the prompt overload.
    OmniDecodedResults generate(const ChatHistory& history, const ov::AnyMap& config_map);

    /// @brief Variadic-properties chat-history overload.
    template <typename... Properties>
    util::EnableIfAllStringAny<OmniDecodedResults, Properties...> generate(const ChatHistory& history,
                                                                           Properties&&... properties) {
        return generate(history, AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief Return the underlying VLM. Useful for inspecting model metadata or reusing the
    /// same thinker across pipelines.
    std::shared_ptr<VLMPipelineBase> get_vlm() const;

    /// @brief Return the underlying TalkerBase. Speaker enumeration and embedding retrieval
    /// live here: `get_talker()->list_speakers()`, `get_talker()->get_speaker_embedding(name)`.
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
