// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS VLMDecodedResults : public DecodedResults{
public:
    VLMPerfMetrics perf_metrics;

    // Hidden-state fields for speech generation (exposed to Python to enable custom Talker impls).
    // Populated by the CB pipeline when `return_omni_outputs` is set on the GenerationConfig.
    // Consumed by Talker / TalkerBase impls to drive Qwen3-Omni speech generation.
    // Both fields are empty (default-constructed) on the text-only path.
    // Preview API: subject to change.
    //
    // Outer index of intermediate_hidden_states is per return sequence;
    // inner is one tensor per generation step.
    std::vector<std::vector<ov::Tensor>> intermediate_hidden_states;
    // Full prompt + generated token ids (talker input construction).
    // Outer index is per return sequence, aligned with intermediate_hidden_states;
    // inner is prompt tokens followed by that sequence's generated tokens.
    std::vector<std::vector<int64_t>> full_token_ids;
};

/**
 * @brief Public abstract interface for VLM-style pipelines.
 *
 * Promoted to a public top-level type so callers can hold a `std::shared_ptr<VLMPipelineBase>`
 * without depending on `VLMPipeline`'s internal pimpl. Carries the user-visible surface —
 * `generate()` overloads, tokenizer/config accessors, chat-template setter — plus the Qwen3-Omni
 * capability queries OmniPipeline needs to compose a `shared_ptr<VLMPipelineBase>`. Backend-only
 * machinery lives on internal sub-classes declared in `src/`-private headers, not here.
 */
class OPENVINO_GENAI_EXPORTS VLMPipelineBase {
public:
    virtual ~VLMPipelineBase() = default;

    /// @brief Generate a response given a prompt and any number of
    /// uint8 RGB images with [NHWC] or [HWC] layout.
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param images Images to be prepended to a prompt.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    /// chat_template will be applied to the prompt, run pipe.set_chat_template(custom_chat_template) to update it.
    /// To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    virtual VLMDecodedResults generate(const std::string& prompt,
                                       const std::vector<ov::Tensor>& images,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) = 0;

    /// @brief Generate a response given a prompt and uint8 RGB image with [NHWC] or [HWC] layout.
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param images Image to be prepended to a prompt.
    /// @param videos Multiple videos, each providing multiple frames, to be prepended to a prompt.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    /// chat_template will be applied to the prompt, run pipe.set_chat_template(custom_chat_template) to update it.
    /// To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    virtual VLMDecodedResults generate(const std::string& prompt,
                                       const std::vector<ov::Tensor>& images,
                                       const std::vector<ov::Tensor>& videos,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) = 0;

    /// @brief Generate a response given a prompt and a single uint8 RGB image with [NHWC] or [HWC] layout.
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param image Image to be prepended to a prompt.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    virtual VLMDecodedResults generate(const std::string& prompt,
                                       const ov::Tensor& image,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) {
        return generate(prompt, std::vector<ov::Tensor>{image}, generation_config, streamer);
    }

    /// @brief Generate a response given a prompt and config.
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param config_map A config may contain GenerationConfig, values
    /// for its members, StreamerVariant, a single image or multiple
    /// images/videos, and audios.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    /// chat_template will be applied to the prompt, run pipe.set_chat_template(custom_chat_template) to update it.
    /// To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    virtual VLMDecodedResults generate(const std::string& prompt, const ov::AnyMap& config_map) = 0;

    /// @brief Generate a response given a chat history and any number of
    /// uint8 RGB images with [NHWC] or [HWC] layout.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param images Images to be associated with the last chat history user message.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    virtual VLMDecodedResults generate(const ChatHistory& history,
                                       const std::vector<ov::Tensor>& images,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) = 0;

    /// @brief Generate a response given a chat history and any number of
    /// uint8 RGB images/videos with [NHWC] layout.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param images Images to be associated with the last chat history user message.
    /// @param videos Videos (each providing multiple frames) to be associated with the last chat history user message.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    virtual VLMDecodedResults generate(const ChatHistory& history,
                                       const std::vector<ov::Tensor>& images,
                                       const std::vector<ov::Tensor>& videos,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) = 0;

    /// @brief Generate a response given a chat history and a single uint8 RGB image with [NHWC] or [HWC] layout.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param image Image to be associated with the last chat history user message.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    virtual VLMDecodedResults generate(const ChatHistory& history,
                                       const ov::Tensor& image,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) {
        return generate(history, std::vector<ov::Tensor>{image}, generation_config, streamer);
    }

    /// @brief Generate a response given a chat history and config.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param config_map A config may contain GenerationConfig, values
    /// for its members, StreamerVariant, a single image or multiple
    /// images/videos, and audios.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    virtual VLMDecodedResults generate(const ChatHistory& history, const ov::AnyMap& config_map) = 0;

    /// @brief Get a Tokenizer used to tokenize input and detokenize
    /// output.
    virtual Tokenizer get_tokenizer() const = 0;

    /// @brief Set a custom chat template. Can be used to deactivate
    /// chat_template application for chat mode if called with
    /// "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    /// or workaround unsupported chat_template entries in a default
    /// model chat_template.
    /// @param new_template A new template to override with.
    virtual void set_chat_template(const std::string& new_template) = 0;

    /// @brief Extract GenerationConfig used to get default values.
    /// @return Default values used.
    virtual GenerationConfig get_generation_config() const = 0;

    /// @brief Override default values for GenerationConfig
    /// @param new_config A config to override default values with.
    virtual void set_generation_config(const GenerationConfig& new_config) = 0;

    // --- Qwen3-Omni support hooks --------------------------------------------------------------
    // Promoted onto the public base so OmniPipeline can compose a `shared_ptr<VLMPipelineBase>`
    // and still reach the Qwen3-Omni-specific plumbing without
    // depending on any internal sub-class. Meaningful only for Qwen3-Omni-capable pipelines;
    // conventional VLM implementations report `false` for the capability queries below.

    /// @brief videos_metadata-aware generate. Forwards to the backing implementation; used by
    /// OmniPipeline to drive Qwen3-VL-style timestamp prompts. `audios` carries raw audio inputs
    /// (Qwen3-Omni); conventional VLM implementations ignore it.
    virtual VLMDecodedResults generate(const std::string& prompt,
                                       const std::vector<ov::Tensor>& images,
                                       const std::vector<ov::Tensor>& videos,
                                       const std::vector<ov::Tensor>& audios,
                                       const std::vector<VideoMetadata>& videos_metadata,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) = 0;

    /// @brief videos_metadata-aware ChatHistory generate.
    virtual VLMDecodedResults generate(const ChatHistory& history,
                                       const std::vector<ov::Tensor>& images,
                                       const std::vector<ov::Tensor>& videos,
                                       const std::vector<ov::Tensor>& audios,
                                       const std::vector<VideoMetadata>& videos_metadata,
                                       const GenerationConfig& generation_config,
                                       const StreamerVariant& streamer) = 0;

    /// @brief Backend capability: true when the active execution path emits the per-step hidden
    /// states the talker consumes. Depends on how the model was loaded — only the continuous-batching
    /// backend collects them today; the SDPA fallback returns false. Independent of the model itself:
    /// the same Omni model reports true on the CB path and false on SDPA. OmniPipeline asserts on this
    /// so speech requests against the SDPA fallback fail early with a clear message.
    /// @see is_audio_output_enabled(), which instead reflects a fixed property of the loaded model.
    /// @note This is a preview API and is subject to change.
    virtual bool supports_hidden_states_collection() const = 0;

    /// @brief Model capability: true when the loaded model has a speech head (config.json
    /// enable_audio_output=true, i.e. Qwen3-Omni). Depends on which model was loaded, not on the
    /// backend — both backends report the same value for a given model. OmniPipeline's ctor rejects
    /// non-Omni-capable models on this signal.
    /// @see supports_hidden_states_collection(), which instead reflects the active backend.
    /// @note This is a preview API and is subject to change.
    virtual bool is_audio_output_enabled() const = 0;
};

/// @brief A Visual language modeling pipeline class used to generate a
/// response or run a chat given a prompt and an image.
class OPENVINO_GENAI_EXPORTS VLMPipeline : public VLMPipelineBase {
public:
    /// @brief Construct a pipeline from a folder containing tokenizer
    /// and model IRs.
    /// @param models_path A folder to read tokenizer and model IRs.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    VLMPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /// @brief Construct a pipeline from a map of models and their weights.
    /// @param models_map A map where key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
    /// and value is a pair of model IR as string and weights as tensor.
    /// @param tokenizer A tokenizer.
    /// @param config_dir_path A path to directory containing config.json.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    /// @param generation_config Optional generation configuration for the pipeline.
    VLMPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties = {},
        const ov::genai::GenerationConfig& generation_config = {}
    );

    /// @brief Construct a pipeline from a folder containing tokenizer
    /// and model IRs. Accepts arbitrary list of optional properties.
    /// @param models_path A folder to read tokenizer and model IRs.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VLMPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        Properties&&... properties)
        : VLMPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /// @brief Construct a pipeline from a map of models and their weights.
    /// @param models_map A map where key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
    /// and value is a pair of model IR as string and weights as tensor.
    /// @param tokenizer A tokenizer.
    /// @param config_dir_path A path to directory containing config.json.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VLMPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        Properties&&... properties)
        : VLMPipeline(models_map, tokenizer, config_dir_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /// @brief Default destructor.
    ~VLMPipeline();

    /// @brief Generate a response given a prompt and any number of
    /// uint8 RGB images with [NHWC] or [HWC] layout.
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param images Images to be prepended to a prompt.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    /// chat_template will be applied to the prompt, run pipe.set_chat_template(custom_chat_template) to update it.
    /// To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    /// @brief Generate a response given a prompt and uint8 RGB image with [NHWC] or [HWC] layout.
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param images Image to be prepended to a prompt.
    /// @param videos Multiple videos, each providing multiple frames, to be prepended to a prompt.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    /// chat_template will be applied to the prompt, run pipe.set_chat_template(custom_chat_template) to update it.
    /// To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    /// @brief Generate a response given a prompt and a single uint8 RGB image with [NHWC] or [HWC] layout.
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param image Image to be prepended to a prompt.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::Tensor& image,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    /// @brief Generate a response given a prompt and config.
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param config_map A config may contain GenerationConfig, values
    /// for its members, StreamerVariant, a single image or multiple
    /// images/videos, and audios.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    /// chat_template will be applied to the prompt, run pipe.set_chat_template(custom_chat_template) to update it.
    /// To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) override;

    /// @brief Generate a response given a prompt and arbitrary number
    /// of ov::Property instances.
    /// Example:
    /// generate("text", image(rgb), do_sample(true));
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param ...properties ov::Property instances to be combined into
    /// ov::AnyMap.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    /// chat_template will be applied to the prompt, run pipe.set_chat_template(custom_chat_template) to update it.
    /// To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    template <typename... Properties>
    util::EnableIfAllStringAny<VLMDecodedResults, Properties...> generate(
        const std::string& prompt,
        Properties&&... properties
    ) {
        return generate(
            prompt, AnyMap{std::forward<Properties>(properties)...}
        );
    }

    /// @brief Generate a response given a chat history and any number of
    /// uint8 RGB images with [NHWC] or [HWC] layout.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param images Images to be associated with the last chat history user message.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    /// @brief Generate a response given a chat history and any number of
    /// uint8 RGB images/videos with [NHWC] layout.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param images Images to be associated with the last chat history user message.
    /// @param videos Videos (each providing multiple frames) to be associated with the last chat history user message.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    /// @brief videos_metadata-aware generate. Forwards to the backing implementation; used by
    /// OmniPipeline to drive Qwen3-VL-style timestamp prompts. `audios` carries raw audio inputs
    /// (Qwen3-Omni); conventional VLM implementations ignore it.
    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<ov::Tensor>& audios,
        const std::vector<VideoMetadata>& videos_metadata,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    /// @brief videos_metadata-aware ChatHistory generate.
    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<ov::Tensor>& audios,
        const std::vector<VideoMetadata>& videos_metadata,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    /// @brief Generate a response given a chat history and a single uint8 RGB image with [NHWC] or [HWC] layout.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param image Image to be associated with the last chat history user message.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    VLMDecodedResults generate(
        const ChatHistory& history,
        const ov::Tensor& image,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) override;

    /// @brief Generate a response given a chat history and arbitrary number
    /// of ov::Property instances.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param config_map A config may contain GenerationConfig, values
    /// for its members, StreamerVariant, a single image or multiple
    /// images/videos, and audios.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    VLMDecodedResults generate(
        const ChatHistory& history,
        const ov::AnyMap& config_map
    ) override;

    /// @brief Generate a response given a chat history and config.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param ...properties ov::Property instances to be combined into
    /// ov::AnyMap.
    /// @return VLMDecodedResults structure containing generated texts, scores and perf metrics.
    template <typename... Properties>
    util::EnableIfAllStringAny<VLMDecodedResults, Properties...> generate(
        const ChatHistory& history,
        Properties&&... properties
    ) {
        return generate(
            history, AnyMap{std::forward<Properties>(properties)...}
        );
    }

    /// @brief Activate chat mode. Chat preserves previous history.
    /// Calling start_chat() again or finish_chat() drops the memorized history.
    /// @param system_message Some chat_templates contain system role
    /// in addition to user and assistant roles. Set a message for that
    /// role.
    /// @deprecated start_chat() / finish_chat() API is deprecated and will be removed in the next major release.
    /// Please, use generate() with ChatHistory argument.
    OPENVINO_DEPRECATED(
        "start_chat() / finish_chat() API is deprecated and will be removed in the next major release. "
        "Please, use generate() with ChatHistory argument.")
    void start_chat(const std::string& system_message="");

    /// @brief Deactivate chat mode.
    /// @deprecated start_chat() / finish_chat() API is deprecated and will be removed in the next major release.
    /// Please, use generate() with ChatHistory argument.
    OPENVINO_DEPRECATED(
        "start_chat() / finish_chat() API is deprecated and will be removed in the next major release. "
        "Please, use generate() with ChatHistory argument.")
    void finish_chat();

    /// @brief Set a custom chat template. Can be used to deactivate
    /// chat_template application for chat mode if called with
    /// "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    /// or workaround unsupported chat_template entries in a default
    /// model chat_template.
    /// @param new_template A new template to override with.
    void set_chat_template(const std::string& new_template) override;

    /// @brief Get a Tokenizer used to tokenize input and detokenize
    /// output.
    ov::genai::Tokenizer get_tokenizer() const override;

    /// @brief Extract GenerationConfig used to get default values.
    /// @return Default values used.
    GenerationConfig get_generation_config() const override;

    /// @brief Override default values for GenerationConfig
    /// @param new_config A config to override default values with.
    void set_generation_config(const GenerationConfig& new_config) override;

    // Qwen3-Omni support hooks forwarded to the backing implementation. See VLMPipelineBase.
    bool supports_hidden_states_collection() const override;
    bool is_audio_output_enabled() const override;

private:
    class VLMBackend;  // internal Omni-aware base; defined in src/cpp/src/visual_language/pipeline_base.hpp.
    class VLMPipelineImpl;
    class VLMContinuousBatchingAdapter;

    std::shared_ptr<VLMBackend> m_pimpl;
};

/*
 * utils that allow to use generate() in the following way:
 * pipe.generate(prompt, ov::genai::image(image_tensor)).
 * pipe.generate(prompt, ov::genai::images(image_tensors)).
 * pipe.generate(prompt, ov::genai::videos(videos_tensors)).
*/
static constexpr ov::Property<ov::Tensor> image{"image"};
static constexpr ov::Property<std::vector<ov::Tensor>> images{"images"};
static constexpr ov::Property<std::vector<ov::Tensor>> videos{"videos"};
static constexpr ov::Property<std::vector<VideoMetadata>> videos_metadata{"videos_metadata"};
static constexpr ov::Property<std::vector<ov::Tensor>> audios{"audios"};
}
