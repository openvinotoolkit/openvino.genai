// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <optional>

#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/lora_adapter.hpp"

namespace ov::genai {
/**
 * Generation config used for Text2VideoPipeline.
 * Note, that not all values are applicable for all pipelines and models - please, refer
 * to corresponding properties for ImageGenerationConfig to understand a meaning and applicability for specific models.
 */
struct VideoGenerationConfig {
    /**
     * Negative prompt
     */
    std::optional<std::string> negative_prompt = std::nullopt;

    /**
     * A number of videos to generate per 'generate()' call
     */
    size_t num_videos_per_prompt = 1;

    /**
     * Random generator to initialize latents, add noise to initial images in case of image to image / inpainting pipelines
     * By default, random generator is initialized as `CppStdGenerator(generation_config.rng_seed)`
     * @note If `generator` is specified, it has higher priority than `rng_seed` parameter.
     */
    std::shared_ptr<Generator> generator = nullptr;

    float guidance_scale = 7.5f;
    int64_t height = -1;
    int64_t width = -1;
    int64_t num_inference_steps = -1;

    /**
     * Max sequence length for T5 encoder / tokenizer used in SD3 / FLUX models
     */
    int max_sequence_length = -1;

    /// guidance_rescale (`float`, *optional*, defaults to 0.0):
    /// Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
    /// Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
    /// [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
    /// Guidance rescale factor should fix overexposure when using zero terminal SNR.
    /// Mixes with the original results from guidance by factor guidance_rescale to avoid "plain looking" images.
    /// negative or 0.0 disables rescaling. NaN corresponds to model default which is 0.0 for LTX-Video thus disabled.
    std::optional<float> guidance_rescale = std::nullopt;
    /// The number of video frames to generate. 0 corresponds to model default which is 161 for LTX-Video.
    size_t num_frames = 0;
    /// Video frame rate. Affects rope_interpolation_scale. Any value can be used although positive
    /// non-infinity makes the most sense. NaN corresponds to model default which is 25.0f for LTX-Video.
    std::optional<float> frame_rate = std::nullopt;

    /**
     * TaylorSeer configuration for caching transformer outputs.
     * When set, enables TaylorSeer Lite acceleration which skips some transformer inferences
     * and predicts outputs using Taylor series approximation.
     */
    std::optional<TaylorSeerCacheConfig> taylorseer_config = std::nullopt;

    /// LoRA adapters applied during generation.
    std::optional<AdapterConfig> adapters = std::nullopt;

    /// Guidance scale for the audio modality.
    /// @note When unset, a pipeline first substitutes the loaded model's own default - LTX-2.3 ships one
    /// (7.0) that is independent of `guidance_scale`, so the audio modality can request classifier-free
    /// guidance on its own. Only a model with no audio default of its own, such as LTX-2.0, falls back to
    /// `guidance_scale`. The same applies to every `audio_*` field below.
    std::optional<float> audio_guidance_scale = std::nullopt;

    // NOTE: new fields must be appended here rather than inserted above. Pipeline default configs
    // (e.g. LTX2_DEFAULT_CONFIG) are positional aggregate initializers, so inserting mid-struct
    // silently shifts every value after the insertion point.

    /// Scale of the Spatio-Temporal Guidance term (LTX-2.3). Enables an extra transformer pass in
    /// which the blocks listed in `spatio_temporal_guidance_blocks` are perturbed. 0.0 disables it.
    std::optional<float> stg_scale = std::nullopt;
    /// Spatio-Temporal Guidance scale for the audio modality. 0.0 disables the audio term.
    std::optional<float> audio_stg_scale = std::nullopt;
    /// Scale of the modality isolation guidance term (LTX-2.3). Enables an extra transformer pass
    /// with audio-to-video and video-to-audio cross attention turned off. <= 1.0 disables it.
    /// @note The pass is shared with `audio_modality_scale`, but each modality is guided only when its
    /// own scale exceeds 1.0, so one modality can be disabled while the other stays on.
    std::optional<float> modality_scale = std::nullopt;
    /// Modality isolation guidance scale for audio. <= 1.0 disables the audio term.
    std::optional<float> audio_modality_scale = std::nullopt;
    /// Guidance rescale factor for the audio modality.
    std::optional<float> audio_guidance_rescale = std::nullopt;
    /// Transformer block indices perturbed by Spatio-Temporal Guidance. Indices outside the model's
    /// block range are ignored, matching the reference implementation.
    std::optional<std::vector<int64_t>> spatio_temporal_guidance_blocks = std::nullopt;

    /// Custom denoising schedule, a decreasing list of sigmas in (0, 1]. It replaces the schedule the scheduler
    /// would build from `num_inference_steps`, and its length is the number of steps. The scheduler's shifting
    /// is applied to it as diffusers does. LTX-2.5 exports default to the 8 sigmas of its distilled transformer.
    std::optional<std::vector<float>> sigmas = std::nullopt;

    /// Video-to-video edit: the noise level, a sigma in (0, 1], the source video is noised to before denoising.
    /// Unset with `edit_num_steps` set means the first sigma of the steps that run.
    std::optional<float> noise_scale = std::nullopt;

    /// Video-to-video edit: run only the last `edit_num_steps` steps of the schedule. Unset with `noise_scale` set
    /// means the steps whose sigma is not above `noise_scale`.
    std::optional<size_t> edit_num_steps = std::nullopt;

    /// LTX-2.5: rewrite the prompt with the model's prompt enhancer before generating, as diffusers'
    /// 'enable_prompt_enhancement' does. For image-to-video it also describes the conditioning image.
    bool enable_prompt_enhancement = false;

    /// System prompt of the prompt enhancer. Unset means LTX-2.5's default for text-to-video, or for image-to-video
    /// when an image or video conditions the generation.
    std::optional<std::string> system_prompt = std::nullopt;
};

/**
 * A number of videos to generate per generate() call. If you want to generate multiple images
 * for the same combination of generation parameters and text prompts, you can use this parameter
 * for better performance as internally compuations will be performed with batch for Transformer model
 * and text embeddings tensors will also be computed only once.
 */
static constexpr ov::Property<size_t> num_videos_per_prompt{"num_videos_per_prompt"};

/**
 * guidance_rescale (`float`, *optional*, defaults to 0.0):
 * Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
 * Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
 * [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
 * Guidance rescale factor should fix overexposure when using zero terminal SNR.
 * Mixes with the original results from guidance by factor guidance_rescale to avoid "plain looking" images.
 * 0.0 disables rescaling.
 */
static constexpr ov::Property<float> guidance_rescale{"guidance_rescale"};

/// The number of video frames to generate.
static constexpr ov::Property<size_t> num_frames{"num_frames"};
/// Video frame rate.
static constexpr ov::Property<float> frame_rate{"frame_rate"};
/// Guidance scale for the audio modality. Unset means the loaded model's default, which on LTX-2.3 is
/// independent of `guidance_scale` - see 'VideoGenerationConfig::audio_guidance_scale'.
static constexpr ov::Property<float> audio_guidance_scale{"audio_guidance_scale"};
/// Scale of the Spatio-Temporal Guidance term. 0.0 disables the extra transformer pass.
static constexpr ov::Property<float> stg_scale{"stg_scale"};
/// Spatio-Temporal Guidance scale for audio. 0.0 disables the audio term.
static constexpr ov::Property<float> audio_stg_scale{"audio_stg_scale"};
/// Scale of the modality isolation guidance term. <= 1.0 disables it for the video modality.
static constexpr ov::Property<float> modality_scale{"modality_scale"};
/// Modality isolation guidance scale for audio. <= 1.0 disables it for the audio modality.
static constexpr ov::Property<float> audio_modality_scale{"audio_modality_scale"};
/// Guidance rescale factor for audio.
static constexpr ov::Property<float> audio_guidance_rescale{"audio_guidance_rescale"};
/// Transformer block indices perturbed by Spatio-Temporal Guidance.
static constexpr ov::Property<std::vector<int64_t>> spatio_temporal_guidance_blocks{
    "spatio_temporal_guidance_blocks"};
/// Custom denoising schedule; see 'VideoGenerationConfig::sigmas'.
static constexpr ov::Property<std::vector<float>> sigmas{"sigmas"};
/// Video-to-video edit noise level; see 'VideoGenerationConfig::noise_scale'.
static constexpr ov::Property<float> noise_scale{"noise_scale"};
/// Video-to-video edit step count; see 'VideoGenerationConfig::edit_num_steps'.
static constexpr ov::Property<size_t> edit_num_steps{"edit_num_steps"};
/// Rewrite the prompt with LTX-2.5's prompt enhancer; see 'VideoGenerationConfig::enable_prompt_enhancement'.
static constexpr ov::Property<bool> enable_prompt_enhancement{"enable_prompt_enhancement"};
/// System prompt of the prompt enhancer; see 'VideoGenerationConfig::system_prompt'.
static constexpr ov::Property<std::string> system_prompt{"system_prompt"};

/**
 * Function to pass 'VideoGenerationConfig' as property to 'generate()' call.
 * @param generation_config An video generation config to convert to property-like format
 */
OPENVINO_GENAI_EXPORTS
std::pair<std::string, ov::Any> generation_config(const VideoGenerationConfig& generation_config);

}  // namespace ov::genai
