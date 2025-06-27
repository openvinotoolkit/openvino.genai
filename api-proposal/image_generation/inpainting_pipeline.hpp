// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <random>
#include <filesystem>

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"

#include "openvino/genai/image_generation/scheduler.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"

#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/unet2d_condition_model.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"
#include "openvino/genai/image_generation/flux_transformer_2d_model.hpp"

namespace ov {
namespace genai {

// forward declaration
class DiffusionPipeline;
class Text2ImagePipeline;
class Image2ImagePipeline;

//
// Inpainting pipeline
//

class OPENVINO_GENAI_EXPORTS InpaintingPipeline {
public:
    explicit InpaintingPipeline(const std::filesystem::path& models_path);

    InpaintingPipeline(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    InpaintingPipeline(const std::filesystem::path& models_path,
                       const std::string& device,
                       Properties&&... properties)
        : InpaintingPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    InpaintingPipeline(const Image2ImagePipeline& pipe);

    // creates either LCM or SD pipeline from building blocks
    static InpaintingPipeline stable_diffusion(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates either LCM or SD pipeline from building blocks
    static InpaintingPipeline latent_consistency_model(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates SDXL pipeline from building blocks
    static InpaintingPipeline stable_diffusion_xl(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const CLIPTextModelWithProjection& clip_text_model_with_projection,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates Flux pipeline from building blocks
    static InpaintingPipeline flux(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModel& clip_text_model,
        const T5EncoderModel& t5_text_encoder,
        const FluxTransformer2DModel& transformer,
        const AutoencoderKL& vae);

    // creates Flux pipeline from building blocks
    static InpaintingPipeline flux_fill(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModel& clip_text_model,
        const T5EncoderModel& t5_text_encoder,
        const FluxTransformer2DModel& transformer,
        const AutoencoderKL& vae);

    // creates SD3 pipeline from building blocks
    static InpaintingPipeline stable_diffusion_3(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModelWithProjection& clip_text_model_1,
        const CLIPTextModelWithProjection& clip_text_model_2,
        const T5EncoderModel& t5_encoder_model,
        const SD3Transformer2DModel& transformer,
        const AutoencoderKL& vae);

    // creates SD3 pipeline from building blocks
    static InpaintingPipeline stable_diffusion_3(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModelWithProjection& clip_text_model_1,
        const CLIPTextModelWithProjection& clip_text_model_2,
        const SD3Transformer2DModel& transformer,
        const AutoencoderKL& vae);

    ImageGenerationConfig get_generation_config() const;
    void set_generation_config(const ImageGenerationConfig& generation_config);

    // ability to override scheduler
    void set_scheduler(std::shared_ptr<Scheduler> scheduler);

    // with static shapes performance is better
    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale);

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Compiles image generation pipeline for given devices for text encoding, denoising, and vae decoding.
     * @param text_encode_device A device to compile text encoder(s) with
     * @param denoise_device A device to compile denoiser (e.g. UNet, SD3 Transformer, etc.) with
     * @param vae_device A device to compile VAE encoder / decoder(s) with
     * @param properties A map of properties which affect models compilation
     * @note If pipeline was compiled before, an exception is thrown.
     */
    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> compile(const std::string& text_encode_device,
                                                                const std::string& denoise_device,
                                                                const std::string& vae_device,
                                                                Properties&&... properties) {
        return compile(text_encode_device,
                       denoise_device,
                       vae_device,
                       ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Inpaints an initial image within an area defined by mask and conditioned on prompt
     * @param positive_prompt Prompt to generate image(s) from
     * @param initial_image RGB/BGR image of [1, height, width, 3] shape used to initialize latent image
     * @param mask_image RGB/BGR or GRAY/BINARY image of [1, height, width, 3 or 1] shape used as a mask
     * @param properties Image generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns A tensor which has dimensions [num_images_per_prompt, height, width, 3]
     */
    ov::Tensor generate(const std::string& positive_prompt, ov::Tensor initial_image, ov::Tensor mask_image, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
            const std::string& positive_prompt,
            ov::Tensor initial_image,
            ov::Tensor mask,
            Properties&&... properties) {
        return generate(positive_prompt, initial_image, mask, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ov::Tensor decode(const ov::Tensor latent);

    ImageGenerationPerfMetrics get_performance_metrics();

private:
    std::shared_ptr<DiffusionPipeline> m_impl;

    explicit InpaintingPipeline(const std::shared_ptr<DiffusionPipeline>& impl);

    // to create other pipelines from inpainting
    friend class Text2ImagePipeline;
    friend class Image2ImagePipeline;
};

} // namespace genai
} // namespace ov
