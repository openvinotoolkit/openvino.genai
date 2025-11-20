// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/image_generation/image2image_pipeline.hpp"
#include "openvino/genai/common_types.hpp"

namespace ov {
namespace genai {

/**
 * Text to image pipelines which provides unified API to all supported models types.
 * Models specific aspects are hidden in image generation config, which includes multiple prompts support or
 * other specific parameters like max_sequence_length
 */
class OPENVINO_GENAI_EXPORTS Text2ImagePipeline {
public:
    /**
     * Initializes text to image generation pipeline from a folder with models.
     * Note, such pipeline is not ready to use as models are not compiled internally.
     * 
     * Typical scenario is to initialize models using this constructor and then reshape pipeline
     * with 'reshape()' method and then perform compilation using 'compile()' method.
     * @param models_path A models path to read models and config files from
     */
    explicit Text2ImagePipeline(const std::filesystem::path& models_path);

    /**
     * Initializes text to image pipelines from a folder with models and performs compilation after it
     * @param models_path A models path to read models and config files from
     * @param device A single device used for all models
     * @param properties Properties to pass to 'compile_model' or other pipeline properties like LoRA adapters
     * @note If you want to compile each model on a dedicated device or with specific properties, you can create 
     * models individually and then combine a final pipeline using static methods like 'latent_consistency_model' or
     * 'stable_diffusion_3'. See 'samples/cpp/image_generation/heterogeneous_stable_diffusion.cpp' for example
     */
    Text2ImagePipeline(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Text2ImagePipeline(const std::filesystem::path& models_path,
                       const std::string& device,
                       Properties&&... properties)
        : Text2ImagePipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /**
     * Creates text to image pipeline based on image to image pipeline and shares models
     * @param pipe Image to image pipeline to share models with
     * @note Generation config is not shared with image to image pipeline and default one is created
     */
    Text2ImagePipeline(const Image2ImagePipeline& pipe);

    /**
     * Creates text to image pipeline based on inpainting pipeline and shares models
     * @param pipe Inpainting pipeline to share models with
     * @note Generation config is not shared with image to image pipeline and default one is created
     */
    Text2ImagePipeline(const InpaintingPipeline& pipe);

    /**
     * Creates Stable Diffusion pipeline from individual models
     * @param scheduler A scheduler used to denoise final image
     * @param clip_text_model A CLIP text encoder model
     * @param unet An Unet model
     * @param vae VAE auto encoder model
     */
    static Text2ImagePipeline stable_diffusion(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    /**
     * Creates Latent Consistency Model pipeline from individual models
     * @param scheduler A scheduler used to denoise final image
     * @param clip_text_model A CLIP text encoder model
     * @param unet An Unet denoising model
     * @param vae VAE auto encoder model
     */
    static Text2ImagePipeline latent_consistency_model(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    /**
     * Creates Stable Diffusion XL pipeline from individual models
     * @param scheduler A scheduler used to denoise final image
     * @param clip_text_model A CLIP text encoder model
     * @param clip_text_model_with_projection A CLIP text encoder with projection model
     * @param unet An Unet denoising model
     * @param vae VAE auto encoder model
     */
    static Text2ImagePipeline stable_diffusion_xl(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModel& clip_text_model,
        const CLIPTextModelWithProjection& clip_text_model_with_projection,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    /**
     * Creates Stable Diffusion 3 pipeline from individual models with T5 text encoder
     * @param scheduler A scheduler used to denoise final image
     * @param clip_text_model_1 A first CLIP text encoder model
     * @param clip_text_model_1 A second CLIP text encoder model
     * @param t5_encoder_model A T5 text encoder model.
     * @param transformer A Transformer denoising model
     * @param vae VAE auto encoder model
     */
    static Text2ImagePipeline stable_diffusion_3(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModelWithProjection& clip_text_model_1,
        const CLIPTextModelWithProjection& clip_text_model_2,
        const T5EncoderModel& t5_encoder_model,
        const SD3Transformer2DModel& transformer,
        const AutoencoderKL& vae);

    /**
     * Creates Stable Diffusion 3 pipeline from individual models without T5 text encoder
     * @param scheduler A scheduler used to denoise final image
     * @param clip_text_model_1 A first CLIP text encoder model
     * @param clip_text_model_1 A second CLIP text encoder model
     * @param transformer A Transformer denoising model
     * @param vae VAE auto encoder model
     */
    static Text2ImagePipeline stable_diffusion_3(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModelWithProjection& clip_text_model_1,
        const CLIPTextModelWithProjection& clip_text_model_2,
        const SD3Transformer2DModel& transformer,
        const AutoencoderKL& vae);

    /**
     * Creates FLUX pipeline from individual models
     * @param scheduler A scheduler used to denoise final image
     * @param clip_text_model A CLIP text encoder model
     * @param t5_encoder_model A T5 text encoder model
     * @param transformer A Transformer denoising model
     * @param vae VAE auto encoder model
     */
    static Text2ImagePipeline flux(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const T5EncoderModel t5_encoder_model,
        const FluxTransformer2DModel& transformer,
        const AutoencoderKL& vae);

    /**
     * Method to clone the pipeline to be used in parallel by another thread.
     * Reuses underlying models and recreates scheduler and generation config.
     * @returns A new pipeline for concurrent usage
     */
    Text2ImagePipeline clone();

    /**
     * Returns default image generation config created internally based on model type.
     * @returns Image generation config
     */
    ImageGenerationConfig get_generation_config() const;

    /**
     * Sets image generation config
     * @returns An image generation config
     */
    void set_generation_config(const ImageGenerationConfig& generation_config);

    /**
     * Overrides default scheduler used to denoise initial latent
     * @param scheduler A scheduler to set to a pipeline
     */
    void set_scheduler(std::shared_ptr<Scheduler> scheduler);

    /**
     * Reshapes pipeline based on a given set of reshape parameters, which affect shapes of models within pipeline
     * @note Reshaping can be useful to get maximum performance, but limit image generation to specific output sizes
     * @param num_images_per_prompt A number of image to generate per 'generate()' call
     * @param height A height of resulting image
     * @param width A width of resulting image
     * @param guidance_scale A guidance scale. Note, that it's important whether guidance_scale > 1, which affects whether negative prompts
     * are used or not. For example, all values > 1 are the same for reshape perspective and may vary in subsequent 'generate()' calls.
     * @note If pipeline has been already compiled, it cannot be reshaped and an exception is thrown.
     * 
     * Example how to reshape SD3 or Flux models for specific max sequence length:
     * @code
     *  ov::genai::Text2ImagePipeline pipe("/path");
     *  ov::genai::ImageGenerationConfig default_config = pipe.get_generation_config();
     *  default_config.max_sequence_length = 30;
     *  pipe.set_generation_config(default_config);
     *  pipe.reshape(1, 512, 512, default_config.guidance_scale); // reshape will bypass `max_sequence_length` to T5 encoder model
     * @endcode
     */
    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale);

    /**
     * Compiles image generation pipeline for a given device
     * @param device A device to compile models with
     * @param properties A map of properties which affect models compilation
     * @note If pipeline was compiled before, an exception is thrown.
     */
    void compile(const std::string& device, const ov::AnyMap& properties = {});

    /**
     * Compiles image generation pipeline for given devices for text encoding, denoising, and vae decoding.
     * @param text_encode_device A device to compile text encoder(s) with
     * @param denoise_device A device to compile denoiser (e.g. UNet, SD3 Transformer, etc.) with
     * @param vae_device A device to compile VAE decoder(s) with
     * @param properties A map of properties which affect models compilation
     * @note If pipeline was compiled before, an exception is thrown.
     */
    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> compile(const std::string& text_encode_device,
                                                                const std::string& denoise_device,
                                                                const std::string& vae_device,
                                                                Properties&&... properties) {
        return compile(text_encode_device,
                       denoise_device,
                       vae_device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Generates image(s) based on prompt and other image generation parameters
     * @param positive_prompt Prompt to generate image(s) from
     * @param properties Image generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns A tensor which has dimensions [num_images_per_prompt, height, width, 3]
     */
    ov::Tensor generate(const std::string& positive_prompt, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
            const std::string& positive_prompt,
            Properties&&... properties) {
        return generate(positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Performs latent image decoding. It can be useful to use within 'callback' which accepts current latent image
     * @param latent A latent image
     * @returns An image decoding with VAE auto encoder
     */
    ov::Tensor decode(const ov::Tensor latent);

    ImageGenerationPerfMetrics get_performance_metrics();

    /**
     * @brief Exports compiled models to a specified directory.
     * @param export_path A path to a directory to export compiled models to
     *
     * See @ref ov::genai::blob_path property to load previously exported models and for more details.
     */
    void export_model(const std::filesystem::path& export_path);

private:
    std::shared_ptr<DiffusionPipeline> m_impl;

    explicit Text2ImagePipeline(const std::shared_ptr<DiffusionPipeline>& impl);
};

} // namespace genai
} // namespace ov
