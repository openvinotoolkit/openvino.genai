// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/video_generation/generation_config.hpp"
#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"
#include "openvino/genai/image_generation/scheduler.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "openvino/genai/video_generation/autoencoder_kl_ltx_video.hpp"
#include "openvino/genai/video_generation/ltx_video_transformer_3d_model.hpp"

namespace ov::genai {

struct VideoGenerationPerfMetrics : public ImageGenerationPerfMetrics {};

struct VideoGenerationResult {
    ov::Tensor video;
    ov::genai::VideoGenerationPerfMetrics performance_stat;
};

class OPENVINO_GENAI_EXPORTS Text2VideoPipeline {
public:
    /**
     * Initializes text to video generation pipeline from a folder with models.
     * Note, such pipeline is not ready to use as models are not compiled internally.
     *
     * Typical scenario is to initialize models using this constructor and then reshape pipeline
     * with 'reshape()' method and then perform compilation using 'compile()' method.
     * @param models_path A models path to read models and config files from
     */
    explicit Text2VideoPipeline(const std::filesystem::path& models_path);

    /**
     * Initializes text to video pipelines from a folder with models and performs compilation after it
     * @param models_path A models path to read models and config files from
     * @param device A single device used for all models
     * @param properties Properties to pass to 'compile_model' or other pipeline properties like LoRA adapters
     * @note If you want to compile each model on a dedicated device or with specific properties, you can create
     * models individually and then combine a final pipeline using static methods
     */
    Text2VideoPipeline(const std::filesystem::path& models_dir, const std::string& device, const AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Text2VideoPipeline(const std::filesystem::path& models_path,
                       const std::string& device,
                       Properties&&... properties)
        : Text2VideoPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /**
     * Creates LTX pipeline from individual models
     * @param scheduler A scheduler used to denoise final image
     * @param t5_text_encoder A T5 text encoder model
     * @param transformer A Transformer denoising model
     * @param vae VAE auto encoder model
     */
    static Text2VideoPipeline ltx_video(std::shared_ptr<Scheduler> m_scheduler,
                                        const T5EncoderModel& m_t5_text_encoder,
                                        const LTXVideoTransformer3DModel& m_transformer,
                                        const AutoencoderKLLTXVideo& m_vae);

    /**
     * Method to clone the pipeline to be used in parallel by another thread.
     * Reuses underlying models and recreates scheduler and generation config.
     * @returns A new pipeline for concurrent usage
     */
    Text2VideoPipeline clone();

    /**
     * Returns default video generation config created internally based on model type.
     * @returns Video generation config
     */
    const VideoGenerationConfig& get_generation_config() const;

    /**
     * Sets video generation config
     * @returns An video generation config
     */
    void set_generation_config(const VideoGenerationConfig& generation_config);

    /**
     * Reshapes pipeline based on a given set of reshape parameters, which affect shapes of models within pipeline
     * @note Reshaping can be useful to get maximum performance, but limit image generation to specific output sizes
     * @param num_videos_per_prompt A number of image to generate per 'generate()' call
     * @param num_frames A number of video frames to generate per 'generate()' call
     * @param height A height of resulting video
     * @param width A width of resulting video
     * @param guidance_scale A guidance scale. Note, that it's important whether guidance_scale > 1, which affects whether negative prompts
     * are used or not. For example, all values > 1 are the same for reshape perspective and may vary in subsequent 'generate()' calls.
     * @note If pipeline has been already compiled, it cannot be reshaped and an exception is thrown.
     *
     * Example how to reshape SD3 or Flux models for specific max sequence length:
     * @code
     *  ov::genai::Text2VideoPipeline pipe("/path");
     *  ov::genai::VideoGenerationConfig default_config = pipe.get_generation_config();
     *  default_config.max_sequence_length = 30;
     *  pipe.set_generation_config(default_config);
     *  pipe.reshape(1, 25, 512, 512, default_config.guidance_scale); // reshape will bypass `max_sequence_length` to T5 encoder model
     * @endcode
     */
    void reshape(const int64_t num_videos_per_prompt, const int64_t num_frames, const int64_t height, const int64_t width, const float guidance_scale);

    /**
     * Compiles video generation pipeline for a given device
     * @param device A device to compile models with
     * @param properties A map of properties which affect models compilation
     * @note If pipeline was compiled before, an exception is thrown.
     */
    void compile(const std::string& device, const ov::AnyMap& properties = {});

    /**
     * Compiles video generation pipeline for given devices for text encoding, denoising, and vae decoding.
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
     * Generates video(s) based on prompt and other video generation parameters
     * @param positive_prompt Prompt to generate video(s) from
     * @param properties Video generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns VideoGenerationResult with:
     *   - video: a tensor shaped as [num_videos_per_prompt, num_frames, height, width, 3]
     *   - performance_stat: ov::genai::VideoGenerationPerfMetrics with timing and other performance metrics for the generation run.
     */

    VideoGenerationResult generate(const std::string& positive_prompt, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<VideoGenerationResult, Properties...> generate(
        const std::string& positive_prompt,
        Properties&&... properties
    ) {
        return generate(positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * Performs latent video decoding. It can be useful to use within 'callback' which accepts current latent video
     * @param latent A latent video
     * @returns VideoGenerationResult with:
     *   - video: a video tensor decoded with VAE auto encoder shaped as [num_videos_per_prompt, num_frames, height, width, 3]
     *   - performance_stat: ov::genai::VideoGenerationPerfMetrics with timing and other performance metrics for the generation run.
     */
    VideoGenerationResult decode(const ov::Tensor& latent);

    /**
     * @brief Exports compiled models to a specified directory.
     * @param export_path A path to a directory to export compiled models to
     *
     * See @ref ov::genai::blob_path property to load previously exported models and for more details.
     */
    void export_model(const std::filesystem::path& export_path);

    ~Text2VideoPipeline();

private:
    class LTXPipeline;
    std::unique_ptr<LTXPipeline> m_impl;
};

}  // namespace ov::genai
