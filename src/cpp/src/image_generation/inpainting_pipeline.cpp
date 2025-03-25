// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ctime>
#include <cstdlib>
#include <filesystem>

#include "openvino/genai/image_generation/inpainting_pipeline.hpp"
#include "openvino/genai/image_generation/image2image_pipeline.hpp"

#include "image_generation/stable_diffusion_pipeline.hpp"
#include "image_generation/stable_diffusion_xl_pipeline.hpp"
#include "image_generation/stable_diffusion_3_pipeline.hpp"
#include "image_generation/flux_pipeline.hpp"
#include "image_generation/flux_fill_pipeline.hpp"

#include "utils.hpp"

namespace ov {
namespace genai {

InpaintingPipeline::InpaintingPipeline(const std::filesystem::path& root_dir) {
    const std::string class_name = get_class_name(root_dir);

    auto start_time = std::chrono::steady_clock::now();
    if (class_name == "StableDiffusionPipeline" || 
        class_name == "LatentConsistencyModelPipeline" ||
        class_name == "StableDiffusionInpaintPipeline") {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, root_dir);
    } else if (class_name == "StableDiffusionXLPipeline" || class_name == "StableDiffusionXLInpaintPipeline") {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::INPAINTING, root_dir);
    } else if (class_name == "FluxPipeline" || class_name == "FluxInpaintPipeline") {
        m_impl = std::make_shared<FluxPipeline>(PipelineType::INPAINTING, root_dir);
    } else if (class_name == "FluxFillPipeline") {
        m_impl = std::make_shared<FluxFillPipeline>(PipelineType::INPAINTING, root_dir);
    } else if (class_name == "StableDiffusion3Pipeline" || class_name == "StableDiffusion3InpaintPipeline") {
        m_impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::INPAINTING, root_dir);
    } else {
        OPENVINO_THROW("Unsupported inpainting pipeline '", class_name, "'");
    }
    m_impl->save_load_time(start_time);
}

InpaintingPipeline::InpaintingPipeline(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties) {
    const std::string class_name = get_class_name(root_dir);

    auto start_time = std::chrono::steady_clock::now();
    if (class_name == "StableDiffusionPipeline" ||
        class_name == "LatentConsistencyModelPipeline" ||
        class_name == "StableDiffusionInpaintPipeline") {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, root_dir, device, properties);
    } else if (class_name == "StableDiffusionXLPipeline" || class_name == "StableDiffusionXLInpaintPipeline") {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::INPAINTING, root_dir, device, properties);
    } else if (class_name == "FluxPipeline" || class_name == "FluxInpaintPipeline") {
        m_impl = std::make_shared<FluxPipeline>(PipelineType::INPAINTING, root_dir, device, properties);
    } else if (class_name == "FluxFillPipeline") {
        m_impl = std::make_shared<FluxFillPipeline>(PipelineType::INPAINTING, root_dir, device, properties);
    } else if (class_name == "StableDiffusion3Pipeline" || class_name == "StableDiffusion3InpaintPipeline") {
        m_impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::INPAINTING, root_dir, device, properties);
    } else {
        OPENVINO_THROW("Unsupported inpainting pipeline '", class_name, "'");
    }
    m_impl->save_load_time(start_time);
}

InpaintingPipeline::InpaintingPipeline(const Image2ImagePipeline& pipe) {
    auto start_time = std::chrono::steady_clock::now();
    if (auto stable_diffusion_xl = std::dynamic_pointer_cast<StableDiffusionXLPipeline>(pipe.m_impl); stable_diffusion_xl != nullptr) {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::INPAINTING, *stable_diffusion_xl);
    } else if (auto stable_diffusion = std::dynamic_pointer_cast<StableDiffusionPipeline>(pipe.m_impl); stable_diffusion != nullptr) {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, *stable_diffusion);
    } else if (auto stable_diffusion_3 = std::dynamic_pointer_cast<StableDiffusion3Pipeline>(pipe.m_impl); stable_diffusion_3 != nullptr) {
        m_impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::INPAINTING, *stable_diffusion_3);
    } else {
        OPENVINO_ASSERT("Cannot convert specified Image2ImagePipeline to InpaintingPipeline");
    }
    m_impl->save_load_time(start_time);
}

InpaintingPipeline::InpaintingPipeline(const std::shared_ptr<DiffusionPipeline>& impl)
    : m_impl(impl) {
    assert(m_impl != nullptr);
}

InpaintingPipeline InpaintingPipeline::stable_diffusion(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const UNet2DConditionModel& unet,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, clip_text_model, unet, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

InpaintingPipeline InpaintingPipeline::latent_consistency_model(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const UNet2DConditionModel& unet,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<StableDiffusionPipeline>(PipelineType::INPAINTING, clip_text_model, unet, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

InpaintingPipeline InpaintingPipeline::stable_diffusion_xl(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const CLIPTextModelWithProjection& clip_text_model_with_projection,
    const UNet2DConditionModel& unet,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::INPAINTING, clip_text_model, clip_text_model_with_projection, unet, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

InpaintingPipeline InpaintingPipeline::flux(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const T5EncoderModel& t5_text_encoder,
    const FluxTransformer2DModel& transformer,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<FluxPipeline>(PipelineType::INPAINTING, clip_text_model, t5_text_encoder, transformer, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

InpaintingPipeline InpaintingPipeline::flux_fill(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const T5EncoderModel& t5_text_encoder,
    const FluxTransformer2DModel& transformer,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<FluxFillPipeline>(PipelineType::INPAINTING, clip_text_model, t5_text_encoder, transformer, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

InpaintingPipeline InpaintingPipeline::stable_diffusion_3(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModelWithProjection& clip_text_model_1,
    const CLIPTextModelWithProjection& clip_text_model_2,
    const T5EncoderModel& t5_encoder_model,
    const SD3Transformer2DModel& transformer,
    const AutoencoderKL& vae){
    auto impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::INPAINTING, clip_text_model_1, clip_text_model_2, t5_encoder_model, transformer, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

InpaintingPipeline InpaintingPipeline::stable_diffusion_3(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModelWithProjection& clip_text_model_1,
    const CLIPTextModelWithProjection& clip_text_model_2,
    const SD3Transformer2DModel& transformer,
    const AutoencoderKL& vae){
    auto impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::INPAINTING, clip_text_model_1, clip_text_model_2, transformer, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return InpaintingPipeline(impl);
}

ImageGenerationConfig InpaintingPipeline::get_generation_config() const {
    return m_impl->get_generation_config();
}

void InpaintingPipeline::set_generation_config(const ImageGenerationConfig& generation_config) {
    m_impl->set_generation_config(generation_config);
}

void InpaintingPipeline::set_scheduler(std::shared_ptr<Scheduler> scheduler) {
    m_impl->set_scheduler(scheduler);
}

void InpaintingPipeline::reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->reshape(num_images_per_prompt, height, width, guidance_scale);
    m_impl->save_load_time(start_time);
}

void InpaintingPipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(device, properties);
    m_impl->save_load_time(start_time);
}

void InpaintingPipeline::compile(const std::string& text_encode_device,
                                 const std::string& denoise_device,
                                 const std::string& vae_device,
                                 const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(text_encode_device, denoise_device, vae_device, properties);
    m_impl->save_load_time(start_time);
}

ov::Tensor InpaintingPipeline::generate(const std::string& positive_prompt, ov::Tensor initial_image, ov::Tensor mask, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(initial_image, "Initial image cannot be empty when passed to InpaintingPipeline::generate");
    OPENVINO_ASSERT(mask, "Mask image cannot be empty when passed to InpaintingPipeline::generate");
    return m_impl->generate(positive_prompt, initial_image, mask, properties);
}

ov::Tensor InpaintingPipeline::decode(const ov::Tensor latent) {
    return m_impl->decode(latent);
}

ImageGenerationPerfMetrics InpaintingPipeline::get_performance_metrics() {
    return m_impl->get_performance_metrics();
}

}  // namespace genai
}  // namespace ov
