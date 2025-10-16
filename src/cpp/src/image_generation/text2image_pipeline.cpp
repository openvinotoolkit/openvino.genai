// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ctime>
#include <cstdlib>
#include <filesystem>

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "image_generation/stable_diffusion_pipeline.hpp"
#include "image_generation/stable_diffusion_xl_pipeline.hpp"
#include "image_generation/stable_diffusion_3_pipeline.hpp"
#include "image_generation/flux_pipeline.hpp"

#include "utils.hpp"

namespace ov {
namespace genai {

Text2ImagePipeline::Text2ImagePipeline(const std::filesystem::path& root_dir) {
    const std::string class_name = get_class_name(root_dir);

    auto start_time = std::chrono::steady_clock::now();
    if (class_name == "StableDiffusionPipeline" || 
        class_name == "LatentConsistencyModelPipeline")   {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::TEXT_2_IMAGE, root_dir);
    } else if (class_name == "StableDiffusionXLPipeline") {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::TEXT_2_IMAGE, root_dir);
    } else if (class_name == "StableDiffusion3Pipeline") {
        m_impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::TEXT_2_IMAGE, root_dir);
    } else if (class_name == "FluxPipeline") {
        m_impl = std::make_shared<FluxPipeline>(PipelineType::TEXT_2_IMAGE, root_dir);
    } else {
        OPENVINO_THROW("Unsupported text to image generation pipeline '", class_name, "'");
    }
    m_impl->save_load_time(start_time);
}

Text2ImagePipeline::Text2ImagePipeline(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties) {
    const std::string class_name = get_class_name(root_dir);

    auto start_time = std::chrono::steady_clock::now();
    if (class_name == "StableDiffusionPipeline" ||
        class_name == "LatentConsistencyModelPipeline") {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::TEXT_2_IMAGE, root_dir, device, properties);
    } else if (class_name == "StableDiffusionXLPipeline") {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::TEXT_2_IMAGE, root_dir, device, properties);
    } else if (class_name == "StableDiffusion3Pipeline") {
        m_impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::TEXT_2_IMAGE, root_dir, device, properties);
    } else if (class_name == "FluxPipeline") {
        m_impl = std::make_shared<FluxPipeline>(PipelineType::TEXT_2_IMAGE, root_dir, device, properties);
    } else {
        OPENVINO_THROW("Unsupported text to image generation pipeline '", class_name, "'");
    }
    m_impl->save_load_time(start_time);
}

Text2ImagePipeline::Text2ImagePipeline(const Image2ImagePipeline& pipe) {
    auto start_time = std::chrono::steady_clock::now();
    if (auto stable_diffusion_xl = std::dynamic_pointer_cast<StableDiffusionXLPipeline>(pipe.m_impl); stable_diffusion_xl != nullptr) {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::TEXT_2_IMAGE, *stable_diffusion_xl);
    } else if (auto stable_diffusion = std::dynamic_pointer_cast<StableDiffusionPipeline>(pipe.m_impl); stable_diffusion != nullptr) {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::TEXT_2_IMAGE, *stable_diffusion);
    } else if (auto stable_diffusion_3 = std::dynamic_pointer_cast<StableDiffusion3Pipeline>(pipe.m_impl); stable_diffusion_3 != nullptr) {
        m_impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::TEXT_2_IMAGE, *stable_diffusion_3);
    } else if (auto flux = std::dynamic_pointer_cast<FluxPipeline>(pipe.m_impl); flux != nullptr) {
        m_impl = std::make_shared<FluxPipeline>(PipelineType::TEXT_2_IMAGE, *flux);
    } else {
        OPENVINO_ASSERT("Cannot convert specified Image2ImagePipeline to Text2ImagePipeline");
    }
    m_impl->save_load_time(start_time);
}

Text2ImagePipeline::Text2ImagePipeline(const InpaintingPipeline& pipe) {
    auto start_time = std::chrono::steady_clock::now();
    if (auto stable_diffusion_xl = std::dynamic_pointer_cast<StableDiffusionXLPipeline>(pipe.m_impl); stable_diffusion_xl != nullptr) {
        m_impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::TEXT_2_IMAGE, *stable_diffusion_xl);
    } else if (auto stable_diffusion = std::dynamic_pointer_cast<StableDiffusionPipeline>(pipe.m_impl); stable_diffusion != nullptr) {
        m_impl = std::make_shared<StableDiffusionPipeline>(PipelineType::TEXT_2_IMAGE, *stable_diffusion);
    } else if (auto stable_diffusion_3 = std::dynamic_pointer_cast<StableDiffusion3Pipeline>(pipe.m_impl); stable_diffusion_3 != nullptr) {
        m_impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::TEXT_2_IMAGE, *stable_diffusion_3);
    } else if (auto flux = std::dynamic_pointer_cast<FluxPipeline>(pipe.m_impl); flux != nullptr) {
        m_impl = std::make_shared<FluxPipeline>(PipelineType::TEXT_2_IMAGE, *flux);
    } else {
        OPENVINO_ASSERT("Cannot convert specified InpaintingPipeline to Text2ImagePipeline");
    }
     m_impl->save_load_time(start_time);
}

Text2ImagePipeline::Text2ImagePipeline(const std::shared_ptr<DiffusionPipeline>& impl)
    : m_impl(impl) {
    assert(m_impl != nullptr);
}

Text2ImagePipeline Text2ImagePipeline::stable_diffusion(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const UNet2DConditionModel& unet,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<StableDiffusionPipeline>(PipelineType::TEXT_2_IMAGE, clip_text_model, unet, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return Text2ImagePipeline(impl);
}

Text2ImagePipeline Text2ImagePipeline::latent_consistency_model(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const UNet2DConditionModel& unet,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<StableDiffusionPipeline>(PipelineType::TEXT_2_IMAGE, clip_text_model, unet, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return Text2ImagePipeline(impl);
}

Text2ImagePipeline Text2ImagePipeline::stable_diffusion_xl(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const CLIPTextModelWithProjection& clip_text_model_with_projection,
    const UNet2DConditionModel& unet,
    const AutoencoderKL& vae) {
    auto impl = std::make_shared<StableDiffusionXLPipeline>(PipelineType::TEXT_2_IMAGE, clip_text_model, clip_text_model_with_projection, unet, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return Text2ImagePipeline(impl);
}

Text2ImagePipeline Text2ImagePipeline::stable_diffusion_3(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModelWithProjection& clip_text_model_1,
    const CLIPTextModelWithProjection& clip_text_model_2,
    const T5EncoderModel& t5_encoder_model,
    const SD3Transformer2DModel& transformer,
    const AutoencoderKL& vae){
    auto impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::TEXT_2_IMAGE, clip_text_model_1, clip_text_model_2, t5_encoder_model, transformer, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return Text2ImagePipeline(impl);
}

Text2ImagePipeline Text2ImagePipeline::stable_diffusion_3(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModelWithProjection& clip_text_model_1,
    const CLIPTextModelWithProjection& clip_text_model_2,
    const SD3Transformer2DModel& transformer,
    const AutoencoderKL& vae){
    auto impl = std::make_shared<StableDiffusion3Pipeline>(PipelineType::TEXT_2_IMAGE, clip_text_model_1, clip_text_model_2, transformer, vae);

    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);

    return Text2ImagePipeline(impl);
}

Text2ImagePipeline Text2ImagePipeline::flux(
    const std::shared_ptr<Scheduler>& scheduler,
    const CLIPTextModel& clip_text_model,
    const T5EncoderModel t5_encoder_model,
    const FluxTransformer2DModel& transformer,
    const AutoencoderKL& vae){
    auto impl = std::make_shared<FluxPipeline>(PipelineType::TEXT_2_IMAGE, clip_text_model, t5_encoder_model, transformer, vae);
    assert(scheduler != nullptr);
    impl->set_scheduler(scheduler);
    return Text2ImagePipeline(impl);
}

ImageGenerationConfig Text2ImagePipeline::get_generation_config() const {
    return m_impl->get_generation_config();
}

void Text2ImagePipeline::set_generation_config(const ImageGenerationConfig& generation_config) {
    m_impl->set_generation_config(generation_config);
}

void Text2ImagePipeline::set_scheduler(std::shared_ptr<Scheduler> scheduler) {
    m_impl->set_scheduler(scheduler);
}

void Text2ImagePipeline::reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->reshape(num_images_per_prompt, height, width, guidance_scale);
    m_impl->save_load_time(start_time);

    // update config with the specified parameters, so that the user doesn't need to explicitly pass these as properties
    // to generate()
    auto config = m_impl->get_generation_config();
    config.num_images_per_prompt = num_images_per_prompt;
    config.height = height;
    config.width = width;
    config.guidance_scale = guidance_scale;
    m_impl->set_generation_config(config);
}

void Text2ImagePipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(device, properties);
    m_impl->save_load_time(start_time);
}

void Text2ImagePipeline::compile(const std::string& text_encode_device,
    const std::string& denoise_device,
    const std::string& vae_device,
    const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl->compile(text_encode_device, denoise_device, vae_device, properties);
    m_impl->save_load_time(start_time);
}

ov::Tensor Text2ImagePipeline::generate(const std::string& positive_prompt, const ov::AnyMap& properties) {
    return m_impl->generate(positive_prompt, {}, {}, properties);
}

ov::Tensor Text2ImagePipeline::decode(const ov::Tensor latent) {
    return m_impl->decode(latent);
}

ImageGenerationPerfMetrics Text2ImagePipeline::get_performance_metrics() {
    return m_impl->get_performance_metrics();
}

Text2ImagePipeline Text2ImagePipeline::clone() {
    Text2ImagePipeline pipe(m_impl->clone());
    return pipe;
}

void Text2ImagePipeline::export_model(const std::filesystem::path& export_dir) {
    m_impl->export_model(export_dir);
}

}  // namespace genai
}  // namespace ov
