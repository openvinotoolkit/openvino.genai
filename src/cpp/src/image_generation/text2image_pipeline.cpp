// Copyright (C) 2023-2024 Intel Corporation
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
}

Text2ImagePipeline::Text2ImagePipeline(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties) {
    const std::string class_name = get_class_name(root_dir);

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
    m_impl->reshape(num_images_per_prompt, height, width, guidance_scale);
}

void Text2ImagePipeline::compile(const std::string& device, const ov::AnyMap& properties) {
    m_impl->compile(device, properties);
}

ov::Tensor Text2ImagePipeline::generate(const std::string& positive_prompt, const ov::AnyMap& properties) {
    return m_impl->generate(positive_prompt, {}, properties);
}

ov::Tensor Text2ImagePipeline::decode(const ov::Tensor latent) {
    return m_impl->decode(latent);
}

}  // namespace genai
}  // namespace ov
