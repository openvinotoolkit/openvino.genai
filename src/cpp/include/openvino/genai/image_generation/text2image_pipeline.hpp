// Copyright (C) 2023-2024 Intel Corporation
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

#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/unet2d_condition_model.hpp"
#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "openvino/genai/image_generation/flux_transformer_2d_model.hpp"

namespace ov {
namespace genai {

// forward declaration
class DiffusionPipeline;

//
// Text to image pipeline
//

class OPENVINO_GENAI_EXPORTS Text2ImagePipeline {
public:
    explicit Text2ImagePipeline(const std::filesystem::path& models_path);

    Text2ImagePipeline(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Text2ImagePipeline(const std::filesystem::path& models_path,
                       const std::string& device,
                       Properties&&... properties)
        : Text2ImagePipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    // creates either LCM or SD pipeline from building blocks
    static Text2ImagePipeline stable_diffusion(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates either LCM or SD pipeline from building blocks
    static Text2ImagePipeline latent_consistency_model(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates SDXL pipeline from building blocks
    static Text2ImagePipeline stable_diffusion_xl(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModel& clip_text_model,
        const CLIPTextModelWithProjection& clip_text_model_with_projection,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates SD3 pipeline from building blocks
    static Text2ImagePipeline stable_diffusion_3(
        const std::shared_ptr<Scheduler>& scheduler,
        const CLIPTextModelWithProjection& clip_text_model_1,
        const CLIPTextModelWithProjection& clip_text_model_2,
        const T5EncoderModel& t5_encoder_model,
        const SD3Transformer2DModel& transformer,
        const AutoencoderKL& vae);

    // creates Flux pipeline from building blocks
    static Text2ImagePipeline flux(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const T5EncoderModel t5_encoder_model,
        const FluxTransformer2DModel& transformer,
        const AutoencoderKL& vae_decoder);

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

    // Returns a tensor with the following dimensions [num_images_per_prompt, height, width, 3]
    ov::Tensor generate(const std::string& positive_prompt, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
            const std::string& positive_prompt,
            Properties&&... properties) {
        return generate(positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ov::Tensor decode(const ov::Tensor latent);

private:
    std::shared_ptr<DiffusionPipeline> m_impl;

    explicit Text2ImagePipeline(const std::shared_ptr<DiffusionPipeline>& impl);
};

} // namespace genai
} // namespace ov
