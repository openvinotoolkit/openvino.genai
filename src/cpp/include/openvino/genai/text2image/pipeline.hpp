// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <random>
#include <filesystem>

#include "openvino/core/any.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

#include "openvino/genai/visibility.hpp"

#include "openvino/genai/lora_adapter.hpp"
#include "openvino/genai/text2image/clip_text_model.hpp"
#include "openvino/genai/text2image/clip_text_model_with_projection.hpp"
#include "openvino/genai/text2image/unet2d_condition_model.hpp"
#include "openvino/genai/text2image/autoencoder_kl.hpp"

namespace ov {
namespace genai {

//
// Random generators
//

class OPENVINO_GENAI_EXPORTS Generator {
public:
    virtual float next() = 0;
    virtual ~Generator();
};

class OPENVINO_GENAI_EXPORTS CppStdGenerator : public Generator {
public:
    // creates 'std::mt19937' with initial 'seed' to generate numbers within a range [0.0f, 1.0f]
    explicit CppStdGenerator(uint32_t seed);

    virtual float next() override;
private:
    std::mt19937 gen;
    std::normal_distribution<float> normal;
};

//
// Text to image pipeline
//

class OPENVINO_GENAI_EXPORTS Text2ImagePipeline {
public:
    class OPENVINO_GENAI_EXPORTS Scheduler {
    public:
        enum Type {
            AUTO,
            LCM,
            LMS_DISCRETE,
            DDIM,
            EULER_DISCRETE
        };

        static std::shared_ptr<Scheduler> from_config(const std::filesystem::path& scheduler_config_path,
                                                      Type scheduler_type = AUTO);

        virtual ~Scheduler();
    };

    struct OPENVINO_GENAI_EXPORTS GenerationConfig {
        // LCM: prompt only w/o negative prompt
        // SD XL: prompt2 and negative_prompt2
        // FLUX: prompt2 (prompt if prompt2 is not defined explicitly)
        // SD 3: prompt2, prompt3 (with fallback to prompt) and negative_prompt2, negative_prompt3
        std::optional<std::string> prompt_2 = std::nullopt, prompt_3 = std::nullopt;
        std::string negative_prompt, negative_prompt_2, negative_prompt_3;

        // Optional image for image to image generation
        ov::Tensor image;

        size_t num_images_per_prompt = 1;

        // random generator to have deterministic results
        std::shared_ptr<Generator> random_generator = std::make_shared<CppStdGenerator>(42);

        // the following values depend on HF diffusers class used to perform generation
        float guidance_scale = 7.5f;
        int64_t height = -1;
        int64_t width = -1;
        size_t num_inference_steps = 50;

        AdapterConfig adapters;

        void update_generation_config(const ov::AnyMap& config_map);

        // checks whether is config is valid
        void validate() const;

        template <typename... Properties>
        ov::util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
            return update_generation_config(ov::AnyMap{std::forward<Properties>(properties)...});
        }
    };

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
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates either LCM or SD pipeline from building blocks
    static Text2ImagePipeline latent_consistency_model(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    // creates SDXL pipeline from building blocks
    static Text2ImagePipeline stable_diffusion_xl(
        const std::shared_ptr<Scheduler>& scheduler_type,
        const CLIPTextModel& clip_text_model,
        const CLIPTextModelWithProjection& clip_text_model_with_projection,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae);

    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& generation_config);

    // ability to override scheduler
    void set_scheduler(std::shared_ptr<Scheduler> scheduler);

    // with static shapes performance is better
    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale);

    void compile(const std::string& device, const ov::AnyMap& properties = {});

    // Returns a tensor with the following dimensions [num_images_per_prompt, height, width, 3]
    ov::Tensor generate(const std::string& positive_prompt, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
            const std::string& positive_prompt,
            Properties&&... properties) {
        return generate(positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

private:
    class DiffusionPipeline;
    std::shared_ptr<DiffusionPipeline> m_impl;

    explicit Text2ImagePipeline(const std::shared_ptr<DiffusionPipeline>& impl);

    class StableDiffusionPipeline;
    class StableDiffusionXLPipeline;
};

//
// Generation config properties
//

static constexpr ov::Property<std::string> prompt_2{"prompt_2"};
static constexpr ov::Property<std::string> prompt_3{"prompt_3"};

static constexpr ov::Property<std::string> negative_prompt{"negative_prompt"};
static constexpr ov::Property<std::string> negative_prompt_2{"negative_prompt_2"};
static constexpr ov::Property<std::string> negative_prompt_3{"negative_prompt_3"};

static constexpr ov::Property<ov::Tensor> image{"image"};

static constexpr ov::Property<size_t> num_images_per_prompt{"num_images_per_prompt"};
static constexpr ov::Property<float> guidance_scale{"guidance_scale"};
static constexpr ov::Property<int64_t> height{"height"};
static constexpr ov::Property<int64_t> width{"width"};
static constexpr ov::Property<size_t> num_inference_steps{"num_inference_steps"};

static constexpr ov::Property<std::shared_ptr<Generator>> random_generator{"random_generator"};

OPENVINO_GENAI_EXPORTS
std::pair<std::string, ov::Any> generation_config(const Text2ImagePipeline::GenerationConfig& generation_config);

} // namespace genai
} // namespace ov
