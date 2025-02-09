// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <random>
#include <optional>

#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/properties.hpp"

#include "openvino/genai/lora_adapter.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

//
// Random generators
//

/**
 * Base class to represent random generator used in Image generation pipelines
 */
class OPENVINO_GENAI_EXPORTS Generator {
public:
    /**
     * The function to return next random floating point value
     * @returns Floating point value within a [0, 1] range
     */
    virtual float next() = 0;

    /**
     * Generates a random tensor of floating point values with a given shape
     * By default, it creates a tensor and fills it using 'Generator::next()' method element by element,
     * but some random generator strategies have different pocilies how tensors are generated and this method
     * provides an ability to change it.
     */
    virtual ov::Tensor randn_tensor(const ov::Shape& shape);

    /**
     * Sets a new initial seed value to random generator
     * @param new_seed A new seed value
     */
    virtual void seed(size_t new_seed) = 0;

    /**
     * Default dtor defined to ensure working RTTI.
     */
    virtual ~Generator();
};

/**
 * Implementation of 'Generator' using standard C++ random library types 'std::mt19937' and 'std::normal_distribution<float>'
 */
class OPENVINO_GENAI_EXPORTS CppStdGenerator : public Generator {
public:
    /**
     * Initialized C++ STD generator with a given seed
     * @param seed A seed value
     */
    explicit CppStdGenerator(uint32_t seed);

    virtual float next() override;

    virtual void seed(size_t new_seed) override;

private:
    std::mt19937 m_gen;
    std::normal_distribution<float> m_normal;
};

/**
 * Generation config used for Image generation pipelines.
 * Note, that not all values are applicable for all pipelines and models - please, refer
 * to documentation of properties below to understand a meaning and applicability for specific models.
 */
struct OPENVINO_GENAI_EXPORTS ImageGenerationConfig {
    /**
     * Prompts and negative prompts
     */
    std::optional<std::string> prompt_2 = std::nullopt, prompt_3 = std::nullopt;
    std::optional<std::string> negative_prompt = std::nullopt, negative_prompt_2 = std::nullopt, negative_prompt_3 = std::nullopt;

    /**
     * A number of images to generate per 'generate()' call
     */
    size_t num_images_per_prompt = 1;

    /**
     * Random generator to initialize latents, add noise to initial images in case of image to image / inpainting pipelines
     * By default, random generator is initialized as `CppStdGenerator(generation_config.rng_seed)`
     * @note If `generator` is specified, it has higher priority than `rng_seed` parameter.
     */
    std::shared_ptr<Generator> generator = nullptr;

    /**
     * Seed for random generator
     * @note If `generator` is specified, it has higher priority than `rng_seed` parameter.
     */
    size_t rng_seed = 42;

    float guidance_scale = 7.5f;
    int64_t height = -1;
    int64_t width = -1;
    size_t num_inference_steps = 50;

    /**
     * Max sequence length for T5 encoder / tokenizer used in SD3 / FLUX models
     */
    int max_sequence_length = -1;

    /**
     * Strength parameter used in Image to imaage / Inpainting pipelines.
     * Must be 1.0 for text to image generation as no initial image is provided in such scenario.
     */
    float strength = 1.0f;

    /**
     * Holds LoRA adapters
     */
    std::optional<AdapterConfig> adapters;

    /**
     * Checks whether image generation config is valid, otherwise throws an exception.
     */
    void validate() const;

    /**
     * Updates generation config from a map of properties.
     * @param properties A map of properties
     */
    void update_generation_config(const ov::AnyMap& properties);

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(ov::AnyMap{std::forward<Properties>(properties)...});
    }
};

//
// Generation config properties
//

/**
 * Prompt 2 for models which have at least two text encoders. Currently, it's used for SDXL, SD3, FLUX
 */
static constexpr ov::Property<std::string> prompt_2{"prompt_2"};

/**
 * Prompt 3 for models which have three text encoders. Currently, it's used only for SD3
 */
static constexpr ov::Property<std::string> prompt_3{"prompt_3"};

/**
 * Negative prompt for models which have negative prompt. Currently, it's used for SD, SDXL, SD3
 */
static constexpr ov::Property<std::string> negative_prompt{"negative_prompt"};

/**
 * Negative prompt 2 for models which have at least two text encoders. Currently, it's used for SDXL, SD3
 */
static constexpr ov::Property<std::string> negative_prompt_2{"negative_prompt_2"};

/**
 * Negative prompt 3 for models which have three text encoders. Currently, it's used only for SD3
 */
static constexpr ov::Property<std::string> negative_prompt_3{"negative_prompt_3"};

/**
 * A number of images to generate per generate() call. If you want to generate multiple images
 * for the same combination of generation parameters and text prompts, you can use this parameter
 * for better performance as internally compuations will be performed with batch for Unet / Transformer models
 * and text embeddings tensors will also be computed only once.
 */
static constexpr ov::Property<size_t> num_images_per_prompt{"num_images_per_prompt"};

/**
 * Guidance scale parameter which controls how model sticks to text embeddings generated
 * by text encoders within a pipeline. Higher value of guidance scale moves image generation towards
 * text embeddings, but resulting image will be less natural and more augmented.
 */
static constexpr ov::Property<float> guidance_scale{"guidance_scale"};

/**
 * Specifies a height of a resulting image. Typically, image height must be divisible by VAE scale factor
 * (which is 8 in most of cases) which represents ratio between latent image / RGB image sizes.
 */
static constexpr ov::Property<int64_t> height{"height"};

/**
 * Specifies a width of a resulting image. Typically, image width must be divisible by VAE scale factor
 * (which is 8 in most of cases) which represents ratio between latent image / RGB image sizes.
 */
static constexpr ov::Property<int64_t> width{"width"};

/**
 * Defines a number of inference steps used to denoise initial noised latent to final image.
 * Note, that in case of image to image / inpainting pipelines, the resulting number of inference steps
 * is scaled with 'strength' parameter.
 */
static constexpr ov::Property<size_t> num_inference_steps{"num_inference_steps"};

/**
 * Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
 * starting point and more noise is added the higher the `strength`. The number of denoising steps depends
 * on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
 * process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
 * essentially ignores `image`.
 */
static constexpr ov::Property<float> strength{"strength"};

/**
 * Overrides default random generator used within image generation pipelines.
 * By default, 'CppStdGenerator' is used, but if you are running Image generation via
 * python code, you can additionally install 'torch' and use OpenVINO GenAI's 'TorchGenerator'
 * which ensures the generated images will look as in HuggingFace when the same sed value if used.
 */
static constexpr ov::Property<std::shared_ptr<Generator>> generator{"generator"};

/**
 * Seed for random generator
 * @note If `generator` is specified, it has higher priority than `rng_seed` parameter.
 */
extern OPENVINO_GENAI_EXPORTS ov::Property<size_t> rng_seed;

/**
 * This parameters limits max sequence length for T5 encoder for SD3 and FLUX models.
 * T5 tokenizer output is padded with pad tokens to 'max_sequence_length' within a pipeline.
 * So, for better performance, you can specify this parameter to lower value to speed-up
 * T5 encoder inference as well as inference of transformer denoising model.
 * For optimal performance it can be set to a number of tokens for 'prompt_3' / 'negative_prompt_3' for SD3
 * or `prompt_2` for FLUX.
 * Note, that images generated with different values of 'max_sequence_length' are slightly different, but quite close.
 */
static constexpr ov::Property<int> max_sequence_length{"max_sequence_length"};

/**
 * User callback for image generation pipelines, which is called within a pipeline with the following arguments:
 * - Current inference step
 * - Total number of inference steps. Note, that in case of 'strength' parameter, the number of inference steps is reduced linearly
 * - Tensor representing current latent. Such latent can be converted to human-readable representation via image generation pipeline 'decode()' method
 */
static constexpr ov::Property<std::function<bool(size_t, size_t, ov::Tensor&)>> callback{"callback"};

/**
 * Function to pass 'ImageGenerationConfig' as property to 'generate()' call.
 * @param generation_config An image generation config to convert to property-like format
 */
OPENVINO_GENAI_EXPORTS
std::pair<std::string, ov::Any> generation_config(const ImageGenerationConfig& generation_config);

} // namespace genai
} // namespace ov
