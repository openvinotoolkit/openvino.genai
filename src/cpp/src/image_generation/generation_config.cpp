// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/generation_config.hpp"

#include <ctime>
#include <cstdlib>

#include "utils.hpp"

namespace ov {
namespace genai {

static constexpr char SD_GENERATION_CONFIG[] = "SD_GENERATION_CONFIG";

Generator::~Generator() = default;

ov::Tensor Generator::randn_tensor(const ov::Shape& shape) {
    ov::Tensor rand_tensor(ov::element::f32, shape);
    float * rand_tensor_data = rand_tensor.data<float>();

    for (size_t i = 0; i < rand_tensor.get_size(); ++i) {
        rand_tensor_data[i] = next();
    }

    return rand_tensor;
}

CppStdGenerator::CppStdGenerator(uint32_t seed)
    : m_gen(seed), m_normal(0.0f, 1.0f) {
}

float CppStdGenerator::next() {
    return m_normal(m_gen);
}

void CppStdGenerator::seed(size_t new_seed) {
    m_gen.seed(new_seed);
}

//
// GenerationConfig
//

std::pair<std::string, ov::Any> generation_config(const ImageGenerationConfig& generation_config) {
    return {SD_GENERATION_CONFIG, ov::Any::make<ImageGenerationConfig>(generation_config)};
}

void ImageGenerationConfig::update_generation_config(const ov::AnyMap& properties) {
    using utils::read_anymap_param;

    // override whole generation config first
    read_anymap_param(properties, SD_GENERATION_CONFIG, *this);

    // then try per-parameter values
    read_anymap_param(properties, "prompt_2", prompt_2);
    read_anymap_param(properties, "prompt_3", prompt_3);
    read_anymap_param(properties, "negative_prompt", negative_prompt);
    read_anymap_param(properties, "negative_prompt_2", negative_prompt_2);
    read_anymap_param(properties, "negative_prompt_3", negative_prompt_3);
    read_anymap_param(properties, "num_images_per_prompt", num_images_per_prompt);
    read_anymap_param(properties, "guidance_scale", guidance_scale);
    read_anymap_param(properties, "height", height);
    read_anymap_param(properties, "width", width);
    read_anymap_param(properties, "num_inference_steps", num_inference_steps);
    read_anymap_param(properties, "strength", strength);
    read_anymap_param(properties, "adapters", adapters);
    read_anymap_param(properties, "max_sequence_length", max_sequence_length);

    // 'generator' has higher priority than 'seed' parameter
    const bool have_generator_param = properties.find(ov::genai::generator.name()) != properties.end();
    if (have_generator_param) {
        read_anymap_param(properties, "generator", generator);
    } else {
        read_anymap_param(properties, "rng_seed", rng_seed);

        // initialize random generator with a given seed value
        if (!generator) {
            generator = std::make_shared<CppStdGenerator>(rng_seed);
        }

        const bool have_rng_seed = properties.find(ov::genai::rng_seed.name()) != properties.end();
        if (have_rng_seed) {
            // we need to change seed as an user have specified it manually
            generator->seed(rng_seed);
        }
    }

    validate();
}

void ImageGenerationConfig::validate() const {
    OPENVINO_ASSERT(guidance_scale > 1.0f || negative_prompt == std::nullopt, "Guidance scale <= 1.0 ignores negative prompt");
    OPENVINO_ASSERT(guidance_scale > 1.0f || negative_prompt_2 == std::nullopt, "Guidance scale <= 1.0 ignores negative prompt 2");
    OPENVINO_ASSERT(guidance_scale > 1.0f || negative_prompt_3 == std::nullopt, "Guidance scale <= 1.0 ignores negative prompt 3");
}

}  // namespace genai
}  // namespace ov
