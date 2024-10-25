// Copyright (C) 2023-2024 Intel Corporation
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

struct OPENVINO_GENAI_EXPORTS ImageGenerationConfig {
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

//
// Generation config properties
//

static constexpr ov::Property<std::string> prompt_2{"prompt_2"};
static constexpr ov::Property<std::string> prompt_3{"prompt_3"};

static constexpr ov::Property<std::string> negative_prompt{"negative_prompt"};
static constexpr ov::Property<std::string> negative_prompt_2{"negative_prompt_2"};
static constexpr ov::Property<std::string> negative_prompt_3{"negative_prompt_3"};

static constexpr ov::Property<size_t> num_images_per_prompt{"num_images_per_prompt"};
static constexpr ov::Property<float> guidance_scale{"guidance_scale"};
static constexpr ov::Property<int64_t> height{"height"};
static constexpr ov::Property<int64_t> width{"width"};
static constexpr ov::Property<size_t> num_inference_steps{"num_inference_steps"};

static constexpr ov::Property<std::shared_ptr<Generator>> random_generator{"random_generator"};

OPENVINO_GENAI_EXPORTS
std::pair<std::string, ov::Any> generation_config(const ImageGenerationConfig& generation_config);

} // namespace genai
} // namespace ov
