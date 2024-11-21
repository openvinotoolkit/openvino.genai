// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>

#include "image_generation/schedulers/ischeduler.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"

#include "json_utils.hpp"
namespace {

const std::string get_class_name(const std::filesystem::path& root_dir) {
    const std::filesystem::path model_index_path = root_dir / "model_index.json";
    std::ifstream file(model_index_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using ov::genai::utils::read_json_param;

    return data["_class_name"].get<std::string>();
}

ov::Tensor get_guidance_scale_embedding(float guidance_scale, uint32_t embedding_dim) {
    float w = guidance_scale * 1000;
    uint32_t half_dim = embedding_dim / 2;
    float emb = std::log(10000) / (half_dim - 1);

    ov::Shape embedding_shape = {1, embedding_dim};
    ov::Tensor w_embedding(ov::element::f32, embedding_shape);
    float* w_embedding_data = w_embedding.data<float>();

    for (size_t i = 0; i < half_dim; ++i) {
        float temp = std::exp((i * (-emb))) * w;
        w_embedding_data[i] = std::sin(temp);
        w_embedding_data[i + half_dim] = std::cos(temp);
    }

    if (embedding_dim % 2 == 1)
        w_embedding_data[embedding_dim - 1] = 0;

    return w_embedding;
}

} // namespace


namespace ov {
namespace genai {

enum class PipelineType {
    TEXT_2_IMAGE = 0,
    IMAGE_2_IMAGE = 1,
    INPAINTING = 2,
};

class DiffusionPipeline {
public:
    explicit DiffusionPipeline(PipelineType pipeline_type) :
        m_pipeline_type(pipeline_type) { }

    ImageGenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const ImageGenerationConfig& generation_config) {
        m_generation_config = generation_config;
        m_generation_config.validate();
    }

    void set_scheduler(std::shared_ptr<Scheduler> scheduler) {
        auto casted = std::dynamic_pointer_cast<IScheduler>(scheduler);
        OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
        m_scheduler = casted;
    }

    virtual void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) = 0;

    virtual void compile(const std::string& device, const ov::AnyMap& properties) = 0;

    virtual ov::Tensor prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) const = 0;

    virtual void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) = 0;

    virtual ov::Tensor generate(const std::string& positive_prompt, ov::Tensor initial_image, ov::Tensor mask_image, const ov::AnyMap& properties) = 0;

    virtual ov::Tensor decode(const ov::Tensor latent) = 0;

    virtual ~DiffusionPipeline() = default;

protected:
    virtual void initialize_generation_config(const std::string& class_name) = 0;

    virtual void check_image_size(const int height, const int width) const = 0;

    virtual void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const = 0;

    PipelineType m_pipeline_type;
    std::shared_ptr<IScheduler> m_scheduler;
    ImageGenerationConfig m_generation_config;
};

} // namespace genai
} // namespace ov
