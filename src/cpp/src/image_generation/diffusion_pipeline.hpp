// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <tuple>

#include "image_generation/schedulers/ischeduler.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "lora_helper.hpp"
#include "lora_names_mapping.hpp"

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

    virtual void compile(const std::string& device, const ov::AnyMap& properties)
    {
        compile(device, device, device, properties);
    }

    virtual void compile(const std::string& text_encode_device,
                         const std::string& denoise_device,
                         const std::string& vae_decode_device,
                         const ov::AnyMap& properties) = 0;

    virtual std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) = 0;

    virtual void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) = 0;

    virtual void set_lora_adapters(std::optional<AdapterConfig> adapters) = 0;

    virtual ov::Tensor generate(const std::string& positive_prompt, ov::Tensor initial_image, ov::Tensor mask_image, const ov::AnyMap& properties) = 0;

    virtual ov::Tensor decode(const ov::Tensor latent) = 0;

    virtual ImageGenerationPerfMetrics get_performance_metrics() = 0;

    void save_load_time(std::chrono::steady_clock::time_point start_time) {
        auto stop_time = std::chrono::steady_clock::now();
        m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
    }

    virtual ~DiffusionPipeline() = default;

protected:
    virtual void initialize_generation_config(const std::string& class_name) = 0;

    virtual void check_image_size(const int height, const int width) const = 0;

    virtual void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const = 0;

    virtual void blend_latents(ov::Tensor image_latent, ov::Tensor noise, ov::Tensor mask, ov::Tensor latent, size_t inference_step) {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::INPAINTING, "'blend_latents' can be called for inpainting pipeline only");
        OPENVINO_ASSERT(image_latent.get_shape() == latent.get_shape(), "Shapes for current", latent.get_shape(), "and initial image latents ", image_latent.get_shape(), " must match");

        ov::Tensor noised_image_latent(image_latent.get_element_type(), {});
        std::vector<std::int64_t> timesteps = m_scheduler->get_timesteps();

        if (inference_step < timesteps.size() - 1) {
            image_latent.copy_to(noised_image_latent);

            int64_t noise_timestep = timesteps[inference_step + 1];
            m_scheduler->add_noise(noised_image_latent, noise, noise_timestep);
        } else {
            noised_image_latent = image_latent;
        }

        ov::Shape shape = image_latent.get_shape();
        size_t batch_size = shape[0], in_channels = shape[1], channel_size = shape[2] * shape[3];
        OPENVINO_ASSERT(batch_size == 1, "Batch size 1 is supported for now");

        const float * mask_data = mask.data<const float>();
        const float * noised_image_latent_data = noised_image_latent.data<const float>();
        float * latent_data = latent.data<float>();

        // blend initial noised and processed latents
        for (size_t i = 0; i < channel_size; ++i) {
            float mask_value = mask_data[i];
            for (size_t j = 0; j < in_channels; ++j) {
                latent_data[j * channel_size + i] = (1.0f - mask_value) * noised_image_latent_data[j * channel_size + i] + mask_value * latent_data[j * channel_size + i];
            }
        }
    }

    static std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters) {
        return ov::genai::derived_adapters(adapters, diffusers_adapter_normalization);
    }

    PipelineType m_pipeline_type;
    std::shared_ptr<IScheduler> m_scheduler;
    ImageGenerationConfig m_generation_config;
    float m_load_time_ms = 0.0f;
    ImageGenerationPerfMetrics m_perf_metrics;
};

} // namespace genai
} // namespace ov
