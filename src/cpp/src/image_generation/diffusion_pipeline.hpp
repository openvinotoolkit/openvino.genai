// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <filesystem>
#include <fstream>
#include <tuple>

#include "image_generation/schedulers/ischeduler.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/image_processor.hpp"

#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"

#include "lora/helper.hpp"
#include "lora/names_mapping.hpp"

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
    explicit DiffusionPipeline(PipelineType pipeline_type) : m_pipeline_type(pipeline_type) {
        // TODO: support GPU as well
        const std::string device = "CPU";

        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
            const bool do_normalize = true, do_binarize = false, gray_scale_source = false;
            m_image_processor = std::make_shared<ImageProcessor>(device, do_normalize, do_binarize, gray_scale_source);
            m_image_resizer = std::make_shared<ImageResizer>(device, ov::element::u8, "NHWC", ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW);
        }

        if (m_pipeline_type == PipelineType::INPAINTING) {
            bool do_normalize = false, do_binarize = true;
            m_mask_processor_rgb = std::make_shared<ImageProcessor>(device, do_normalize, do_binarize, false);
            m_mask_processor_gray = std::make_shared<ImageProcessor>(device, do_normalize, do_binarize, true);
            m_mask_resizer = std::make_shared<ImageResizer>(device, ov::element::f32, "NCHW", ov::op::v11::Interpolate::InterpolateMode::NEAREST);
        }
    }

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

    virtual std::shared_ptr<DiffusionPipeline> clone() = 0;

    virtual void compile(const std::string& device, const ov::AnyMap& properties)
    {
        compile(device, device, device, properties);
    }

    virtual void compile(const std::string& text_encode_device,
                         const std::string& denoise_device,
                         const std::string& vae_device,
                         const ov::AnyMap& properties) = 0;

    virtual std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) = 0;

    virtual void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) = 0;

    virtual void set_lora_adapters(std::optional<AdapterConfig> adapters) = 0;

    virtual ov::Tensor generate(const std::string& positive_prompt, ov::Tensor initial_image, ov::Tensor mask_image, const ov::AnyMap& properties) = 0;

    virtual ov::Tensor decode(const ov::Tensor latent) = 0;

    virtual ImageGenerationPerfMetrics get_performance_metrics() = 0;

    void save_load_time(std::chrono::steady_clock::time_point start_time) {
        auto stop_time = std::chrono::steady_clock::now();
        m_load_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
    }

    virtual void export_model(const std::filesystem::path& export_dir) {
        OPENVINO_THROW("Export model is not implemented for this pipeline");
    }

    virtual ~DiffusionPipeline() = default;

protected:
    virtual void initialize_generation_config(const std::string& class_name) = 0;

    virtual void check_image_size(const int height, const int width) const = 0;

    virtual void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const = 0;

    virtual bool is_inpainting_model() const {
        assert(m_vae != nullptr);
        return get_config_in_channels() == (m_vae->get_config().latent_channels * 2 + 1);
    }

    virtual size_t get_config_in_channels() const = 0;

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

    virtual std::tuple<ov::Tensor, ov::Tensor> prepare_mask_latents(ov::Tensor mask_image,
                                                                    ov::Tensor processed_image,
                                                                    const ImageGenerationConfig& generation_config,
                                                                    const size_t batch_size_multiplier) {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::INPAINTING, "'prepare_mask_latents' can be called for inpainting pipeline only");

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        ov::Shape target_shape = processed_image.get_shape();

        ov::Tensor mask_condition = m_image_resizer->execute(mask_image, target_shape[2], target_shape[3]);
        std::shared_ptr<IImageProcessor> mask_processor = mask_condition.get_shape()[3] == 1 ? m_mask_processor_gray : m_mask_processor_rgb;
        mask_condition = mask_processor->execute(mask_condition);

        // resize mask to shape of latent space
        ov::Tensor mask = m_mask_resizer->execute(mask_condition, target_shape[2] / vae_scale_factor, target_shape[3] / vae_scale_factor);
        mask = numpy_utils::repeat(mask, generation_config.num_images_per_prompt * batch_size_multiplier);

        ov::Tensor masked_image_latent;

        if (is_inpainting_model()) {
            // create masked image
            ov::Tensor masked_image(ov::element::f32, processed_image.get_shape());
            const float * mask_condition_data = mask_condition.data<const float>();
            const float * processed_image_data = processed_image.data<const float>();
            float * masked_image_data = masked_image.data<float>();

            for (size_t i = 0, plane_size = mask_condition.get_shape()[2] * mask_condition.get_shape()[3]; i < mask_condition.get_size(); ++i) {
                masked_image_data[i + 0 * plane_size] = mask_condition_data[i] < 0.5f ? processed_image_data[i + 0 * plane_size] : 0.0f;
                masked_image_data[i + 1 * plane_size] = mask_condition_data[i] < 0.5f ? processed_image_data[i + 1 * plane_size] : 0.0f;
                masked_image_data[i + 2 * plane_size] = mask_condition_data[i] < 0.5f ? processed_image_data[i + 2 * plane_size] : 0.0f;
            }

            // encode masked image to latent scape
            auto encode_start = std::chrono::steady_clock::now();
            masked_image_latent = m_vae->encode(masked_image, generation_config.generator);
            m_perf_metrics.vae_encoder_inference_duration += std::chrono::duration_cast<std::chrono::milliseconds>(
                                                             std::chrono::steady_clock::now() - encode_start).count();
            masked_image_latent = numpy_utils::repeat(masked_image_latent, generation_config.num_images_per_prompt * batch_size_multiplier);
        }

        return std::make_tuple(mask, masked_image_latent);
    }

    PipelineType m_pipeline_type;
    std::shared_ptr<IScheduler> m_scheduler;
    ImageGenerationConfig m_generation_config;
    float m_load_time_ms = 0.0f;
    ImageGenerationPerfMetrics m_perf_metrics;
    std::filesystem::path m_root_dir;

    std::shared_ptr<AutoencoderKL> m_vae = nullptr;
    std::shared_ptr<IImageProcessor> m_image_processor = nullptr, m_mask_processor_rgb = nullptr, m_mask_processor_gray = nullptr;
    std::shared_ptr<ImageResizer> m_image_resizer = nullptr, m_mask_resizer = nullptr;
};

} // namespace genai
} // namespace ov
