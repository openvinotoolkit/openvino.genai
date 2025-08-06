// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/flux_pipeline.hpp"

namespace ov {
namespace genai {

class FluxFillPipeline : public FluxPipeline {
public:
    FluxFillPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir)
        : FluxPipeline(pipeline_type, root_dir) {}

    FluxFillPipeline(PipelineType pipeline_type,
                     const std::filesystem::path& root_dir,
                     const std::string& device,
                     const ov::AnyMap& properties)
        : FluxPipeline(pipeline_type, root_dir, device, properties) {}

    FluxFillPipeline(PipelineType pipeline_type,
                     const CLIPTextModel& clip_text_model,
                     const T5EncoderModel& t5_text_model,
                     const FluxTransformer2DModel& transformer,
                     const AutoencoderKL& vae)
        : FluxPipeline(pipeline_type) {
        m_clip_text_encoder = std::make_shared<CLIPTextModel>(clip_text_model);
        m_t5_text_encoder = std::make_shared<T5EncoderModel>(t5_text_model);
        m_vae = std::make_shared<AutoencoderKL>(vae);
        m_transformer = std::make_shared<FluxTransformer2DModel>(transformer);
        initialize_generation_config("FluxFillPipeline");
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) override {

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        size_t num_channels_latents = m_vae->get_config().latent_channels;
        size_t height = generation_config.height / vae_scale_factor;
        size_t width = generation_config.width / vae_scale_factor;

        ov::Shape latent_shape{generation_config.num_images_per_prompt,
                               num_channels_latents,
                               height,
                               width};
        ov::Tensor latent, noise, proccesed_image, image_latents;

        proccesed_image = m_image_resizer->execute(initial_image, generation_config.height, generation_config.width);
        proccesed_image = m_image_processor->execute(proccesed_image);

        noise = generation_config.generator->randn_tensor(latent_shape);
        latent = pack_latents(noise, generation_config.num_images_per_prompt, num_channels_latents, height, width);

        return std::make_tuple(latent, proccesed_image, image_latents, noise);
    }

    std::tuple<ov::Tensor, ov::Tensor> prepare_mask_latents(ov::Tensor mask_image,
                                                            ov::Tensor processed_image,
                                                            const ImageGenerationConfig& generation_config,
                                                            const size_t batch_size_multiplier = 1) override {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::INPAINTING, "'prepare_mask_latents' can be called for inpainting pipeline only");

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        ov::Shape target_shape = processed_image.get_shape();

        // Prepare mask latent variables
        ov::Tensor mask_condition = m_image_resizer->execute(mask_image, generation_config.height, generation_config.width);
        std::shared_ptr<IImageProcessor> mask_processor = mask_condition.get_shape()[3] == 1 ? m_mask_processor_gray : m_mask_processor_rgb;
        mask_condition = mask_processor->execute(mask_condition);

        const ov::Shape processed_image_shape = processed_image.get_shape();
        const ov::Shape mask_condition_shape = mask_condition.get_shape();
        float * processed_image_data = processed_image.data<float>();
        const float * mask_condition_data = mask_condition.data<float>();

        OPENVINO_ASSERT(processed_image_shape[2] == mask_condition_shape[2] &&
                        processed_image_shape[3] == mask_condition_shape[3],
                        "Image and mask sizes are incompatible"
                       );
        
        if (processed_image.get_shape()[1] == mask_condition.get_shape()[1])  { // rgb mask
            for (size_t i = 0; i < processed_image.get_size(); ++i) {
                processed_image_data[i] *= (1.0f - mask_condition_data[i]);
                
            }
        } else { // gray mask
            size_t spatial_size = processed_image_shape[2] * processed_image_shape[3];
            for (size_t c = 0; c < processed_image_shape[1]; ++c) {
                for (size_t i = 0; i < spatial_size; ++i) {
                    processed_image_data[c * spatial_size + i] *= (1.0f - mask_condition_data[i]);
                }
            }
        }

        size_t num_channels_latents = m_vae->get_config().latent_channels;
        size_t height = generation_config.height / vae_scale_factor;
        size_t width = generation_config.width / vae_scale_factor;

        // Encode the masked image
        auto encode_start = std::chrono::steady_clock::now();
        ov::Tensor masked_image_latent = m_vae->encode(processed_image, generation_config.generator);
        m_perf_metrics.vae_encoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - encode_start).count();

        ov::Tensor repeated_masked_image_latent = repeat_mask(masked_image_latent, generation_config.num_images_per_prompt / masked_image_latent.get_shape()[0]);
        masked_image_latent = pack_latents(repeated_masked_image_latent, generation_config.num_images_per_prompt, num_channels_latents, height, width);

        ov::Tensor repeated_mask_condition = repeat_mask(mask_condition, generation_config.num_images_per_prompt / mask_condition.get_shape()[0]);
        transform_mask(repeated_mask_condition, generation_config.num_images_per_prompt, height, width, vae_scale_factor);
        ov::Tensor mask = pack_latents(repeated_mask_condition, generation_config.num_images_per_prompt, vae_scale_factor * vae_scale_factor, height, width);

        return std::make_tuple(mask, masked_image_latent);
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();
        m_custom_generation_config = m_generation_config;
        m_custom_generation_config.update_generation_config(properties);

        // Use callback if defined
        std::function<bool(size_t, size_t, ov::Tensor&)> callback = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback = callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>();
        }

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();

        if (m_custom_generation_config.height < 0)
            compute_dim(m_custom_generation_config.height, initial_image, 1 /* assume NHWC */);
        if (m_custom_generation_config.width < 0)
            compute_dim(m_custom_generation_config.width, initial_image, 2 /* assume NHWC */);

        check_inputs(m_custom_generation_config, initial_image);

        compute_hidden_states(positive_prompt, m_custom_generation_config);

        // Prepare latent variables
        ov::Tensor latents, processed_image, image_latent, noise;
        std::tie(latents, processed_image, image_latent, noise) = prepare_latents(initial_image, m_custom_generation_config);

        size_t image_seq_len = latents.get_shape()[1];
        m_scheduler->set_timesteps(image_seq_len, m_custom_generation_config.num_inference_steps, m_custom_generation_config.strength);

        // Prepare timesteps
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();
        m_latent_timestep = timesteps[0];

        // Prepare mask latents
        ov::Tensor mask, masked_image_latent;
        std::tie(mask, masked_image_latent) = prepare_mask_latents(mask_image, processed_image, m_custom_generation_config);
        ov::Tensor masked_image_latent_input = numpy_utils::concat(masked_image_latent, mask, -1);

        // Denoising loop
        ov::Tensor timestep(ov::element::f32, {1});
        float * timestep_data = timestep.data<float>();

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();
            timestep_data[0] = timesteps[inference_step] / 1000.0f;

            ov::Tensor latents_input = numpy_utils::concat(latents, masked_image_latent_input, 2);
            auto infer_start = std::chrono::steady_clock::now();
            ov::Tensor noise_pred_tensor = m_transformer->infer(latents_input, timestep);
            auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

            auto scheduler_step_result = m_scheduler->step(noise_pred_tensor, latents, inference_step, m_custom_generation_config.generator);
            latents = scheduler_step_result["latent"];

            if (callback && callback(inference_step, timesteps.size(), latents)) {
                auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
                m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));

                auto image = ov::Tensor(ov::element::u8, {});
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start)
                        .count();
                return image;
            }

            auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
        }

        latents = unpack_latents(latents, m_custom_generation_config.height, m_custom_generation_config.width, vae_scale_factor);
        const auto decode_start = std::chrono::steady_clock::now();
        auto image = m_vae->decode(latents);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();
        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        return image;
    }

private:
    bool is_inpainting_model() const override {
        return true;
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_transformer != nullptr);
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor() * 2;
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) && (width % vae_scale_factor == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ",
                        vae_scale_factor);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::INPAINTING, "FluxFillPipeline supports inpainting mode only");

        check_image_size(generation_config.height, generation_config.width);

        OPENVINO_ASSERT(generation_config.max_sequence_length <= 512, "T5's 'max_sequence_length' must be less or equal to 512");
        OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used by FluxFillPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by FluxFillPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by FluxFillPipeline");
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by FluxFillPipeline");
    }

    void transform_mask(ov::Tensor& mask, size_t batch_size, size_t height, size_t width, size_t vae_scale_factor) {
        ov::Shape mask_shape = mask.get_shape();
        OPENVINO_ASSERT(mask_shape.size() == 4 && mask_shape[1] == 1, "Unexpected mask shape");

        // Permutation to (0, 2, 4, 1, 3)
        auto transpose = [&](float* src, float* dst) {
            size_t height_width = height * width;
            size_t width_vsc = width * vae_scale_factor;
            size_t width_vsc_vsc = width * vae_scale_factor * vae_scale_factor;
            size_t height_width_vsc = height_width * vae_scale_factor;
            size_t height_width_vsc_vsc = height_width_vsc * vae_scale_factor;

            for (size_t b = 0; b < batch_size; ++b) {
                size_t b_height_width_vsc_vsc = b * height_width_vsc_vsc;

                for (size_t h = 0; h < height; ++h) {
                    size_t shift_1_src = b_height_width_vsc_vsc + h * width_vsc_vsc;
                    size_t shift_1_dst = b_height_width_vsc_vsc + h * width;

                    for (size_t w = 0; w < width; ++w) {
                        size_t shift_2_src = shift_1_src + w * vae_scale_factor;
                        size_t shift_2_dst = shift_1_dst + w;

                        for (size_t vh = 0; vh < vae_scale_factor; ++vh) {
                            size_t shift_3_src = shift_2_src + vh * width_vsc;
                            size_t shift_3_dst = shift_2_dst + vh * height_width_vsc;

                            for (size_t vw = 0; vw < vae_scale_factor; ++vw) {
                                size_t src_idx = shift_3_src + vw;
                                size_t dst_idx = shift_3_dst + vw * height_width;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                }
            }
        };

        float* mask_data = mask.data<float>();
        std::vector<size_t> final_shape = {batch_size, vae_scale_factor * vae_scale_factor, height, width};
        ov::Tensor final_mask(mask.get_element_type(), final_shape);
        transpose(mask_data, final_mask.data<float>());

        mask = ov::Tensor(mask.get_element_type(), final_shape);
        final_mask.copy_to(mask);
    }

    ov::Tensor repeat_mask(const ov::Tensor& masked_image_latents, size_t batch_size) {
        const ov::Shape& input_shape = masked_image_latents.get_shape();
        OPENVINO_ASSERT(input_shape.size() == 4, "Input tensor must have 4 dimensions.");

        size_t input_batch_size = input_shape[0], channels = input_shape[1];
        size_t height = input_shape[2], width = input_shape[3];

        OPENVINO_ASSERT(batch_size % input_batch_size == 0, "'batch_size' must be a multiple of the 'input_batch_size'");

        ov::Shape target_shape = {batch_size, channels, height, width};
        ov::Tensor repeated_tensor(masked_image_latents.get_element_type(), target_shape);

        const float* src_data = masked_image_latents.data<float>();
        float* dst_data = repeated_tensor.data<float>();

        size_t input_spatial_size = channels * height * width;

        for (size_t b = 0; b < batch_size; ++b) {
            size_t src_batch_index = b % input_batch_size;
            const float* src_batch = src_data + src_batch_index * input_spatial_size;
            float* dst_batch = dst_data + b * input_spatial_size;
            std::memcpy(dst_batch, src_batch, input_spatial_size * sizeof(float));
        }

        return repeated_tensor;
    }

};

}  // namespace genai
}  // namespace ov
