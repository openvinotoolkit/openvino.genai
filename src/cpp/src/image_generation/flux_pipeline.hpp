// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "utils.hpp"

namespace {

ov::Tensor pack_latents(const ov::Tensor latents, size_t batch_size, size_t num_channels_latents, size_t height, size_t width) {
    size_t h_half = height / 2, w_half = width / 2;

    // Reshape to (batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    ov::Shape final_shape = {batch_size, h_half * w_half, num_channels_latents * 4};
    ov::Tensor permuted_latents = ov::Tensor(latents.get_element_type(), final_shape);

    OPENVINO_ASSERT(latents.get_size() == permuted_latents.get_size(), "Incorrect target shape, tensors must have the same sizes");

    float* src_data = latents.data<float>();
    float* dst_data = permuted_latents.data<float>();

    // Permute to (0, 2, 4, 1, 3, 5)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h2 = 0; h2 < h_half; ++h2) {
            for (size_t w2 = 0; w2 < w_half; ++w2) {
                for (size_t c = 0; c < num_channels_latents; ++c) {
                    for (size_t h3 = 0; h3 < 2; ++h3) {
                        for (size_t w3 = 0; w3 < 2; ++w3) {
                            size_t src_index = ((b * num_channels_latents + c) * h_half + h2) * 2 * w_half * 2 + (h3 * w_half + w2) * 2 + w3;
                            size_t dst_index = ((b * h_half + h2) * w_half + w2) * num_channels_latents * 4 + (c * 4 + h3 * 2 + w3);

                            dst_data[dst_index] = src_data[src_index];
                        }
                    }
                }
            }
        }
    }

    return permuted_latents;
}

ov::Tensor unpack_latents(const ov::Tensor& latents, size_t height, size_t width, size_t vae_scale_factor) {
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape[0], channels = latents_shape[2];

    height /= vae_scale_factor;
    width /= vae_scale_factor;

    // latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    size_t h_half = height / 2;
    size_t w_half = width / 2;
    size_t c_quarter = channels / 4;

    // Reshape to (batch_size, channels // (2 * 2), height, width)
    ov::Shape final_shape = {batch_size, c_quarter, height, width};
    ov::Tensor permuted_latents(latents.get_element_type(), final_shape);

    OPENVINO_ASSERT(latents.get_size() == permuted_latents.get_size(), "Incorrect target shape, tensors must have the same sizes");

    const float* src_data = latents.data<float>();
    float* dst_data = permuted_latents.data<float>();

    // Permutation to (0, 3, 1, 4, 2, 5)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c4 = 0; c4 < c_quarter; ++c4) {
            for (size_t h2 = 0; h2 < h_half; ++h2) {
                for (size_t h3 = 0; h3 < 2; ++h3) {
                    for (size_t w2 = 0; w2 < w_half; ++w2) {
                        for (size_t w3 = 0; w3 < 2; ++w3) {
                            size_t reshaped_index = (((b * h_half + h2) * w_half + w2) * c_quarter + c4) * 4 + h3 * 2 + w3;
                            size_t final_index = (b * c_quarter * height * width) + (c4 * height * width) + (h2 * 2 + h3) * width + (w2 * 2 + w3);

                            dst_data[final_index] = src_data[reshaped_index];
                        }
                    }
                }
            }
        }
    }

    return permuted_latents;
}

ov::Tensor prepare_latent_image_ids(size_t batch_size, size_t height, size_t width) {
    ov::Tensor latent_image_ids(ov::element::f32, {height * width, 3});
    auto* data = latent_image_ids.data<float>();

    std::fill(data, data + height * width * 3, 0.0);

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            data[(i * width + j) * 3 + 1] = static_cast<float>(i);
            data[(i * width + j) * 3 + 2] = static_cast<float>(j);
        }
    }

    return latent_image_ids;
}

}  // namespace

namespace ov {
namespace genai {

class FluxPipeline : public DiffusionPipeline {
public:
    FluxPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) : DiffusionPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string t5_text_encoder = data["text_encoder_2"][1].get<std::string>();
        if (t5_text_encoder == "T5EncoderModel") {
            m_t5_text_encoder = std::make_shared<T5EncoderModel>(root_dir / "text_encoder_2");
        } else {
            OPENVINO_THROW("Unsupported '", t5_text_encoder, "' text encoder type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder");
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "FluxTransformer2DModel") {
            m_transformer = std::make_shared<FluxTransformer2DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    FluxPipeline(PipelineType pipeline_type,
                 const std::filesystem::path& root_dir,
                 const std::string& device,
                 const ov::AnyMap& properties)
        : DiffusionPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir / "text_encoder", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string t5_text_encoder = data["text_encoder_2"][1].get<std::string>();
        if (t5_text_encoder == "T5EncoderModel") {
            m_t5_text_encoder = std::make_shared<T5EncoderModel>(root_dir / "text_encoder_2", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", t5_text_encoder, "' text encoder type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, properties);
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder", device, properties);
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "FluxTransformer2DModel") {
            m_transformer = std::make_shared<FluxTransformer2DModel>(root_dir / "transformer", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    FluxPipeline(PipelineType pipeline_type,
                 const CLIPTextModel& clip_text_model,
                 const T5EncoderModel& t5_text_model,
                 const FluxTransformer2DModel& transformer,
                 const AutoencoderKL& vae)
        : DiffusionPipeline(pipeline_type),
          m_clip_text_encoder(std::make_shared<CLIPTextModel>(clip_text_model)),
          m_t5_text_encoder(std::make_shared<T5EncoderModel>(t5_text_model)),
          m_vae(std::make_shared<AutoencoderKL>(vae)),
          m_transformer(std::make_shared<FluxTransformer2DModel>(transformer)) {
        initialize_generation_config("FluxPipeline");
    }

    FluxPipeline(PipelineType pipeline_type, const FluxPipeline& pipe) :
        FluxPipeline(pipe) {
        OPENVINO_ASSERT(!pipe.is_inpainting_model(), "Cannot create ",
            pipeline_type == PipelineType::TEXT_2_IMAGE ? "'Text2ImagePipeline'" : "'Image2ImagePipeline'", " from InpaintingPipeline with inpainting model");

        m_pipeline_type = pipeline_type;
        initialize_generation_config("FluxPipeline");
    }

    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        check_image_size(height, width);

        m_clip_text_encoder->reshape(1);
        m_t5_text_encoder->reshape(1, m_generation_config.max_sequence_length);
        m_transformer->reshape(num_images_per_prompt, height, width, m_generation_config.max_sequence_length);

        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        m_clip_text_encoder->compile(device, properties);
        m_t5_text_encoder->compile(device, properties);
        m_vae->compile(device, properties);
        m_transformer->compile(device, properties);
    }
    
    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        float infer_duration;
        // encode_prompt
        std::string prompt_2_str = generation_config.prompt_2 != std::nullopt ? *generation_config.prompt_2 : positive_prompt;

        m_clip_text_encoder->infer(positive_prompt, {}, false, infer_duration);
        m_perf_metrics.encoder_inference_duration["text_encoder"] = infer_duration / 1000.0f;
        ov::Tensor pooled_prompt_embeds = m_clip_text_encoder->get_output_tensor(1);
        ov::Tensor prompt_embeds = m_t5_text_encoder->infer(prompt_2_str, "", false, generation_config.max_sequence_length, infer_duration);
        m_perf_metrics.encoder_inference_duration["text_encoder_2"] = infer_duration / 1000.0f;

        pooled_prompt_embeds = numpy_utils::repeat(pooled_prompt_embeds, generation_config.num_images_per_prompt);
        prompt_embeds = numpy_utils::repeat(prompt_embeds, generation_config.num_images_per_prompt);

        // text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        ov::Shape text_ids_shape = {prompt_embeds.get_shape()[1], 3};
        ov::Tensor text_ids(ov::element::f32, text_ids_shape);
        std::fill_n(text_ids.data<float>(), text_ids.get_size(), 0.0f);

        const size_t num_channels_latents = m_transformer->get_config().in_channels / 4;
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        size_t height = generation_config.height / vae_scale_factor;
        size_t width = generation_config.width / vae_scale_factor;

        ov::Tensor latent_image_ids = prepare_latent_image_ids(generation_config.num_images_per_prompt, height / 2, width / 2);

        if (m_transformer->get_config().guidance_embeds) {
            ov::Tensor guidance = ov::Tensor(ov::element::f32, {generation_config.num_images_per_prompt});
            std::fill_n(guidance.data<float>(), guidance.get_size(), static_cast<float>(generation_config.guidance_scale));
            m_transformer->set_hidden_states("guidance", guidance);
        }

        m_transformer->set_hidden_states("pooled_projections", pooled_prompt_embeds);
        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds);
        m_transformer->set_hidden_states("txt_ids", text_ids);
        m_transformer->set_hidden_states("img_ids", latent_image_ids);
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) const override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        size_t num_channels_latents = m_transformer->get_config().in_channels / 4;
        size_t height = generation_config.height / vae_scale_factor;
        size_t width = generation_config.width / vae_scale_factor;

        ov::Shape latent_shape{generation_config.num_images_per_prompt,
                               num_channels_latents,
                               height,
                               width};
        ov::Tensor latent(ov::element::f32, {}), proccesed_image, image_latent, noise;

        if (initial_image) {
            OPENVINO_THROW("StableDiffusion3 image to image is not implemented");
        } else {
            noise = generation_config.generator->randn_tensor(latent_shape);
            latent = pack_latents(noise, generation_config.num_images_per_prompt, num_channels_latents, height, width);
        }

        return std::make_tuple(latent, proccesed_image, image_latent, noise);
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) override {
        OPENVINO_THROW("LORA adapters are not implemented for FLUX pipeline yet");
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        float infer_duration;
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

        ov::Tensor latents, processed_image, image_latent, noise;
        std::tie(latents, processed_image, image_latent, noise) = prepare_latents(initial_image, m_custom_generation_config);

        size_t image_seq_len = latents.get_shape()[1];
        float mu = m_scheduler->calculate_shift(image_seq_len);

        float linspace_end = 1.0f / m_custom_generation_config.num_inference_steps;
        std::vector<float> sigmas = numpy_utils::linspace<float>(1.0f, linspace_end, m_custom_generation_config.num_inference_steps, true);

        m_scheduler->set_timesteps_with_sigma(sigmas, mu);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        // 6. Denoising loop
        ov::Tensor timestep(ov::element::f32, {1});
        float* timestep_data = timestep.data<float>();

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();
            timestep_data[0] = timesteps[inference_step] / 1000;

            ov::Tensor noise_pred_tensor = m_transformer->infer(latents, timestep, infer_duration);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

            auto scheduler_step_result = m_scheduler->step(noise_pred_tensor, latents, inference_step, m_custom_generation_config.generator);
            latents = scheduler_step_result["latent"];

            auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - gen_start);
            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
            if (callback && callback(inference_step, timesteps.size(), latents)) {
                auto image = ov::Tensor(ov::element::u8, {});
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start)
                        .count();
                return image;
            }
        }

        latents = unpack_latents(latents, m_custom_generation_config.height, m_custom_generation_config.width, vae_scale_factor);
        auto image = m_vae->decode(latents, infer_duration);
        m_perf_metrics.vae_decoder_inference_duration = infer_duration / 1000.0f;
        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        return image;
    }

    ov::Tensor decode(const ov::Tensor latent, float& infer_duration) override {
        ov::Tensor unpacked_latent = unpack_latents(latent,
                                                m_custom_generation_config.height,
                                                m_custom_generation_config.width,
                                                m_vae->get_vae_scale_factor());
        return m_vae->decode(unpacked_latent, infer_duration);
    }

    ImageGenerationPerfMetrics get_perfomance_metrics() override {
        m_perf_metrics.load_time = m_load_time_ms;
        return m_perf_metrics;
    }

private:
    bool is_inpainting_model() const {
        assert(m_transformer != nullptr);
        assert(m_vae != nullptr);
        return m_transformer->get_config().in_channels == (m_vae->get_config().latent_channels * 2 + 1);
    }

    void compute_dim(int64_t & generation_config_value, ov::Tensor initial_image, int dim_idx) {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();

        // in case of image to image generation_config_value is just ignored and computed based on initial image
        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            OPENVINO_ASSERT(initial_image, "Initial image is empty for image to image pipeline");
            ov::Shape shape = initial_image.get_shape();
            int64_t dim_val = shape[dim_idx];

            generation_config_value = dim_val - (dim_val % vae_scale_factor);
        }

        if (generation_config_value < 0)
            generation_config_value = transformer_config.m_default_sample_size * vae_scale_factor;
    }

    void initialize_generation_config(const std::string& class_name) override {
        assert(m_transformer != nullptr);
        assert(m_vae != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();

        if (m_pipeline_type != PipelineType::IMAGE_2_IMAGE) {
            m_generation_config.height = transformer_config.m_default_sample_size * vae_scale_factor;
            m_generation_config.width = transformer_config.m_default_sample_size * vae_scale_factor;
        }

        if (class_name == "FluxPipeline" || class_name == "FluxImg2ImgPipeline" || class_name == "FluxInpaintPipeline" ) {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE) {
                m_generation_config.guidance_scale = 3.5f;
                m_generation_config.num_inference_steps = 28;
                m_generation_config.strength = 1.0f;
            } else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_generation_config.guidance_scale = 3.5f;
                m_generation_config.num_inference_steps = 28;
                m_generation_config.strength = 1.0f;
            }
            m_generation_config.max_sequence_length = 512;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_transformer != nullptr);
        // const size_t vae_scale_factor = m_transformer->get_vae_scale_factor();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) && (width % vae_scale_factor == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ",
                        vae_scale_factor);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.width, generation_config.height);

        OPENVINO_ASSERT(generation_config.max_sequence_length <= 512, "T5's 'max_sequence_length' must be less or equal to 512");

        OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used by FluxPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by FluxPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by FluxPipeline");
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by FluxPipeline");

        if ((m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) && initial_image) {
            ov::Shape initial_image_shape = initial_image.get_shape();
            size_t height = initial_image_shape[1], width = initial_image_shape[2];

            OPENVINO_ASSERT(generation_config.height == height,
                "Height for initial (", height, ") and generated (", generation_config.height,") images must be the same");
            OPENVINO_ASSERT(generation_config.width == width,
                "Width for initial (", width, ") and generated (", generation_config.width,") images must be the same");

            OPENVINO_ASSERT(generation_config.strength >= 0.0f && generation_config.strength <= 1.0f,
                "'Strength' generation parameter must be withion [0, 1] range");
        } else {
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
        }
    }

    std::shared_ptr<FluxTransformer2DModel> m_transformer = nullptr;
    std::shared_ptr<CLIPTextModel> m_clip_text_encoder = nullptr;
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder = nullptr;
    std::shared_ptr<AutoencoderKL> m_vae = nullptr;
    ImageGenerationConfig m_custom_generation_config;
    ImageGenerationPerfMetrics m_perf_metrics;
};

}  // namespace genai
}  // namespace ov
