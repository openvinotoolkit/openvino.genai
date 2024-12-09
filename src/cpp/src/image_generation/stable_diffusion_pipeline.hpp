// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ctime>
#include <cassert>
#include <filesystem>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"

#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/unet2d_condition_model.hpp"

#include "json_utils.hpp"
#include "lora_helper.hpp"

namespace ov {
namespace genai {

class StableDiffusionPipeline : public DiffusionPipeline {
public:
    StableDiffusionPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) :
        DiffusionPipeline(pipeline_type) {
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

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet");
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder");
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    StableDiffusionPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties) :
        DiffusionPipeline(pipeline_type) {
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

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, properties);
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder", device, properties);
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());

        update_adapters_from_properties(properties, m_generation_config.adapters);
    }

    StableDiffusionPipeline(
        PipelineType pipeline_type,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae)
        : DiffusionPipeline(pipeline_type),
          m_clip_text_encoder(std::make_shared<CLIPTextModel>(clip_text_model)),
          m_unet(std::make_shared<UNet2DConditionModel>(unet)),
          m_vae(std::make_shared<AutoencoderKL>(vae)) {
        const bool is_lcm = m_unet->get_config().time_cond_proj_dim > 0;
        const char * const pipeline_name = is_lcm ? "LatentConsistencyModelPipeline" : "StableDiffusionPipeline";
        initialize_generation_config(pipeline_name);
    }

    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) override {
        check_image_size(height, width);

        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        m_clip_text_encoder->reshape(batch_size_multiplier);
        m_unet->reshape(num_images_per_prompt * batch_size_multiplier, height, width, m_clip_text_encoder->get_config().max_position_embeddings);
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);

        m_clip_text_encoder->compile(device, properties);
        m_unet->compile(device, properties);
        m_vae->compile(device, properties);
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        const auto& unet_config = m_unet->get_config();
        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG

        std::string negative_prompt = generation_config.negative_prompt != std::nullopt ? *generation_config.negative_prompt : std::string{};
        ov::Tensor encoder_hidden_states = m_clip_text_encoder->infer(positive_prompt, negative_prompt,
            batch_size_multiplier > 1);

        // replicate encoder hidden state to UNet model
        if (generation_config.num_images_per_prompt == 1) {
            // reuse output of text encoder directly w/o extra memory copy
            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states);
        } else {
            ov::Shape enc_shape = encoder_hidden_states.get_shape();
            enc_shape[0] *= generation_config.num_images_per_prompt;

            ov::Tensor encoder_hidden_states_repeated(encoder_hidden_states.get_element_type(), enc_shape);
            for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                numpy_utils::batch_copy(encoder_hidden_states, encoder_hidden_states_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    numpy_utils::batch_copy(encoder_hidden_states, encoder_hidden_states_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states_repeated);
        }

        if (unet_config.time_cond_proj_dim >= 0) { // LCM
            ov::Tensor timestep_cond = get_guidance_scale_embedding(generation_config.guidance_scale - 1.0f, unet_config.time_cond_proj_dim);
            m_unet->set_hidden_states("timestep_cond", timestep_cond);
        }
    }

    ov::Tensor prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) const override {
        const auto& unet_config = m_unet->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        ov::Shape latent_shape{generation_config.num_images_per_prompt, unet_config.in_channels,
                               generation_config.height / vae_scale_factor, generation_config.width / vae_scale_factor};
        ov::Tensor latent;

        if (initial_image) {
            latent = m_vae->encode(initial_image, generation_config.generator);
            if (generation_config.num_images_per_prompt > 1) {
                ov::Tensor batched_latent(ov::element::f32, latent_shape);
                for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                    numpy_utils::batch_copy(latent, batched_latent, 0, n);
                }
                latent = batched_latent;
            }
            m_scheduler->add_noise(latent, generation_config.generator);
        } else {
            latent = generation_config.generator->randn_tensor(latent_shape);

            // latents are multiplied by 'init_noise_sigma'
            float * latent_data = latent.data<float>();
            for (size_t i = 0; i < latent.get_size(); ++i)
                latent_data[i] *= m_scheduler->get_init_noise_sigma();
        }

        return latent;
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        const ov::AnyMap& properties) override {
        using namespace numpy_utils;
        ImageGenerationConfig generation_config = m_generation_config;
        generation_config.update_generation_config(properties);

        if (!initial_image) {
            // in case of typical text to image generation, we need to ignore 'strength'
            generation_config.strength = 1.0f;
        }

        // Stable Diffusion pipeline
        // see https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline

        const auto& unet_config = m_unet->get_config();
        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        if (generation_config.height < 0)
            generation_config.height = unet_config.sample_size * vae_scale_factor;
        if (generation_config.width < 0)
            generation_config.width = unet_config.sample_size * vae_scale_factor;
        check_inputs(generation_config, initial_image);

        m_clip_text_encoder->set_adapters(generation_config.adapters);
        m_unet->set_adapters(generation_config.adapters);

        if (generation_config.generator == nullptr) {
            uint32_t seed = time(NULL);
            generation_config.generator = std::make_shared<CppStdGenerator>(seed);
        }

        m_scheduler->set_timesteps(generation_config.num_inference_steps, generation_config.strength);
        std::vector<std::int64_t> timesteps = m_scheduler->get_timesteps();

        // compute text encoders and set hidden states
        compute_hidden_states(positive_prompt, generation_config);

        // preparate initial latents
        ov::Tensor latent = prepare_latents(initial_image, generation_config);

        // prepare latents passed to models taking into account guidance scale (batch size multipler)
        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);

        // use callback if defined
        std::function<bool(size_t, ov::Tensor&)> callback;
        auto callback_iter = properties.find(ov::genai::callback.name());
        bool do_callback = callback_iter != properties.end();
        if (do_callback) {
            callback = callback_iter->second.as<std::function<bool(size_t, ov::Tensor&)>>();
        }

        ov::Tensor denoised, noisy_residual_tensor(ov::element::f32, {});
        for (size_t inference_step = 0; inference_step < timesteps.size(); inference_step++) {
            numpy_utils::batch_copy(latent, latent_cfg, 0, 0, generation_config.num_images_per_prompt);
            // concat the same latent twice along a batch dimension in case of CFG
            if (batch_size_multiplier > 1) {
                numpy_utils::batch_copy(latent, latent_cfg, 0, generation_config.num_images_per_prompt, generation_config.num_images_per_prompt);
            }

            m_scheduler->scale_model_input(latent_cfg, inference_step);

            ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
            ov::Tensor noise_pred_tensor = m_unet->infer(latent_cfg, timestep);

            ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
            noise_pred_shape[0] /= batch_size_multiplier;

            if (batch_size_multiplier > 1) {
                noisy_residual_tensor.set_shape(noise_pred_shape);

                // perform guidance
                float* noisy_residual = noisy_residual_tensor.data<float>();
                const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
                const float* noise_pred_text = noise_pred_uncond + noisy_residual_tensor.get_size();

                for (size_t i = 0; i < noisy_residual_tensor.get_size(); ++i) {
                    noisy_residual[i] = noise_pred_uncond[i] +
                        generation_config.guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);
                }
            } else {
                noisy_residual_tensor = noise_pred_tensor;
            }

            auto scheduler_step_result = m_scheduler->step(noisy_residual_tensor, latent, inference_step, generation_config.generator);
            latent = scheduler_step_result["latent"];

            // check whether scheduler returns "denoised" image, which should be passed to VAE decoder
            const auto it = scheduler_step_result.find("denoised");
            denoised = it != scheduler_step_result.end() ? it->second : latent;

            if (do_callback) {
                if (callback(inference_step, denoised)) {
                    return ov::Tensor(ov::element::u8, {});
                }
            }
        }

        return decode(denoised);
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        return m_vae->decode(latent);
    }

private:
    void initialize_generation_config(const std::string& class_name) override {
        assert(m_unet != nullptr);
        assert(m_vae != nullptr);
        const auto& unet_config = m_unet->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config.height = unet_config.sample_size * vae_scale_factor;
        m_generation_config.width = unet_config.sample_size * vae_scale_factor;

        if (class_name == "StableDiffusionPipeline") {
            m_generation_config.guidance_scale = 7.5f;
            m_generation_config.num_inference_steps = 50;
            m_generation_config.strength = m_pipeline_type == PipelineType::IMAGE_2_IMAGE ? 0.8f : 1.0f;
        } else if (class_name == "LatentConsistencyModelPipeline") {
            m_generation_config.guidance_scale = 8.5f;
            m_generation_config.num_inference_steps = 4;
            m_generation_config.strength = m_pipeline_type == PipelineType::IMAGE_2_IMAGE ? 0.8f : 1.0f;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_vae != nullptr);
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) &&
            (width % vae_scale_factor == 0 || width < 0), "Both 'width' and 'height' must be divisible by",
            vae_scale_factor);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.width, generation_config.height);

        const bool is_classifier_free_guidance = m_unet->do_classifier_free_guidance(generation_config.guidance_scale);
        const bool is_lcm = m_unet->get_config().time_cond_proj_dim > 0;
        const char * const pipeline_name = is_lcm ? "Latent Consistency Model" : "Stable Diffusion";

        OPENVINO_ASSERT(generation_config.prompt_2 == std::nullopt, "Prompt 2 is not used by ", pipeline_name);
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by ", pipeline_name);
        if (is_lcm) {
            OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used by ", pipeline_name);
        } else if (!is_classifier_free_guidance) {
            OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used when guidance scale <= 1.0");
        }
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by ", pipeline_name);
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by ", pipeline_name);

        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE && initial_image) {
            ov::Shape initial_image_shape = initial_image.get_shape();
            size_t height = initial_image_shape[1], width = initial_image_shape[2];

            OPENVINO_ASSERT(generation_config.height == height,
                "Height for initial (", height, ") and generated (", generation_config.height,") images must be the same");
            OPENVINO_ASSERT(generation_config.width == width,
                "Width for initial (", width, ") and generated (", generation_config.width,") images must be the same");
            OPENVINO_ASSERT(generation_config.strength >= 0.0f && generation_config.strength <= 1.0f,
                "'Strength' generation parameter must be withion [0, 1] range");
        } else {
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
        }
    }

    friend class Text2ImagePipeline;
    friend class Image2ImagePipeline;

    std::shared_ptr<CLIPTextModel> m_clip_text_encoder = nullptr;
    std::shared_ptr<UNet2DConditionModel> m_unet = nullptr;
    std::shared_ptr<AutoencoderKL> m_vae = nullptr;
};

}  // namespace genai
}  // namespace ov
