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

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/runtime/core.hpp"

#include "json_utils.hpp"
#include "lora_helper.hpp"
#include "debug_utils.hpp"
#include "numpy_utils.hpp"

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
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
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
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
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

    ov::Tensor prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) const override {
        using namespace numpy_utils;
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const bool is_strength_max = generation_config.strength == 1.0f && m_pipeline_type == PipelineType::INPAINTING;

        ov::Shape latent_shape{generation_config.num_images_per_prompt, m_vae->get_config().latent_channels,
                               generation_config.height / vae_scale_factor, generation_config.width / vae_scale_factor};
        ov::Tensor latent;

        if (initial_image && !is_strength_max) {
            latent = m_vae->encode(initial_image, generation_config.generator);
            if (generation_config.num_images_per_prompt > 1) {
                ov::Tensor batched_latent(ov::element::f32, latent_shape);
                for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                    batch_copy(latent, batched_latent, 0, n);
                }
                latent = batched_latent;
            }
            m_scheduler->add_noise(latent, generation_config.generator);
        } else {
            latent = generation_config.generator->randn_tensor(latent_shape);

            // // latents are multiplied by 'init_noise_sigma'
            // float * latent_data = latent.data<float>();
            // for (size_t i = 0; i < latent.get_size(); ++i)
            //     latent_data[i] *= m_scheduler->get_init_noise_sigma();
        }

        return latent;
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
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
        bool is_inpainting_model = unet_config.in_channels == 9;

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
                batch_copy(encoder_hidden_states, encoder_hidden_states_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    batch_copy(encoder_hidden_states, encoder_hidden_states_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states_repeated);
        }

        if (unet_config.time_cond_proj_dim >= 0) { // LCM
            ov::Tensor timestep_cond = get_guidance_scale_embedding(generation_config.guidance_scale - 1.0f, unet_config.time_cond_proj_dim);
            m_unet->set_hidden_states("timestep_cond", timestep_cond);
        }

        m_scheduler->set_timesteps(generation_config.num_inference_steps, generation_config.strength);
        std::vector<std::int64_t> timesteps = m_scheduler->get_timesteps();

        // preparate initial latents
        // TODO: pass processed initial_image here
        ov::Tensor latent = prepare_latents(initial_image, generation_config);
        
        ov::Tensor rand_tensor(latent.get_element_type(), latent.get_shape());
        latent.copy_to(rand_tensor);

        // latents are multiplied by 'init_noise_sigma'
        float * latent_data = latent.data<float>();
        for (size_t i = 0; i < latent.get_size(); ++i)
            latent_data[i] *= m_scheduler->get_init_noise_sigma();

        read_tensor("/home/devuser/ilavreno/openvino.genai/latents.txt", latent, false);

        // prepare latents passed to models taking into account guidance scale (batch size multipler)
        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);

        ov::Tensor mask_condition, mask;
        ov::Tensor masked_image_latent(ov::element::f32, {}), image_latent;

        {
            {
                auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(mask_image.get_shape().size()));
                auto result = std::make_shared<ov::op::v0::Result>(parameter);
                auto image_processor = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter}, "mask_processor");

                ov::preprocess::PrePostProcessor ppp(image_processor);

                ppp.input().tensor()
                    .set_layout("NHWC")
                    .set_element_type(ov::element::u8);
                ppp.input().model()
                    .set_layout("NCHW");

                ppp.input().preprocess()
                    .convert_layout()
                    .convert_element_type(ov::element::f32)
                    // this is less accurate that in VaeImageProcessor::normalize
                    .scale(255.0 / 2.0)
                    .mean(1.0f);

                image_processor = ppp.build();

                ov::InferRequest image_processor_request = ov::Core().compile_model(image_processor, "CPU").create_infer_request();
                image_processor_request.set_input_tensor(initial_image);
                image_processor_request.infer();
                initial_image = image_processor_request.get_output_tensor();
            }
            
            read_tensor("/home/devuser/ilavreno/openvino.genai/init_image.txt", initial_image, false);

            {
                auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(mask_image.get_shape().size()));
                auto result = std::make_shared<ov::op::v0::Result>(parameter);
                auto mask_processor = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter}, "mask_processor");

                ov::preprocess::PrePostProcessor ppp(mask_processor);

                ppp.input().tensor()
                    .set_layout("NHWC")
                    .set_element_type(ov::element::u8)
                    .set_color_format(ov::preprocess::ColorFormat::BGR);
                ppp.input().model()
                    .set_layout("NCHW");

                ppp.input().preprocess()
                    .convert_color(ov::preprocess::ColorFormat::GRAY)
                    .convert_element_type(ov::element::f32)
                    .scale(255.0f)
                    .custom([](const ov::Output<ov::Node>& port) {
                        auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
                        auto constant_1_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 1.0f);
                        auto constant_0_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.0f);
                        auto mask_bool = std::make_shared<ov::opset15::GreaterEqual>(port, constant_0_5);
                        auto mask_float = std::make_shared<ov::opset15::Select>(mask_bool, constant_1_0, constant_0_0);
                        return mask_float;
                    })
                    .convert_layout();

                mask_processor = ppp.build();

                ov::InferRequest mask_processor_request = ov::Core().compile_model(mask_processor, "CPU").create_infer_request();
                mask_processor_request.set_input_tensor(mask_image);
                mask_processor_request.infer();
                mask_condition = mask_processor_request.get_output_tensor();
            }

            read_tensor("/home/devuser/ilavreno/openvino.genai/mask_condition.txt", mask_condition, false);

            masked_image_latent.set_shape(initial_image.get_shape());
            float * masked_image_latent_data = masked_image_latent.data<float>();
            const float * mask_condition_data = mask_condition.data<const float>();
            const float * initial_image_data = initial_image.data<const float>();

            for (size_t i = 0, plane_size = mask_condition.get_shape()[2] * mask_condition.get_shape()[3]; i < mask_condition.get_size(); ++i) {
                masked_image_latent_data[i + 0 * plane_size] = mask_condition_data[i] < 0.5f ? initial_image_data[i + 0 * plane_size] : 0.0f;
                masked_image_latent_data[i + 1 * plane_size] = mask_condition_data[i] < 0.5f ? initial_image_data[i + 1 * plane_size] : 0.0f;
                masked_image_latent_data[i + 2 * plane_size] = mask_condition_data[i] < 0.5f ? initial_image_data[i + 2 * plane_size] : 0.0f;
            }

            read_tensor("/home/devuser/ilavreno/openvino.genai/masked_image.txt", masked_image_latent, false);

            {
                auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(mask_image.get_shape().size()));
                auto result = std::make_shared<ov::op::v0::Result>(parameter);
                auto mask_latent = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter}, "mask_latent");

                ov::preprocess::PrePostProcessor ppp(mask_latent);

                size_t dst_height = mask_image.get_shape()[1] / m_vae->get_vae_scale_factor();
                size_t dst_width = mask_image.get_shape()[2] / m_vae->get_vae_scale_factor();

                ppp.input().tensor().set_layout("NCHW");
                ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST, dst_height, dst_width);

                mask_latent = ppp.build();

                ov::InferRequest mask_latent_request = ov::Core().compile_model(mask_latent, "CPU").create_infer_request();
                mask_latent_request.set_input_tensor(mask_condition);
                mask_latent_request.infer();
                mask = mask_latent_request.get_output_tensor();
            }

            mask = numpy_utils::repeat(mask, 2);
            read_tensor("/home/devuser/ilavreno/openvino.genai/mask.txt", mask, false);

            if (is_inpainting_model) {
                masked_image_latent = m_vae->encode(masked_image_latent, generation_config.generator);
                if (generation_config.num_images_per_prompt > 1) {
                    ov::Tensor batched_masked_image_latent(ov::element::f32, masked_image_latent.get_shape());
                    for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                        batch_copy(masked_image_latent, batched_masked_image_latent, 0, n);
                    }
                    masked_image_latent = batched_masked_image_latent;
                }

                masked_image_latent = numpy_utils::repeat(masked_image_latent, 2);
                read_tensor("/home/devuser/ilavreno/openvino.genai/masked_image_latents.txt", masked_image_latent, false);
            } else {
                image_latent = m_vae->encode(initial_image, generation_config.generator);
            }
        }

        ov::Tensor denoised, noisy_residual_tensor(ov::element::f32, {});
        ov::Tensor latent_model_input;

        for (size_t inference_step = 0; inference_step < timesteps.size(); inference_step++) {
            batch_copy(latent, latent_cfg, 0, 0, generation_config.num_images_per_prompt);
            // concat the same latent twice along a batch dimension in case of CFG
            if (batch_size_multiplier > 1) {
                batch_copy(latent, latent_cfg, 0, generation_config.num_images_per_prompt, generation_config.num_images_per_prompt);
            }

            {
                std::stringstream stream;
                stream << "/home/devuser/ilavreno/openvino.genai/latent_cfg_" << inference_step << ".txt";
                read_tensor(stream.str(), latent_cfg, false);
            }

            m_scheduler->scale_model_input(latent_cfg, inference_step);

            if (is_inpainting_model) {
                ov::Shape final_shape_merged{latent_shape_cfg[0], unet_config.in_channels, latent_shape_cfg[2] * latent_shape_cfg[3]};
                ov::Shape final_shape{final_shape_merged[0], final_shape_merged[1], latent_shape_cfg[2], latent_shape_cfg[3]};
                ov::Shape latent_shape_cfg_merged{latent_shape_cfg[0], latent_shape_cfg[1], latent_shape_cfg[2] * latent_shape_cfg[3]};

                ov::Tensor merged_input(ov::element::f32, final_shape);
                {
                    ov::Shape temp_shape = final_shape_merged;
                    temp_shape[1] = latent_shape_cfg[1] + mask.get_shape()[1];
                    ov::Tensor tmp(ov::element::f32, temp_shape);

                    ov::Shape mask_shape = mask.get_shape();
                    ov::Shape mask_shape_merged{mask_shape[0], mask_shape[1], mask_shape[2] * mask_shape[3]};

                    numpy_utils::concat_3d_by_cols(latent_cfg.data<float>(), mask.data<float>(), tmp.data<float>(), latent_shape_cfg_merged, mask_shape_merged);
                    numpy_utils::concat_3d_by_cols(tmp.data<float>(), masked_image_latent.data<float>(), merged_input.data<float>(), temp_shape, latent_shape_cfg_merged);
                }

                latent_model_input = merged_input;
            } else {
                std::cout << "latent_model_input = latent_cfg" << std::endl;
                latent_model_input = latent_cfg;
            }

            {
                std::stringstream stream;
                stream << "/home/devuser/ilavreno/openvino.genai/latent_model_input_" << inference_step << ".txt";
                read_tensor(stream.str(), latent_model_input, false);
            }

            std::cout << timesteps[inference_step] << std::endl;
            ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
            ov::Tensor noise_pred_tensor = m_unet->infer(latent_model_input, timestep);

            ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
            noise_pred_shape[0] /= batch_size_multiplier;
 
            {
                std::stringstream stream;
                stream << "/home/devuser/ilavreno/openvino.genai/noise_pred_" << inference_step << ".txt";
                read_tensor(stream.str(), noise_pred_tensor, false);

                std::cout << "noise_pred_tensor" << std::endl;
                for (int i = 0; i < 10; ++i) {
                    std::cout << noise_pred_tensor.data<float>()[i] << " ";
                }
                std::cout << std::endl;
            }

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

            {
                std::stringstream stream;
                stream << "/home/devuser/ilavreno/openvino.genai/before_step_" << inference_step << ".txt";
                read_tensor(stream.str(), noisy_residual_tensor, false);
            }

            auto scheduler_step_result = m_scheduler->step(noisy_residual_tensor, latent, inference_step, generation_config.generator);
            latent = scheduler_step_result["latent"];

            {
                std::stringstream stream;
                stream << "/home/devuser/ilavreno/openvino.genai/latents_after_step_" << inference_step << ".txt";
                read_tensor(stream.str(), latent, false);
            }

            if (!is_inpainting_model && m_pipeline_type == PipelineType::INPAINTING) {
                ov::Tensor init_latents_proper(image_latent.get_element_type(), {});

                if (inference_step < timesteps.size() - 1) {
                    image_latent.copy_to(init_latents_proper);

                    int64_t noise_timestep = timesteps[inference_step + 1];
                    m_scheduler->add_noise(init_latents_proper, rand_tensor, noise_timestep);

                    {
                        std::stringstream stream;
                        stream << "/home/devuser/ilavreno/openvino.genai/init_latents_proper_" << inference_step << ".txt";
                        read_tensor(stream.str(), init_latents_proper, false);
                    }
                } else {
                    init_latents_proper = image_latent;
                }

                const float * mask_data = mask.data<const float>();
                const float * init_latents_proper_data = init_latents_proper.data<const float>();
                float * latent_data = latent.data<float>();

                std::cout << "latent shape " << latent.get_shape() << std::endl;
                std::cout << "mask shape " << mask.get_shape() << std::endl;
                std::cout << "init_latents_proper shape " << init_latents_proper.get_shape() << std::endl;

                // blend initial and processed latents
                for (size_t i = 0, channel_size = mask.get_shape()[2] * mask.get_shape()[3]; i < mask.get_size(); ++i) {
                    float mask_value = mask_data[i];
                    for (size_t j = 0; j < m_vae->get_config().in_channels; ++j) {
                        latent_data[j * channel_size + i] = (1.0f - mask_value) * init_latents_proper_data[j * channel_size + i] + mask_value * latent_data[j * channel_size + i];
                    }
                }

                {
                    std::stringstream stream;
                    stream << "/home/devuser/ilavreno/openvino.genai/blend_latents_" << inference_step << ".txt";
                    read_tensor(stream.str(), latent, false);

                    std::cout << "blend latents" << std::endl;
                    for (int i = 0; i < 10; ++i) {
                        std::cout << latent.data<float>()[i] << " ";
                    }
                    std::cout << std::endl;
                }
            }

            // check whether scheduler returns "denoised" image, which should be passed to VAE decoder
            const auto it = scheduler_step_result.find("denoised");
            denoised = it != scheduler_step_result.end() ? it->second : latent;

            // if (inference_step == 1)
            //     exit(1);
        }

        {
            std::stringstream stream;
            stream << "/home/devuser/ilavreno/openvino.genai/final_latents.txt";
            read_tensor(stream.str(), denoised, false);
        }

        std::cout << "final_latents" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << denoised.data<float>()[i] << " ";
        }
        std::cout << std::endl;

        ov::Tensor image__xxx = m_vae->decode(denoised);

        {
            std::stringstream stream;
            stream << "/home/devuser/ilavreno/openvino.genai/decoded.txt";
            read_tensor(stream.str(), image__xxx, false);
        }

        return image__xxx;
    }

private:
    void initialize_generation_config(const std::string& class_name) override {
        assert(m_unet != nullptr);
        assert(m_vae != nullptr);
        const auto& unet_config = m_unet->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config.height = unet_config.sample_size * vae_scale_factor;
        m_generation_config.width = unet_config.sample_size * vae_scale_factor;

        if (class_name == "StableDiffusionPipeline" || class_name == "StableDiffusionInpaintPipeline" || class_name == "StableDiffusionInpaintPipeline") {
            m_generation_config.guidance_scale = 7.5f;
            m_generation_config.num_inference_steps = 50;
            m_generation_config.strength = m_pipeline_type == PipelineType::IMAGE_2_IMAGE ? 0.8f : 1.0f;
        } else if (class_name == "LatentConsistencyModelPipeline" || class_name == "LatentConsistencyModelImg2ImgPipeline") {
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
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
        }
    }

    friend class Text2ImagePipeline;
    friend class Image2ImagePipeline;

    std::shared_ptr<CLIPTextModel> m_clip_text_encoder;
    std::shared_ptr<UNet2DConditionModel> m_unet;
    std::shared_ptr<AutoencoderKL> m_vae;
};

}  // namespace genai
}  // namespace ov
