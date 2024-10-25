// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <ctime>

#include "text2image/diffusion_pipeline.hpp"
#include "text2image/numpy_utils.hpp"
#include "utils.hpp"

namespace {

// src - input tensor with data for padding
// res - zeros tonsor with target shape
void padding_right(const float* src, float* res, const ov::Shape src_size, const ov::Shape res_size) {
    OPENVINO_ASSERT(src_size[0] == res_size[0] && src_size[1] == res_size[1],
                    "Tensors for padding_right must have the same dimensions");

    for (size_t i = 0; i < res_size[0]; ++i) {
        for (size_t j = 0; j < res_size[1]; ++j) {
            size_t offset_1 = (i * res_size[1] + j) * res_size[2];
            size_t offset_2 = (i * src_size[1] + j) * src_size[2];

            std::memcpy(res + offset_1, src + offset_2, src_size[2] * sizeof(float));
        }
    }
}

ov::Tensor tensor_batch_copy(const ov::Tensor input, const size_t num_images_per_prompt, size_t batch_size_multiplier) {
    ov::Shape repeated_shape = input.get_shape();
    repeated_shape[0] *= num_images_per_prompt;
    ov::Tensor tensor_repeated(input.get_element_type(), repeated_shape);

    for (size_t n = 0; n < num_images_per_prompt; ++n) {
        batch_copy(input, tensor_repeated, 0, n);
    }

    return tensor_repeated;
}

}  // namespace

namespace ov {
namespace genai {

class Text2ImagePipeline::StableDiffusion3Pipeline : public Text2ImagePipeline::DiffusionPipeline {
public:
    explicit StableDiffusion3Pipeline(const std::string& root_dir) {
        const std::string model_index_path = root_dir + "/model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir + "/scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_1 = std::make_shared<CLIPTextModelWithProjection>(root_dir + "/text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string text_encoder_2 = data["text_encoder_2"][1].get<std::string>();
        if (text_encoder_2 == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_2 = std::make_shared<CLIPTextModelWithProjection>(root_dir + "/text_encoder_2");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        // TODO:
        // const std::string text_encoder_3 = data["text_encoder_3"][1].get<std::string>();
        // if (text_encoder_2 == "T5EncoderModel") {
        //     m_t5_encoder_model = std::make_shared<T5EncoderModel>(root_dir + "/text_encoder_3");
        // } else {
        //     OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        // }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            m_vae_decoder = std::make_shared<AutoencoderKL>(root_dir + "/vae_decoder");
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "SD3Transformer2DModel") {
            m_transformer = std::make_shared<SD3Transformer2DModel>(root_dir + "/transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "'Transformer type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    StableDiffusion3Pipeline(const std::string& root_dir, const std::string& device, const ov::AnyMap& properties) {
        const std::string model_index_path = root_dir + "/model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir + "/scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_1 =
                std::make_shared<CLIPTextModelWithProjection>(root_dir + "/text_encoder", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string text_encoder_2 = data["text_encoder_2"][1].get<std::string>();
        if (text_encoder_2 == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_2 =
                std::make_shared<CLIPTextModelWithProjection>(root_dir + "/text_encoder_2", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        // TODO: text_encoder_3

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            m_vae_decoder = std::make_shared<AutoencoderKL>(root_dir + "/vae_decoder", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "SD3Transformer2DModel") {
            m_transformer = std::make_shared<SD3Transformer2DModel>(root_dir + "/transformer", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "'Transformer type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    StableDiffusion3Pipeline(const CLIPTextModelWithProjection& clip_text_model_1,
                             const CLIPTextModelWithProjection& clip_text_model_2,
                             const SD3Transformer2DModel& transformer,
                             const AutoencoderKL& vae_decoder)
        : m_clip_text_encoder_1(std::make_shared<CLIPTextModelWithProjection>(clip_text_model_1)),
          m_clip_text_encoder_2(std::make_shared<CLIPTextModelWithProjection>(clip_text_model_2)),
          m_vae_decoder(std::make_shared<AutoencoderKL>(vae_decoder)),
          m_transformer(std::make_shared<SD3Transformer2DModel>(transformer)) {}

    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        check_image_size(height, width);

        const size_t batch_size_multiplier =
            do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        m_clip_text_encoder_1->reshape(batch_size_multiplier);
        m_clip_text_encoder_2->reshape(batch_size_multiplier);
        m_vae_decoder->reshape(num_images_per_prompt, height, width);
        m_transformer->reshape(num_images_per_prompt * batch_size_multiplier,
                               height,
                               width,
                               m_clip_text_encoder_1->get_config().max_position_embeddings);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        m_clip_text_encoder_1->compile(device, properties);
        m_clip_text_encoder_2->compile(device, properties);
        m_vae_decoder->compile(device, properties);
        m_transformer->compile(device, properties);
    }

    ov::Tensor generate(const std::string& positive_prompt, const ov::AnyMap& properties) override {
        using namespace numpy_utils;
        GenerationConfig generation_config = m_generation_config;
        generation_config.update_generation_config(properties);

        const auto& transformer_config = m_transformer->get_config();
        const size_t batch_size_multiplier = do_classifier_free_guidance(generation_config.guidance_scale)
                                                 ? 2
                                                 : 1;  // Transformer accepts 2x batch in case of CFG

        size_t vae_out_channels_size = m_vae_decoder->get_config().block_out_channels.size();
        const size_t vae_scale_factor = std::pow(2, vae_out_channels_size - 1);

        if (generation_config.height < 0)
            generation_config.height = transformer_config.sample_size * vae_scale_factor;
        if (generation_config.width < 0)
            generation_config.width = transformer_config.sample_size * vae_scale_factor;

        check_image_size(generation_config.height, generation_config.width);

        if (generation_config.random_generator == nullptr) {
            uint32_t seed = time(NULL);
            generation_config.random_generator = std::make_shared<CppStdGenerator>(seed);
        }

        // Input tensors for transformer model
        ov::Tensor prompt_embeds_inp, pooled_prompt_embeds_inp;

        // 1. Encode positive prompt:
        std::string prompt_2_str =
            generation_config.prompt_2 != std::nullopt ? *generation_config.prompt_2 : positive_prompt;
        std::string prompt_3_str =
            generation_config.prompt_3 != std::nullopt ? *generation_config.prompt_3 : positive_prompt;

        ov::Tensor pooled_prompt_embed_out = m_clip_text_encoder_1->infer(positive_prompt, "", false);
        size_t idx_hidden_state_1 = m_clip_text_encoder_1->get_config().num_hidden_layers + 1;
        ov::Tensor prompt_embed_out = m_clip_text_encoder_1->get_output_tensor(idx_hidden_state_1);

        ov::Tensor pooled_prompt_2_embed_out = m_clip_text_encoder_2->infer(prompt_2_str, "", false);
        size_t idx_hidden_state_2 = m_clip_text_encoder_2->get_config().num_hidden_layers + 1;
        ov::Tensor prompt_2_embed_out = m_clip_text_encoder_2->get_output_tensor(idx_hidden_state_2);

        ov::Tensor pooled_prompt_embed, prompt_embed, pooled_prompt_2_embed, prompt_2_embed;
        if (generation_config.num_images_per_prompt == 1) {
            pooled_prompt_embed = pooled_prompt_embed_out;
            prompt_embed = prompt_embed_out;
            pooled_prompt_2_embed = pooled_prompt_2_embed_out;
            prompt_2_embed = prompt_2_embed_out;
        } else {
            pooled_prompt_embed = tensor_batch_copy(pooled_prompt_embed_out,
                                                    generation_config.num_images_per_prompt,
                                                    batch_size_multiplier);
            prompt_embed =
                tensor_batch_copy(prompt_embed_out, generation_config.num_images_per_prompt, batch_size_multiplier);
            pooled_prompt_2_embed = tensor_batch_copy(pooled_prompt_2_embed_out,
                                                      generation_config.num_images_per_prompt,
                                                      batch_size_multiplier);
            prompt_2_embed =
                tensor_batch_copy(prompt_2_embed_out, generation_config.num_images_per_prompt, batch_size_multiplier);
        }

        // concatenate hidden_states from two encoders
        ov::Shape pr_emb_shape = prompt_embed.get_shape();
        ov::Shape pr_emb_2_shape = prompt_2_embed.get_shape();

        ov::Shape clip_prompt_embeds_shape = {pr_emb_shape[0], pr_emb_shape[1], pr_emb_shape[2] + pr_emb_2_shape[2]};
        ov::Tensor clip_prompt_embeds(prompt_embed.get_element_type(), clip_prompt_embeds_shape);

        const float* pr_emb_1_data = prompt_embed.data<const float>();
        const float* pr_emb_2_data = prompt_2_embed.data<const float>();
        float* clip_prompt_embeds_data = clip_prompt_embeds.data<float>();

        concat_3d_by_rows(pr_emb_1_data, pr_emb_2_data, clip_prompt_embeds_data, pr_emb_shape, pr_emb_2_shape);

        // TODO: text_encoder_3
        ov::Shape t5_prompt_embed_shape = {generation_config.num_images_per_prompt,
                                           77,  // TODO: self.tokenizer.model_max_length
                                           transformer_config.joint_attention_dim};

        std::vector<float> t5_prompt_embed(
            t5_prompt_embed_shape[0] * t5_prompt_embed_shape[1] * t5_prompt_embed_shape[2],
            0.0f);

        // padding for clip_prompt_embeds
        ov::Shape pad_embeds_shape = {clip_prompt_embeds_shape[0],
                                      clip_prompt_embeds_shape[1],
                                      t5_prompt_embed_shape[2]};

        std::vector<float> pad_embeds(pad_embeds_shape[0] * pad_embeds_shape[1] * pad_embeds_shape[2], 0.0f);
        padding_right(clip_prompt_embeds_data, pad_embeds.data(), clip_prompt_embeds_shape, pad_embeds_shape);

        // prompt_embeds = torch.cat([pad_embeds, t5_prompt_embed], dim=-2)
        ov::Shape prompt_embeds_shape = {pad_embeds_shape[0],
                                         pad_embeds_shape[1] + t5_prompt_embed_shape[1],
                                         pad_embeds_shape[2]};
        ov::Tensor prompt_embeds(ov::element::f32, prompt_embeds_shape);
        float* prompt_embeds_data = prompt_embeds.data<float>();
        concat_3d_by_cols(pad_embeds.data(),
                          t5_prompt_embed.data(),
                          prompt_embeds_data,
                          pad_embeds_shape,
                          t5_prompt_embed_shape);

        // pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        ov::Shape p_pr_emb_shape = pooled_prompt_embed.get_shape();
        ov::Shape p_pr_emb_2_shape = pooled_prompt_2_embed.get_shape();

        const float* pooled_prompt_embed_data = pooled_prompt_embed.data<float>();
        const float* pooled_prompt_2_embed_data = pooled_prompt_2_embed.data<float>();

        ov::Shape pooled_prompt_embeds_shape = {p_pr_emb_shape[0], p_pr_emb_shape[1] + p_pr_emb_2_shape[1]};
        ov::Tensor pooled_prompt_embeds(ov::element::f32, pooled_prompt_embeds_shape);
        float* pooled_prompt_embeds_data = pooled_prompt_embeds.data<float>();

        concat_2d_by_rows(pooled_prompt_embed_data,
                          pooled_prompt_2_embed_data,
                          pooled_prompt_embeds_data,
                          p_pr_emb_shape,
                          p_pr_emb_2_shape);
        // From steps above we'll use prompt_embeds and pooled_prompt_embeds tensors

        if (do_classifier_free_guidance(generation_config.guidance_scale)) {
            // 2. Encode negative prompt:
            std::string negative_prompt_1_str =
                !generation_config.negative_prompt.empty() ? generation_config.negative_prompt : "";
            std::string negative_prompt_2_str = !generation_config.negative_prompt_2.empty()
                                                    ? generation_config.negative_prompt_2
                                                    : negative_prompt_1_str;
            std::string negative_prompt_3_str = !generation_config.negative_prompt_3.empty()
                                                    ? generation_config.negative_prompt_3
                                                    : negative_prompt_1_str;

            ov::Tensor negative_pooled_prompt_embed_out =
                m_clip_text_encoder_1->infer(negative_prompt_1_str, "", false);
            ov::Tensor negative_prompt_embed_out = m_clip_text_encoder_1->get_output_tensor(idx_hidden_state_1);

            ov::Tensor negative_pooled_prompt_2_embed_out =
                m_clip_text_encoder_2->infer(negative_prompt_2_str, "", false);
            ov::Tensor negative_prompt_2_embed_out = m_clip_text_encoder_2->get_output_tensor(idx_hidden_state_2);

            ov::Tensor negative_pooled_prompt_embed, negative_prompt_embed, negative_pooled_prompt_2_embed,
                negative_prompt_2_embed;
            if (generation_config.num_images_per_prompt == 1) {
                negative_pooled_prompt_embed = negative_pooled_prompt_embed_out;
                negative_prompt_embed = negative_prompt_embed_out;
                negative_pooled_prompt_2_embed = negative_pooled_prompt_2_embed_out;
                negative_prompt_2_embed = negative_prompt_2_embed_out;
            } else {
                negative_pooled_prompt_embed = tensor_batch_copy(negative_pooled_prompt_embed_out,
                                                                 generation_config.num_images_per_prompt,
                                                                 batch_size_multiplier);
                negative_prompt_embed = tensor_batch_copy(negative_prompt_embed_out,
                                                          generation_config.num_images_per_prompt,
                                                          batch_size_multiplier);
                negative_pooled_prompt_2_embed = tensor_batch_copy(negative_pooled_prompt_2_embed_out,
                                                                   generation_config.num_images_per_prompt,
                                                                   batch_size_multiplier);
                negative_prompt_2_embed = tensor_batch_copy(negative_prompt_2_embed_out,
                                                            generation_config.num_images_per_prompt,
                                                            batch_size_multiplier);
            }

            // concatenate hidden_states from two encoders
            ov::Shape n_pr_emb_1_shape = negative_prompt_embed.get_shape();
            ov::Shape n_pr_emb_2_shape = negative_prompt_2_embed.get_shape();

            ov::Shape neg_clip_prompt_embeds_shape = {n_pr_emb_1_shape[0],
                                                      n_pr_emb_1_shape[1],
                                                      n_pr_emb_1_shape[2] + n_pr_emb_2_shape[2]};
            ov::Tensor neg_clip_prompt_embeds(prompt_embed.get_element_type(), neg_clip_prompt_embeds_shape);

            const float* neg_pr_emb_1_data = negative_prompt_embed.data<const float>();
            const float* neg_pr_emb_2_data = negative_prompt_2_embed.data<const float>();
            float* neg_clip_prompt_embeds_data = neg_clip_prompt_embeds.data<float>();

            concat_3d_by_rows(neg_pr_emb_1_data,
                              neg_pr_emb_2_data,
                              neg_clip_prompt_embeds_data,
                              n_pr_emb_1_shape,
                              n_pr_emb_2_shape);

            std::vector<float> t5_neg_prompt_embed(
                t5_prompt_embed_shape[0] * t5_prompt_embed_shape[1] * t5_prompt_embed_shape[2],
                0.0f);

            // padding for neg_clip_prompt_embeds
            ov::Shape neg_pad_embeds_shape = {neg_clip_prompt_embeds_shape[0],
                                              neg_clip_prompt_embeds_shape[1],
                                              t5_prompt_embed_shape[2]};

            std::vector<float> neg_pad_embeds(
                neg_pad_embeds_shape[0] * neg_pad_embeds_shape[1] * neg_pad_embeds_shape[2],
                0.0f);

            padding_right(neg_clip_prompt_embeds_data,
                          neg_pad_embeds.data(),
                          neg_clip_prompt_embeds_shape,
                          neg_pad_embeds_shape);

            // negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            ov::Shape neg_prompt_embeds_shape = {neg_pad_embeds_shape[0],
                                                 neg_pad_embeds_shape[1] + t5_prompt_embed_shape[1],
                                                 neg_pad_embeds_shape[2]};
            ov::Tensor neg_prompt_embeds(ov::element::f32, neg_prompt_embeds_shape);
            float* neg_prompt_embeds_data = neg_prompt_embeds.data<float>();

            concat_3d_by_cols(neg_pad_embeds.data(),
                              t5_neg_prompt_embed.data(),
                              neg_prompt_embeds_data,
                              neg_pad_embeds_shape,
                              t5_prompt_embed_shape);

            // neg_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embed, negative_pooled_prompt_2_embed],
            // dim=-1)
            ov::Shape neg_pooled_pr_emb_shape = negative_pooled_prompt_embed.get_shape();
            ov::Shape neg_pooled_pr_2_emb_shape = negative_pooled_prompt_2_embed.get_shape();

            const float* neg_pooled_pr_emb_data = negative_pooled_prompt_embed.data<float>();
            const float* neg_pooled_pr_2_emb_data = negative_pooled_prompt_2_embed.data<float>();

            ov::Shape neg_pooled_prompt_embeds_shape = {neg_pooled_pr_emb_shape[0],
                                                        neg_pooled_pr_emb_shape[1] + neg_pooled_pr_2_emb_shape[1]};
            ov::Tensor neg_pooled_prompt_embeds(ov::element::f32, neg_pooled_prompt_embeds_shape);
            float* neg_pooled_prompt_embeds_data = neg_pooled_prompt_embeds.data<float>();

            concat_2d_by_rows(neg_pooled_pr_emb_data,
                              neg_pooled_pr_2_emb_data,
                              neg_pooled_prompt_embeds_data,
                              neg_pooled_pr_emb_shape,
                              neg_pooled_pr_2_emb_shape);
            // From steps above we'll use neg_prompt_embeds and neg_pooled_prompt_embeds tensors

            // Fill in transformer inputs: concat positive and negative prompt_embeds
            ov::Shape prompt_embeds_inp_shape = {prompt_embeds_shape[0] + neg_prompt_embeds_shape[0],
                                                 prompt_embeds_shape[1],
                                                 prompt_embeds_shape[2]};
            prompt_embeds_inp = ov::Tensor(ov::element::f32, prompt_embeds_inp_shape);
            float* prompt_embeds_inp_data = prompt_embeds_inp.data<float>();
            concat_3d_by_channels(neg_prompt_embeds_data,
                                  prompt_embeds_data,
                                  prompt_embeds_inp_data,
                                  neg_prompt_embeds_shape,
                                  prompt_embeds_shape);

            ov::Shape pooled_prompt_embeds_inp_shape = {
                neg_pooled_prompt_embeds_shape[0] + pooled_prompt_embeds_shape[0],
                pooled_prompt_embeds_shape[1]};

            pooled_prompt_embeds_inp = ov::Tensor(ov::element::f32, pooled_prompt_embeds_inp_shape);
            float* pooled_prompt_embeds_input_data = pooled_prompt_embeds_inp.data<float>();
            concat_2d_by_channels(neg_pooled_prompt_embeds_data,
                                  pooled_prompt_embeds_data,
                                  pooled_prompt_embeds_input_data,
                                  neg_pooled_prompt_embeds_shape,
                                  pooled_prompt_embeds_shape);
        } else {
            // Fill in transformer inputs
            prompt_embeds_inp = prompt_embeds;
            pooled_prompt_embeds_inp = pooled_prompt_embeds;
        }

        // 3. Prepare timesteps
        m_scheduler->set_timesteps(generation_config.num_inference_steps);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        // 4. Set model inputs
        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds_inp);
        m_transformer->set_hidden_states("pooled_projections", pooled_prompt_embeds_inp);

        // 5. Prepare latent variables
        size_t num_channels_latents = m_transformer->get_config().in_channels;
        ov::Shape latent_shape{generation_config.num_images_per_prompt,
                               num_channels_latents,
                               generation_config.height / vae_scale_factor,
                               generation_config.width / vae_scale_factor};

        ov::Shape latent_shape_cfg = latent_shape;
        latent_shape_cfg[0] *= batch_size_multiplier;

        ov::Tensor latent(ov::element::f32, latent_shape), latent_cfg(ov::element::f32, latent_shape_cfg);
        std::generate_n(latent.data<float>(), latent.get_size(), [&]() -> float {
            return generation_config.random_generator->next() * m_scheduler->get_init_noise_sigma();
        });

        // 6. Denoising loop
        ov::Tensor noisy_residual_tensor(ov::element::f32, {});
        ov::Tensor timestep;

        for (size_t inference_step = 0; inference_step < generation_config.num_inference_steps; ++inference_step) {
            // concat the same latent twice along a batch dimension in case of CFG
            if (batch_size_multiplier > 1) {
                batch_copy(latent, latent_cfg, 0, 0, generation_config.num_images_per_prompt);
                batch_copy(latent,
                           latent_cfg,
                           0,
                           generation_config.num_images_per_prompt,
                           generation_config.num_images_per_prompt);

                size_t timestep_size = generation_config.num_images_per_prompt * batch_size_multiplier;
                timestep = ov::Tensor(ov::element::f32, {timestep_size});
                float* timestep_data = timestep.data<float>();

                for (size_t i = 0; i < timestep_size; ++i) {
                    timestep_data[i] = timesteps[inference_step];
                }
            } else {
                // just assign to save memory copy
                latent_cfg = latent;
                timestep = ov::Tensor(ov::element::f32, {1}, &timesteps[inference_step]);
            }

            ov::Tensor noise_pred_tensor = m_transformer->infer(latent_cfg, timestep);

            ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
            noise_pred_shape[0] /= batch_size_multiplier;
            noisy_residual_tensor.set_shape(noise_pred_shape);

            if (batch_size_multiplier > 1) {
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

            auto scheduler_step_result = m_scheduler->step(noisy_residual_tensor, latent, inference_step);
            latent = scheduler_step_result["latent"];
        }

        float* latent_data = latent.data<float>();
        for (size_t i = 0; i < latent.get_size(); ++i) {
            latent_data[i] = (latent_data[i] / m_vae_decoder->get_config().scaling_factor) +
                             m_vae_decoder->get_config().shift_factor;
        }

        return m_vae_decoder->infer(latent);
    }

private:
    bool do_classifier_free_guidance(float guidance_scale) const {
        return guidance_scale > 1.0;
    }

    void initialize_generation_config(const std::string& class_name) override {
        assert(m_transformer != nullptr);
        const auto& transformer_config = m_transformer->get_config();

        size_t vae_out_channels_size = m_vae_decoder->get_config().block_out_channels.size();
        const size_t vae_scale_factor = std::pow(2, vae_out_channels_size - 1);
        m_transformer->set_vae_scale_factor(vae_scale_factor);

        m_generation_config.height = transformer_config.sample_size * vae_scale_factor;
        m_generation_config.width = transformer_config.sample_size * vae_scale_factor;

        if (class_name == "StableDiffusion3Pipeline") {
            m_generation_config.guidance_scale = 7.5f;
            m_generation_config.num_inference_steps = 50;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_transformer != nullptr);
        const size_t vae_scale_factor = m_transformer->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) && (width % vae_scale_factor == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by",
                        vae_scale_factor);
    }

    void check_inputs(const GenerationConfig& generation_config) const override {
        check_image_size(generation_config.width, generation_config.height);

        const bool is_classifier_free_guidance = do_classifier_free_guidance(generation_config.guidance_scale);
        const char* const pipeline_name = "Stable Diffusion 3";

        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt.empty(),
                        "Negative prompt is not used when guidance scale <= 1.0");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt_2.empty(),
                        "Negative prompt 2 is not used when guidance scale <= 1.0");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt_3.empty(),
                        "Negative prompt 3 is not used when guidance scale <= 1.0");
    }

    std::shared_ptr<SD3Transformer2DModel> m_transformer;
    std::shared_ptr<CLIPTextModelWithProjection> m_clip_text_encoder_1;
    std::shared_ptr<CLIPTextModelWithProjection> m_clip_text_encoder_2;
    // TODO:
    // std::shared_ptr<T5EncoderModel> m_t5_encoder_model;
    std::shared_ptr<AutoencoderKL> m_vae_decoder;
};

}  // namespace genai
}  // namespace ov
