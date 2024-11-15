// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <ctime>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"

#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"

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

ov::Tensor split_2d_by_batch(const ov::Tensor input, size_t batch_num) {
    ov::Tensor result(input.get_element_type(), {1, input.get_shape()[1]});

    size_t shift = batch_num * input.get_shape()[1];
    std::memcpy(result.data<float>(), input.data<float>() + shift, result.get_shape()[1] * sizeof(float));

    return result;
}

ov::Tensor split_3d_by_batch(const ov::Tensor input, size_t batch_num) {
    ov::Tensor result(input.get_element_type(), {1, input.get_shape()[1], input.get_shape()[2]});

    size_t shift = batch_num * input.get_shape()[1] * input.get_shape()[2];
    std::memcpy(result.data<float>(),
                input.data<float>() + shift,
                result.get_shape()[1] * input.get_shape()[2] * sizeof(float));

    return result;
}

}  // namespace

namespace ov {
namespace genai {

class StableDiffusion3Pipeline : public DiffusionPipeline {
public:
    StableDiffusion3Pipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) :
        DiffusionPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_1 = std::make_shared<CLIPTextModelWithProjection>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string text_encoder_2 = data["text_encoder_2"][1].get<std::string>();
        if (text_encoder_2 == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_2 = std::make_shared<CLIPTextModelWithProjection>(root_dir / "text_encoder_2");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder_2, "' text encoder type");
        }

        // TODO:
        // const std::string text_encoder_3 = data["text_encoder_3"][1].get<std::string>();
        // if (text_encoder_2 == "T5EncoderModel") {
        //     m_t5_encoder = std::make_shared<T5EncoderModel>(root_dir / "text_encoder_3");
        // } else {
        //     OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        // }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "SD3Transformer2DModel") {
            m_transformer = std::make_shared<SD3Transformer2DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "'Transformer type");
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

    StableDiffusion3Pipeline(PipelineType pipeline_type,
                             const std::filesystem::path& root_dir,
                             const std::string& device,
                             const ov::AnyMap& properties) :
        DiffusionPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_1 =
                std::make_shared<CLIPTextModelWithProjection>(root_dir / "text_encoder", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string text_encoder_2 = data["text_encoder_2"][1].get<std::string>();
        if (text_encoder_2 == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_2 =
                std::make_shared<CLIPTextModelWithProjection>(root_dir / "text_encoder_2", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder_2, "' text encoder type");
        }

        // TODO:
        // const std::string text_encoder_3 = data["text_encoder_3"][1].get<std::string>();
        // if (text_encoder_2 == "T5EncoderModel") {
        //     m_t5_encoder = std::make_shared<T5EncoderModel>(root_dir / "text_encoder_3", device, properties);
        // } else {
        //     OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        // }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "SD3Transformer2DModel") {
            m_transformer = std::make_shared<SD3Transformer2DModel>(root_dir / "transformer", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "'Transformer type");
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

    StableDiffusion3Pipeline(PipelineType pipeline_type,
                             const CLIPTextModelWithProjection& clip_text_model_1,
                             const CLIPTextModelWithProjection& clip_text_model_2,
                             const SD3Transformer2DModel& transformer,
                             const AutoencoderKL& vae)
        : DiffusionPipeline(pipeline_type),
          m_clip_text_encoder_1(std::make_shared<CLIPTextModelWithProjection>(clip_text_model_1)),
          m_clip_text_encoder_2(std::make_shared<CLIPTextModelWithProjection>(clip_text_model_2)),
          m_vae(std::make_shared<AutoencoderKL>(vae)),
          m_transformer(std::make_shared<SD3Transformer2DModel>(transformer)) {
        initialize_generation_config("StableDiffusion3Pipeline");
    }

    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        check_image_size(height, width);

        const size_t batch_size_multiplier =
            do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Transformer accepts 2x batch in case of CFG
        m_clip_text_encoder_1->reshape(batch_size_multiplier);
        m_clip_text_encoder_2->reshape(batch_size_multiplier);
        m_transformer->reshape(num_images_per_prompt * batch_size_multiplier,
                               height,
                               width,
                               m_clip_text_encoder_1->get_config().max_position_embeddings);
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);

        m_clip_text_encoder_1->compile(device, properties);
        m_clip_text_encoder_2->compile(device, properties);
        m_transformer->compile(device, properties);
        m_vae->compile(device, properties);
    }

    ov::Tensor prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) const override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        ov::Shape latent_shape{generation_config.num_images_per_prompt,
                               m_transformer->get_config().in_channels,
                               generation_config.height / vae_scale_factor,
                               generation_config.width / vae_scale_factor};

        ov::Tensor latent(ov::element::f32, {});

        if (initial_image) {
            OPENVINO_THROW("StableDiffusion3 image to image is not implemented");
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

        const auto& transformer_config = m_transformer->get_config();
        const size_t batch_size_multiplier = do_classifier_free_guidance(generation_config.guidance_scale)
                                                 ? 2
                                                 : 1;  // Transformer accepts 2x batch in case of CFG

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        if (generation_config.height < 0)
            generation_config.height = transformer_config.sample_size * vae_scale_factor;
        if (generation_config.width < 0)
            generation_config.width = transformer_config.sample_size * vae_scale_factor;

        check_inputs(generation_config, initial_image);

        if (generation_config.generator == nullptr) {
            uint32_t seed = time(NULL);
            generation_config.generator = std::make_shared<CppStdGenerator>(seed);
        }

        // Input tensors for transformer model
        ov::Tensor prompt_embeds_inp, pooled_prompt_embeds_inp;

        // 1. Encode positive prompt:
        std::string prompt_2_str =
            generation_config.prompt_2 != std::nullopt ? *generation_config.prompt_2 : positive_prompt;
        std::string prompt_3_str =
            generation_config.prompt_3 != std::nullopt ? *generation_config.prompt_3 : positive_prompt;

        std::string negative_prompt_1_str = generation_config.negative_prompt != std::nullopt
                                                ? *generation_config.negative_prompt
                                                : std::string{};
        std::string negative_prompt_2_str = generation_config.negative_prompt_2 != std::nullopt
                                                ? *generation_config.negative_prompt_2
                                                : negative_prompt_1_str;
        std::string negative_prompt_3_str = generation_config.negative_prompt_3 != std::nullopt
                                                ? *generation_config.negative_prompt_3
                                                : negative_prompt_1_str;

        // text_encoder_1_output - stores positive and negative pooled_prompt_embeds
        ov::Tensor text_encoder_1_output =
            m_clip_text_encoder_1->infer(positive_prompt,
                                         negative_prompt_1_str,
                                         do_classifier_free_guidance(generation_config.guidance_scale));

        // text_encoder_1_hidden_state - stores positive and negative prompt_embeds
        size_t idx_hidden_state_1 = m_clip_text_encoder_1->get_config().num_hidden_layers + 1;
        ov::Tensor text_encoder_1_hidden_state = m_clip_text_encoder_1->get_output_tensor(idx_hidden_state_1);

        // text_encoder_2_output - stores positive and negative pooled_prompt_2_embeds
        ov::Tensor text_encoder_2_output =
            m_clip_text_encoder_2->infer(prompt_2_str,
                                         negative_prompt_2_str,
                                         do_classifier_free_guidance(generation_config.guidance_scale));

        // text_encoder_2_hidden_state - stores positive and negative prompt_2_embeds
        size_t idx_hidden_state_2 = m_clip_text_encoder_2->get_config().num_hidden_layers + 1;
        ov::Tensor text_encoder_2_hidden_state = m_clip_text_encoder_2->get_output_tensor(idx_hidden_state_2);
        // get positive prompt_2_embed_out

        ov::Tensor pooled_prompt_embed_out, prompt_embed_out, pooled_prompt_2_embed_out, prompt_2_embed_out;

        if (do_classifier_free_guidance(generation_config.guidance_scale)) {
            pooled_prompt_embed_out = split_2d_by_batch(text_encoder_1_output, 1);
            prompt_embed_out = split_3d_by_batch(text_encoder_1_hidden_state, 1);
            pooled_prompt_2_embed_out = split_2d_by_batch(text_encoder_2_output, 1);
            prompt_2_embed_out = split_3d_by_batch(text_encoder_2_hidden_state, 1);
        } else {
            pooled_prompt_embed_out = text_encoder_1_output;
            prompt_embed_out = text_encoder_1_hidden_state;
            pooled_prompt_2_embed_out = text_encoder_2_output;
            prompt_2_embed_out = text_encoder_2_hidden_state;
        }

        ov::Tensor pooled_prompt_embed, prompt_embed, pooled_prompt_2_embed, prompt_2_embed;
        if (generation_config.num_images_per_prompt == 1) {
            pooled_prompt_embed = pooled_prompt_embed_out;
            prompt_embed = prompt_embed_out;
            pooled_prompt_2_embed = pooled_prompt_2_embed_out;
            prompt_2_embed = prompt_2_embed_out;
        } else {
            pooled_prompt_embed = repeat(pooled_prompt_embed_out, generation_config.num_images_per_prompt);
            prompt_embed = repeat(prompt_embed_out, generation_config.num_images_per_prompt);
            pooled_prompt_2_embed = repeat(pooled_prompt_2_embed_out, generation_config.num_images_per_prompt);
            prompt_2_embed = repeat(prompt_2_embed_out, generation_config.num_images_per_prompt);
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
                                           m_clip_text_encoder_1->get_config().max_position_embeddings,
                                           transformer_config.joint_attention_dim};

        std::vector<float> t5_prompt_embed(ov::shape_size(t5_prompt_embed_shape), 0.0f);

        // padding for clip_prompt_embeds
        ov::Shape pad_embeds_shape = {clip_prompt_embeds_shape[0],
                                      clip_prompt_embeds_shape[1],
                                      t5_prompt_embed_shape[2]};

        std::vector<float> pad_embeds(ov::shape_size(pad_embeds_shape), 0.0f);
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

            ov::Tensor negative_pooled_prompt_embed_out = split_2d_by_batch(text_encoder_1_output, 0);
            ov::Tensor negative_prompt_embed_out = split_3d_by_batch(text_encoder_1_hidden_state, 0);
            ov::Tensor negative_pooled_prompt_2_embed_out = split_2d_by_batch(text_encoder_2_output, 0);
            ov::Tensor negative_prompt_2_embed_out = split_3d_by_batch(text_encoder_2_hidden_state, 0);

            ov::Tensor negative_pooled_prompt_embed, negative_prompt_embed, negative_pooled_prompt_2_embed,
                negative_prompt_2_embed;
            if (generation_config.num_images_per_prompt == 1) {
                negative_pooled_prompt_embed = negative_pooled_prompt_embed_out;
                negative_prompt_embed = negative_prompt_embed_out;
                negative_pooled_prompt_2_embed = negative_pooled_prompt_2_embed_out;
                negative_prompt_2_embed = negative_prompt_2_embed_out;
            } else {
                negative_pooled_prompt_embed = repeat(negative_pooled_prompt_embed_out, generation_config.num_images_per_prompt);
                negative_prompt_embed = repeat(negative_prompt_embed_out, generation_config.num_images_per_prompt);
                negative_pooled_prompt_2_embed = repeat(negative_pooled_prompt_2_embed_out, generation_config.num_images_per_prompt);
                negative_prompt_2_embed = repeat(negative_prompt_2_embed_out, generation_config.num_images_per_prompt);
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
        m_scheduler->set_timesteps(generation_config.num_inference_steps, generation_config.strength);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        // 4. Set model inputs
        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds_inp);
        m_transformer->set_hidden_states("pooled_projections", pooled_prompt_embeds_inp);

        // 5. Prepare latent variables
        ov::Tensor latent = prepare_latents(initial_image, generation_config);

        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);

        // 6. Denoising loop
        ov::Tensor noisy_residual_tensor(ov::element::f32, {});

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            // concat the same latent twice along a batch dimension in case of CFG
            if (batch_size_multiplier > 1) {
                batch_copy(latent, latent_cfg, 0, 0, generation_config.num_images_per_prompt);
                batch_copy(latent, latent_cfg, 0, generation_config.num_images_per_prompt, generation_config.num_images_per_prompt);
            } else {
                // just assign to save memory copy
                latent_cfg = latent;
            }

            ov::Tensor timestep(ov::element::f32, {1}, &timesteps[inference_step]);
            ov::Tensor noise_pred_tensor = m_transformer->infer(latent_cfg, timestep);

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
        }

        return m_vae->decode(latent);
    }

private:
    bool do_classifier_free_guidance(float guidance_scale) const {
        return guidance_scale > 1.0;
    }

    void initialize_generation_config(const std::string& class_name) override {
        assert(m_transformer != nullptr);
        assert(m_vae != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config.height = transformer_config.sample_size * vae_scale_factor;
        m_generation_config.width = transformer_config.sample_size * vae_scale_factor;

        if (class_name == "StableDiffusion3Pipeline") {
            m_generation_config.guidance_scale = 7.0f;
            m_generation_config.num_inference_steps = 28;
            m_generation_config.max_sequence_length = 256;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_transformer != nullptr);
        assert(m_vae != nullptr);

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t patch_size = m_transformer->get_config().patch_size;

        OPENVINO_ASSERT((height % (vae_scale_factor * patch_size) == 0 || height < 0) &&
                            (width % (vae_scale_factor * patch_size) == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by",
                        vae_scale_factor);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.width, generation_config.height);

        const bool is_classifier_free_guidance = do_classifier_free_guidance(generation_config.guidance_scale);

        OPENVINO_ASSERT(generation_config.max_sequence_length < 512, "T5's 'max_sequence_length' must be less than 512");
        OPENVINO_ASSERT(
            generation_config.prompt_3 == std::nullopt || generation_config.negative_prompt_3 == std::nullopt,
            "T5Encoder is not currently supported, 'prompt_3' and 'negative_prompt_3' can't be used. Please, add "
            "support.");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt == std::nullopt,
                        "Negative prompt is not used when guidance scale < 1.0");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt_2 == std::nullopt,
                        "Negative prompt 2 is not used when guidance scale < 1.0");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt_3 == std::nullopt,
                        "Negative prompt 3 is not used when guidance scale < 1.0");

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
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
        }
    }

    friend class Text2ImagePipeline;
    friend class Image2ImagePipeline;

    std::shared_ptr<CLIPTextModelWithProjection> m_clip_text_encoder_1;
    std::shared_ptr<CLIPTextModelWithProjection> m_clip_text_encoder_2;
    // TODO:
    // std::shared_ptr<T5EncoderModel> m_t5_encoder;
    std::shared_ptr<SD3Transformer2DModel> m_transformer;
    std::shared_ptr<AutoencoderKL> m_vae;
};

}  // namespace genai
}  // namespace ov
