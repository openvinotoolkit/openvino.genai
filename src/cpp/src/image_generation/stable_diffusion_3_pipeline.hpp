// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>

#include "image_generation/diffusion_pipeline.hpp"

#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"

#include "utils.hpp"
#include "lora/helper.hpp"

namespace {

// src - input tensor with data for padding
// res - tensor with target shape, remaining side will be padded with zeros
void padding_right(ov::Tensor src, ov::Tensor res) {
    const ov::Shape src_shape = src.get_shape(), res_shape = res.get_shape();
    OPENVINO_ASSERT(src_shape.size() == 3 && src_shape.size() == res_shape.size(), "Rank of tensors within 'padding_right' must be 3");
    OPENVINO_ASSERT(src_shape[0] == res_shape[0] && src_shape[1] == res_shape[1], "Tensors for padding_right must have the same dimensions");

    // since torch.nn.functional.pad can also perform trancation in case of negative pad size value
    // we need to find minimal amoung src and res and respect it
    size_t min_size = std::min(src_shape[2], res_shape[2]);

    const float* src_data = src.data<const float>();
    float* res_data = res.data<float>();

    for (size_t i = 0; i < res_shape[0]; ++i) {
        for (size_t j = 0; j < res_shape[1]; ++j) {
            size_t offset_1 = (i * res_shape[1] + j) * res_shape[2];
            size_t offset_2 = (i * src_shape[1] + j) * src_shape[2];

            std::memcpy(res_data + offset_1, src_data + offset_2, min_size * sizeof(float));
            if (res_shape[2] > src_shape[2]) {
                // peform actual padding if required
                std::fill_n(res_data + offset_1 + src_shape[2], res_shape[2] - src_shape[2], 0.0f);
            }
        }
    }
}

// returns tensor, which shares data with input tensor and pointing to a given batch slice
ov::Tensor get_tensor_batch(const ov::Tensor input, size_t batch_id) {
    ov::Shape target_shape = input.get_shape();

    OPENVINO_ASSERT(target_shape.at(0) > batch_id, "Cannot get batch with id ", batch_id, ", total batch size is ", target_shape.at(0));
    target_shape[0] = 1;

    auto target_data = input.data<float>() + batch_id * ov::shape_size(target_shape);
    ov::Tensor target_tensor(input.get_element_type(), target_shape, target_data);

    return target_tensor;
}

}  // namespace

namespace ov {
namespace genai {

class StableDiffusion3Pipeline : public DiffusionPipeline {
public:
    StableDiffusion3Pipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) :
        DiffusionPipeline(pipeline_type) {
        m_root_dir = root_dir;
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

        const auto text_encoder_3_json = data["text_encoder_3"][1];
        if (!text_encoder_3_json.is_null()) {
            const std::string text_encoder_3 = text_encoder_3_json.get<std::string>();
            if (text_encoder_3 == "T5EncoderModel") {
                m_t5_text_encoder = std::make_shared<T5EncoderModel>(root_dir / "text_encoder_3");
            } else {
                OPENVINO_THROW("Unsupported '", text_encoder_3, "' text encoder type");
            }
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "SD3Transformer2DModel") {
            m_transformer = std::make_shared<SD3Transformer2DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
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

    StableDiffusion3Pipeline(PipelineType pipeline_type,
                             const std::filesystem::path& root_dir,
                             const std::string& device,
                             const ov::AnyMap& properties) :
        DiffusionPipeline(pipeline_type) {
        m_root_dir = root_dir;
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
            m_clip_text_encoder_2 = std::make_shared<CLIPTextModelWithProjection>(root_dir / "text_encoder_2", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder_2, "' text encoder type");
        }
        const auto text_encoder_3_json = data["text_encoder_3"][1];
        if (!text_encoder_3_json.is_null()) {
            const std::string text_encoder_3 = text_encoder_3_json.get<std::string>();
            if (text_encoder_3 == "T5EncoderModel") {
                m_t5_text_encoder = std::make_shared<T5EncoderModel>(root_dir / "text_encoder_3", device, properties);
            } else {
                OPENVINO_THROW("Unsupported '", text_encoder_3, "' text encoder type");
            }
        }
        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "SD3Transformer2DModel") {
            m_transformer = std::make_shared<SD3Transformer2DModel>(root_dir / "transformer", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
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

    StableDiffusion3Pipeline(PipelineType pipeline_type,
                             const CLIPTextModelWithProjection& clip_text_model_1,
                             const CLIPTextModelWithProjection& clip_text_model_2,
                             const T5EncoderModel& t5_encoder_model,
                             const SD3Transformer2DModel& transformer,
                             const AutoencoderKL& vae)
        : DiffusionPipeline(pipeline_type) {
        m_clip_text_encoder_1 = std::make_shared<CLIPTextModelWithProjection>(clip_text_model_1);
        m_clip_text_encoder_2 = std::make_shared<CLIPTextModelWithProjection>(clip_text_model_2);
        m_t5_text_encoder = std::make_shared<T5EncoderModel>(t5_encoder_model);
        m_vae = std::make_shared<AutoencoderKL>(vae);
        m_transformer = std::make_shared<SD3Transformer2DModel>(transformer);
        initialize_generation_config("StableDiffusion3Pipeline");
    }

    StableDiffusion3Pipeline(PipelineType pipeline_type,
                             const CLIPTextModelWithProjection& clip_text_model_1,
                             const CLIPTextModelWithProjection& clip_text_model_2,
                             const SD3Transformer2DModel& transformer,
                             const AutoencoderKL& vae)
        : DiffusionPipeline(pipeline_type) {
        m_clip_text_encoder_1 = std::make_shared<CLIPTextModelWithProjection>(clip_text_model_1);
        m_clip_text_encoder_2 = std::make_shared<CLIPTextModelWithProjection>(clip_text_model_2);
        m_vae = std::make_shared<AutoencoderKL>(vae);
        m_transformer = std::make_shared<SD3Transformer2DModel>(transformer);
        initialize_generation_config("StableDiffusion3Pipeline");
    }

    StableDiffusion3Pipeline(PipelineType pipeline_type, const StableDiffusion3Pipeline& pipe) :
        DiffusionPipeline(pipeline_type) {
        OPENVINO_ASSERT(!pipe.is_inpainting_model(), "Cannot create ",
            pipeline_type == PipelineType::TEXT_2_IMAGE ? "'Text2ImagePipeline'" : "'Image2ImagePipeline'", " from InpaintingPipeline with inpainting model");

        m_root_dir = pipe.m_root_dir;

        if (pipe.m_t5_text_encoder) {
            m_t5_text_encoder = std::make_shared<T5EncoderModel>(*pipe.m_t5_text_encoder);
        }

        m_clip_text_encoder_1 = std::make_shared<CLIPTextModelWithProjection>(*pipe.m_clip_text_encoder_1);
        m_clip_text_encoder_2 = std::make_shared<CLIPTextModelWithProjection>(*pipe.m_clip_text_encoder_2);
        m_transformer = std::make_shared<SD3Transformer2DModel>(*pipe.m_transformer);
        m_vae = std::make_shared<AutoencoderKL>(*pipe.m_vae);

        // initialize generation config

        m_pipeline_type = pipeline_type;
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

        int transformer_tokenizer_max_length = m_clip_text_encoder_1->get_config().max_position_embeddings;
        if (m_t5_text_encoder) {
            m_t5_text_encoder->reshape(batch_size_multiplier, m_generation_config.max_sequence_length);
            transformer_tokenizer_max_length += m_generation_config.max_sequence_length;
        }
        else {
            transformer_tokenizer_max_length *= 2;
        }

        m_transformer->reshape(num_images_per_prompt * batch_size_multiplier,
                               height,
                               width,
                               transformer_tokenizer_max_length);
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);

        m_clip_text_encoder_1->compile(text_encode_device, properties);
        m_clip_text_encoder_2->compile(text_encode_device, properties);
        if (m_t5_text_encoder) {
            m_t5_text_encoder->compile(text_encode_device, properties);
        }
        m_transformer->compile(denoise_device, properties);
        m_vae->compile(vae_device, properties);
    }

    std::shared_ptr<DiffusionPipeline> clone() override {
        OPENVINO_ASSERT(!m_root_dir.empty(), "Cannot clone pipeline without root directory");

        std::shared_ptr<AutoencoderKL> vae = std::make_shared<AutoencoderKL>(m_vae->clone());
        std::shared_ptr<CLIPTextModelWithProjection> clip_text_encoder_1 = std::static_pointer_cast<CLIPTextModelWithProjection>(m_clip_text_encoder_1->clone());
        std::shared_ptr<CLIPTextModelWithProjection> clip_text_encoder_2 = std::static_pointer_cast<CLIPTextModelWithProjection>(m_clip_text_encoder_2->clone());
        std::shared_ptr<SD3Transformer2DModel> transformer = std::make_shared<SD3Transformer2DModel>(m_transformer->clone());

        std::shared_ptr<StableDiffusion3Pipeline> pipeline;
        if (m_t5_text_encoder) {
            std::shared_ptr<T5EncoderModel> t5_text_encoder = m_t5_text_encoder->clone();
            pipeline = std::make_shared<StableDiffusion3Pipeline>(m_pipeline_type,
                                                                  *clip_text_encoder_1,
                                                                  *clip_text_encoder_2,
                                                                  *t5_text_encoder,
                                                                  *transformer,
                                                                  *vae);
        } else {
            pipeline = std::make_shared<StableDiffusion3Pipeline>(m_pipeline_type,
                                                                  *clip_text_encoder_1,
                                                                  *clip_text_encoder_2,
                                                                  *m_transformer,
                                                                  *vae);
        }

        pipeline->m_root_dir = m_root_dir;
        pipeline->set_scheduler(Scheduler::from_config(m_root_dir / "scheduler/scheduler_config.json"));
        pipeline->set_generation_config(m_generation_config);
        return pipeline;
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        const auto& transformer_config = m_transformer->get_config();
        const size_t batch_size_multiplier = do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Transformer accepts 2x batch in case of CFG

        // Input tensors for transformer model
        ov::Tensor prompt_embeds_inp, pooled_prompt_embeds_inp;

        // 1. Encode positive prompt:
        std::string prompt_2_str = generation_config.prompt_2 != std::nullopt ? *generation_config.prompt_2 : positive_prompt;
        std::string prompt_3_str = generation_config.prompt_3 != std::nullopt ? *generation_config.prompt_3 : positive_prompt;

        std::string negative_prompt_1_str = generation_config.negative_prompt != std::nullopt ? *generation_config.negative_prompt : std::string{};
        std::string negative_prompt_2_str = generation_config.negative_prompt_2 != std::nullopt ? *generation_config.negative_prompt_2 : negative_prompt_1_str;
        std::string negative_prompt_3_str = generation_config.negative_prompt_3 != std::nullopt ? *generation_config.negative_prompt_3 : negative_prompt_1_str;

        // text_encoder_1_output - stores positive and negative pooled_prompt_embeds
        auto infer_start = std::chrono::steady_clock::now();
        ov::Tensor text_encoder_1_output = m_clip_text_encoder_1->infer(positive_prompt, negative_prompt_1_str, do_classifier_free_guidance(generation_config.guidance_scale));
        auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
        m_perf_metrics.encoder_inference_duration["text_encode"] = infer_duration;

        // text_encoder_1_hidden_state - stores positive and negative prompt_embeds
        size_t idx_hidden_state_1 = m_clip_text_encoder_1->get_config().num_hidden_layers + 1;
        ov::Tensor text_encoder_1_hidden_state = m_clip_text_encoder_1->get_output_tensor(idx_hidden_state_1);

        // text_encoder_2_output - stores positive and negative pooled_prompt_2_embeds
        infer_start = std::chrono::steady_clock::now();
        ov::Tensor text_encoder_2_output = m_clip_text_encoder_2->infer(prompt_2_str, negative_prompt_2_str, do_classifier_free_guidance(generation_config.guidance_scale));
        infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
        m_perf_metrics.encoder_inference_duration["text_encode_2"] = infer_duration;

        // text_encoder_2_hidden_state - stores positive and negative prompt_2_embeds
        size_t idx_hidden_state_2 = m_clip_text_encoder_2->get_config().num_hidden_layers + 1;
        ov::Tensor text_encoder_2_hidden_state = m_clip_text_encoder_2->get_output_tensor(idx_hidden_state_2);

        ov::Tensor text_encoder_3_output;
        if (m_t5_text_encoder) {
            infer_start = std::chrono::steady_clock::now();
            text_encoder_3_output = m_t5_text_encoder->infer(prompt_3_str,
                                                             negative_prompt_3_str,
                                                             do_classifier_free_guidance(generation_config.guidance_scale),
                                                             generation_config.max_sequence_length);
            auto infer_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start)
                    .count();
            m_perf_metrics.encoder_inference_duration["text_encode_3"] = infer_duration;
        } else {
            ov::Shape t5_prompt_embed_shape = {batch_size_multiplier,
                                               m_clip_text_encoder_1->get_config().max_position_embeddings,
                                               transformer_config.joint_attention_dim};
            text_encoder_3_output = ov::Tensor(ov::element::f32, t5_prompt_embed_shape);
            std::fill_n(text_encoder_3_output.data<float>(), text_encoder_3_output.get_size(), 0.0f);
            m_perf_metrics.encoder_inference_duration["text_encode_3"] = 0.0f;
        }

        ov::Tensor pooled_prompt_embed_out, prompt_embed_out, pooled_prompt_2_embed_out, prompt_2_embed_out, t5_prompt_embed_out;

        if (do_classifier_free_guidance(generation_config.guidance_scale)) {
            pooled_prompt_embed_out = get_tensor_batch(text_encoder_1_output, 1);
            prompt_embed_out = get_tensor_batch(text_encoder_1_hidden_state, 1);
            pooled_prompt_2_embed_out = get_tensor_batch(text_encoder_2_output, 1);
            prompt_2_embed_out = get_tensor_batch(text_encoder_2_hidden_state, 1);
            t5_prompt_embed_out = get_tensor_batch(text_encoder_3_output, 1);
        } else {
            pooled_prompt_embed_out = text_encoder_1_output;
            prompt_embed_out = text_encoder_1_hidden_state;
            pooled_prompt_2_embed_out = text_encoder_2_output;
            prompt_2_embed_out = text_encoder_2_hidden_state;
            t5_prompt_embed_out = text_encoder_3_output;
        }

        ov::Tensor pooled_prompt_embed, prompt_embed, pooled_prompt_2_embed, prompt_2_embed, t5_prompt_embed;
        if (generation_config.num_images_per_prompt == 1) {
            pooled_prompt_embed = pooled_prompt_embed_out;
            prompt_embed = prompt_embed_out;
            pooled_prompt_2_embed = pooled_prompt_2_embed_out;
            prompt_2_embed = prompt_2_embed_out;
            t5_prompt_embed = t5_prompt_embed_out;
        } else {
            pooled_prompt_embed = numpy_utils::repeat(pooled_prompt_embed_out, generation_config.num_images_per_prompt);
            prompt_embed = numpy_utils::repeat(prompt_embed_out, generation_config.num_images_per_prompt);
            pooled_prompt_2_embed = numpy_utils::repeat(pooled_prompt_2_embed_out, generation_config.num_images_per_prompt);
            prompt_2_embed = numpy_utils::repeat(prompt_2_embed_out, generation_config.num_images_per_prompt);
            t5_prompt_embed = numpy_utils::repeat(t5_prompt_embed_out, generation_config.num_images_per_prompt);
        }

        // concatenate hidden_states from two encoders
        ov::Tensor clip_prompt_embeds = numpy_utils::concat(prompt_embed, prompt_2_embed, -1);
        ov::Shape clip_prompt_embeds_shape = clip_prompt_embeds.get_shape();

        ov::Shape t5_prompt_embed_shape = t5_prompt_embed.get_shape();

        // padding for clip_prompt_embeds
        ov::Shape pad_embeds_shape = {clip_prompt_embeds_shape[0], clip_prompt_embeds_shape[1], t5_prompt_embed_shape[2]};
        ov::Tensor pad_embeds(ov::element::f32, pad_embeds_shape);
        padding_right(clip_prompt_embeds, pad_embeds);

        // prompt_embeds = torch.cat([pad_embeds, t5_prompt_embed], dim=-2)
        ov::Tensor prompt_embeds = numpy_utils::concat(pad_embeds, t5_prompt_embed, -2);
        // pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        ov::Tensor pooled_prompt_embeds = numpy_utils::concat(pooled_prompt_embed, pooled_prompt_2_embed, -1);

        if (do_classifier_free_guidance(generation_config.guidance_scale)) {
            // 2. Encode negative prompt:

            ov::Tensor negative_pooled_prompt_embed_out = get_tensor_batch(text_encoder_1_output, 0);
            ov::Tensor negative_prompt_embed_out = get_tensor_batch(text_encoder_1_hidden_state, 0);
            ov::Tensor negative_pooled_prompt_2_embed_out = get_tensor_batch(text_encoder_2_output, 0);
            ov::Tensor negative_prompt_2_embed_out = get_tensor_batch(text_encoder_2_hidden_state, 0);
            ov::Tensor negative_t5_prompt_embed_out = get_tensor_batch(text_encoder_3_output, 0);

            ov::Tensor negative_pooled_prompt_embed, negative_prompt_embed, negative_pooled_prompt_2_embed,
                negative_prompt_2_embed, negative_t5_prompt_embed;
            if (generation_config.num_images_per_prompt == 1) {
                negative_pooled_prompt_embed = negative_pooled_prompt_embed_out;
                negative_prompt_embed = negative_prompt_embed_out;
                negative_pooled_prompt_2_embed = negative_pooled_prompt_2_embed_out;
                negative_prompt_2_embed = negative_prompt_2_embed_out;
                negative_t5_prompt_embed = negative_t5_prompt_embed_out;
            } else {
                negative_pooled_prompt_embed = numpy_utils::repeat(negative_pooled_prompt_embed_out, generation_config.num_images_per_prompt);
                negative_prompt_embed = numpy_utils::repeat(negative_prompt_embed_out, generation_config.num_images_per_prompt);
                negative_pooled_prompt_2_embed = numpy_utils::repeat(negative_pooled_prompt_2_embed_out, generation_config.num_images_per_prompt);
                negative_prompt_2_embed = numpy_utils::repeat(negative_prompt_2_embed_out, generation_config.num_images_per_prompt);
                negative_t5_prompt_embed = numpy_utils::repeat(negative_t5_prompt_embed_out, generation_config.num_images_per_prompt);
            }

            // concatenate hidden_states from two encoders
            ov::Tensor neg_clip_prompt_embeds = numpy_utils::concat(negative_prompt_embed, negative_prompt_2_embed, -1);

            // padding for neg_clip_prompt_embeds
            padding_right(neg_clip_prompt_embeds, pad_embeds);

            // negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            ov::Tensor neg_prompt_embeds = numpy_utils::concat(pad_embeds, negative_t5_prompt_embed, -2);
            // neg_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1)
            ov::Tensor neg_pooled_prompt_embeds = numpy_utils::concat(negative_pooled_prompt_embed, negative_pooled_prompt_2_embed, -1);

            // 3. Fill in transformer inputs: concat positive and negative prompt_embeds
            prompt_embeds_inp = numpy_utils::concat(neg_prompt_embeds, prompt_embeds, 0);
            pooled_prompt_embeds_inp = numpy_utils::concat(neg_pooled_prompt_embeds, pooled_prompt_embeds, 0);
        } else {
            // 3. Fill in transformer inputs
            prompt_embeds_inp = prompt_embeds;
            pooled_prompt_embeds_inp = pooled_prompt_embeds;
        }

        // 4. Set model inputs
        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds_inp);
        m_transformer->set_hidden_states("pooled_projections", pooled_prompt_embeds_inp);
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        ov::Shape latent_shape{generation_config.num_images_per_prompt,
                               m_transformer->get_config().in_channels,
                               generation_config.height / vae_scale_factor,
                               generation_config.width / vae_scale_factor};

        ov::Tensor latent(ov::element::f32, {}), proccesed_image, image_latents, noise;

        if (initial_image) {
            proccesed_image = m_image_resizer->execute(initial_image, generation_config.height, generation_config.width);
            proccesed_image = m_image_processor->execute(proccesed_image);

            image_latents = m_vae->encode(proccesed_image, generation_config.generator);
            if (m_pipeline_type == PipelineType::INPAINTING) {
                image_latents = numpy_utils::repeat(image_latents, generation_config.num_images_per_prompt);
            }

            noise = generation_config.generator->randn_tensor(latent_shape);
            latent = ov::Tensor(image_latents.get_element_type(), image_latents.get_shape());
            image_latents.copy_to(latent);
            m_scheduler->scale_noise(latent, m_latent_timestep, noise);
        } else {
            noise = generation_config.generator->randn_tensor(latent_shape);
            latent.set_shape(latent_shape);

            // latents are multiplied by 'init_noise_sigma'
            const float * noise_data = noise.data<const float>();
            float * latent_data = latent.data<float>();
            for (size_t i = 0; i < latent.get_size(); ++i)
                latent_data[i] = noise_data[i] * m_scheduler->get_init_noise_sigma();
        }

        return std::make_tuple(latent, proccesed_image, image_latents, noise);
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) override {
        if(adapters) {
            if(auto updated_adapters = derived_adapters(*adapters)) {
                adapters = updated_adapters;
            }
            // TODO: Add LoRA Adapter support for text encoders
            m_transformer->set_adapters(adapters);
        }
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();
        ImageGenerationConfig generation_config = m_generation_config;
        generation_config.update_generation_config(properties);

        // Use callback if defined
        std::function<bool(size_t, size_t, ov::Tensor&)> callback = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback = callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>();
        }

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t batch_size_multiplier = do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Transformer accepts 2x batch in case of CFG

        if (generation_config.height < 0)
            compute_dim(generation_config.height, initial_image, 1 /* assume NHWC */);
        if (generation_config.width < 0)
            compute_dim(generation_config.width, initial_image, 2 /* assume NHWC */);

        check_inputs(generation_config, initial_image);

        set_lora_adapters(generation_config.adapters);

        // 3. Prepare timesteps
        m_scheduler->set_timesteps(generation_config.num_inference_steps, generation_config.strength);

        std::vector<float> timesteps = m_scheduler->get_float_timesteps();
        m_latent_timestep = timesteps[0];

        // 4. Compute text encoders and set hidden states
        compute_hidden_states(positive_prompt, generation_config);

        // 5. Prepare latent variables
        ov::Tensor latent, processed_image, image_latent, noise;
        std::tie(latent, processed_image, image_latent, noise) = prepare_latents(initial_image, generation_config);

        // 6. Prepare mask latents
        ov::Tensor mask, masked_image_latent;
        if (m_pipeline_type == PipelineType::INPAINTING) {
            std::tie(mask, masked_image_latent) = prepare_mask_latents(mask_image, processed_image, generation_config, batch_size_multiplier);
        }

        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);

        // 7. Denoising loop
        ov::Tensor noisy_residual_tensor(ov::element::f32, {});

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();
            // concat the same latent twice along a batch dimension in case of CFG
            if (batch_size_multiplier > 1) {
                numpy_utils::batch_copy(latent, latent_cfg, 0, 0, generation_config.num_images_per_prompt);
                numpy_utils::batch_copy(latent, latent_cfg, 0, generation_config.num_images_per_prompt, generation_config.num_images_per_prompt);
            } else {
                // just assign to save memory copy
                latent_cfg = latent;
            }
            ov::Tensor timestep(ov::element::f32, {1}, &timesteps[inference_step]);
            auto infer_start = std::chrono::steady_clock::now();
            ov::Tensor noise_pred_tensor = m_transformer->infer(latent_cfg, timestep);
            auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

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

            if (m_pipeline_type == PipelineType::INPAINTING && !is_inpainting_model()) {
                blend_latents(image_latent, noise, mask, latent, inference_step);
            }

            if (callback && callback(inference_step, timesteps.size(), latent)) {
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
        auto decode_start = std::chrono::steady_clock::now();
        auto image = decode(latent);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();
        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        return image;
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        return m_vae->decode(latent);
    }

    ImageGenerationPerfMetrics get_performance_metrics() override {
        m_perf_metrics.load_time = m_load_time_ms;
        return m_perf_metrics;
    }

protected:
    // Returns non-empty updated adapters if they are required to be updated
    static std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters) {
        return ov::genai::derived_adapters(adapters, flux_adapter_normalization);
    }

private:
    size_t get_config_in_channels() const override {
        assert(m_transformer != nullptr);
        return m_transformer->get_config().in_channels;
    }

    void blend_latents(ov::Tensor image_latent, ov::Tensor noise, ov::Tensor mask, ov::Tensor latent, size_t inference_step) override {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::INPAINTING, "'blend_latents' can be called for inpainting pipeline only");
        OPENVINO_ASSERT(image_latent.get_shape() == latent.get_shape(), "Shapes for current", latent.get_shape(), "and initial image latents ", image_latent.get_shape(), " must match");

        ov::Tensor noised_image_latent(image_latent.get_element_type(), {});

        std::vector<float> timesteps = m_scheduler->get_float_timesteps();
        if (inference_step < timesteps.size() - 1) {
            image_latent.copy_to(noised_image_latent);

            float noise_timestep = timesteps[inference_step + 1];
            m_scheduler->scale_noise(noised_image_latent, noise_timestep, noise);
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
            generation_config_value = transformer_config.sample_size * vae_scale_factor;
    }

    bool do_classifier_free_guidance(float guidance_scale) const {
        return guidance_scale > 1.0;
    }

    void initialize_generation_config(const std::string& class_name) override {
        OPENVINO_ASSERT(m_transformer != nullptr);
        OPENVINO_ASSERT(m_vae != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();

        m_generation_config.height = transformer_config.sample_size * vae_scale_factor;
        m_generation_config.width = transformer_config.sample_size * vae_scale_factor;

        if (class_name == "StableDiffusion3Pipeline" || class_name == "StableDiffusion3Img2ImgPipeline" || class_name == "StableDiffusion3InpaintPipeline") {
            m_generation_config.guidance_scale = 7.0f;
            m_generation_config.num_inference_steps = 28;
            m_generation_config.max_sequence_length = 256;
            m_generation_config.strength = m_pipeline_type == PipelineType::TEXT_2_IMAGE ? 1.0f : 0.6f;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        OPENVINO_ASSERT(m_transformer != nullptr);
        OPENVINO_ASSERT(m_vae != nullptr);

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t patch_size = m_transformer->get_config().patch_size;

        OPENVINO_ASSERT((height % (vae_scale_factor * patch_size) == 0 || height < 0) &&
                            (width % (vae_scale_factor * patch_size) == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ",
                        vae_scale_factor);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.height, generation_config.width);

        const bool is_classifier_free_guidance = do_classifier_free_guidance(generation_config.guidance_scale);

        OPENVINO_ASSERT(generation_config.max_sequence_length <= 512, "T5's 'max_sequence_length' must be less or equal to 512");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt == std::nullopt,
                        "Negative prompt is not used when guidance scale < 1.0");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt_2 == std::nullopt,
                        "Negative prompt 2 is not used when guidance scale < 1.0");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt_3 == std::nullopt,
                        "Negative prompt 3 is not used when guidance scale < 1.0");

        if ((m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) && initial_image) {
            ov::Shape initial_image_shape = initial_image.get_shape();
            size_t height = initial_image_shape[1], width = initial_image_shape[2];

            OPENVINO_ASSERT(generation_config.strength >= 0.0f && generation_config.strength <= 1.0f,
                "'Strength' generation parameter must be withion [0, 1] range");
        } else {
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
        }
    }

    friend class Text2ImagePipeline;
    friend class Image2ImagePipeline;

    std::shared_ptr<CLIPTextModelWithProjection> m_clip_text_encoder_1 = nullptr;
    std::shared_ptr<CLIPTextModelWithProjection> m_clip_text_encoder_2 = nullptr;
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder = nullptr;
    std::shared_ptr<SD3Transformer2DModel> m_transformer = nullptr;

    float m_latent_timestep = -1;
};

}  // namespace genai
}  // namespace ov
