// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <ctime>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "utils.hpp"

namespace {

ov::Tensor pack_latents(const ov::Tensor latents, size_t batch_size, size_t num_channels_latents, size_t height, size_t width) {
    // Reshape to (batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    ov::Shape final_shape = {batch_size, (height / 2) * (width / 2), num_channels_latents * 4};
    ov::Tensor permuted_latents = ov::Tensor(latents.get_element_type(), final_shape);

    float* src_data = latents.data<float>();
    float* dst_data = permuted_latents.data<float>();

    // Permute to (0, 2, 4, 1, 3, 5)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h2 = 0; h2 < height / 2; ++h2) {
            for (size_t w2 = 0; w2 < width / 2; ++w2) {
                for (size_t c = 0; c < num_channels_latents; ++c) {
                    for (size_t h3 = 0; h3 < 2; ++h3) {
                        for (size_t w3 = 0; w3 < 2; ++w3) {
                            size_t src_index = ((b * num_channels_latents + c) * (height / 2) + h2) * 2 * (width / 2) * 2 +
                                            (h3 * width / 2 + w2) * 2 + w3;
                            size_t dst_index = ((b * (height / 2) + h2) * (width / 2) + w2) * num_channels_latents * 4 +
                                            (c * 4 + h3 * 2 + w3);
                            dst_data[dst_index] = src_data[src_index];
                        }
                    }
                }
            }
        }
    }

    return permuted_latents;
}

ov::Tensor unpack_latents(const ov::Tensor latents, size_t height, size_t width, size_t vae_scale_factor) {
    auto latents_shape = latents.get_shape();
    size_t batch_size = latents_shape[0], channels = latents_shape[2];

    height /= vae_scale_factor;
    width /= vae_scale_factor;

    // Reshape to (batch_size, channels / 4, height * 2, width * 2)
    ov::Shape output_shape = {batch_size, channels / 4, height * 2, width * 2};
    ov::Tensor permuted_latents = ov::Tensor(latents.get_element_type(), output_shape);

    const float* src_data = latents.data<float>();
    float* dst_data = permuted_latents.data<float>();

    // Permute (0, 3, 1, 4, 2, 5)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels / 4; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    for (size_t i = 0; i < 2; ++i) {
                        for (size_t j = 0; j < 2; ++j) {
                            size_t src_idx = (((b * height + h) * width + w) * (channels / 4) + c) * 4 + i * 2 + j;
                            size_t dst_idx =
                                ((((((b * (channels / 4) + c) * height + h) * 2 + i) * width + w) * 2 + j));
                            dst_data[dst_idx] = src_data[src_idx];
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
            m_vae_decoder = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "FluxTransformer2DModel") {
            m_transformer = std::make_shared<FluxTransformer2DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "'Transformer type");
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
            m_vae_decoder = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "FluxTransformer2DModel") {
            m_transformer = std::make_shared<FluxTransformer2DModel>(root_dir / "transformer", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "'Transformer type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    FluxPipeline(PipelineType pipeline_type,
                 const CLIPTextModel& clip_text_model,
                 const T5EncoderModel& t5_text_model,
                 const FluxTransformer2DModel& transformer,
                 const AutoencoderKL& vae_decoder)
        : DiffusionPipeline(pipeline_type),
          m_clip_text_encoder(std::make_shared<CLIPTextModel>(clip_text_model)),
          m_t5_text_encoder(std::make_shared<T5EncoderModel>(t5_text_model)),
          m_vae_decoder(std::make_shared<AutoencoderKL>(vae_decoder)),
          m_transformer(std::make_shared<FluxTransformer2DModel>(transformer)) {}

    // TODO
    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        // check_image_size(height, width);

        // const size_t batch_size_multiplier =
        //     do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Transformer accepts 2x batch in case of CFG
        // m_clip_text_encoder_1->reshape(batch_size_multiplier);
        // m_clip_text_encoder_2->reshape(batch_size_multiplier);
        // m_transformer->reshape(num_images_per_prompt * batch_size_multiplier,
        //                        height,
        //                        width,
        //                        m_clip_text_encoder_1->get_config().max_position_embeddings);
        // m_vae_decoder->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        m_clip_text_encoder->compile(device, properties);
        m_t5_text_encoder->compile(device, properties);
        m_vae_decoder->compile(device, properties);
        m_transformer->compile(device, properties);
    }

    ov::Tensor prepare_latents(ov::Tensor initial_image,
                               const ImageGenerationConfig& generation_config) const override {
        const size_t vae_scale_factor = m_vae_decoder->get_vae_scale_factor();
        ov::Shape latent_shape{generation_config.num_images_per_prompt,
                               m_transformer->get_config().in_channels,
                               generation_config.height / vae_scale_factor,
                               generation_config.width / vae_scale_factor};

        ov::Tensor latent = generation_config.generator->randn_tensor(latent_shape);

        return latent;
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        const ov::AnyMap& properties) override {
        using namespace numpy_utils;
        ImageGenerationConfig generation_config = m_generation_config;
        generation_config.update_generation_config(properties);

        const size_t vae_scale_factor = m_transformer->get_vae_scale_factor();

        if (generation_config.height < 0)
            generation_config.height = m_default_sample_size * vae_scale_factor;
        if (generation_config.width < 0)
            generation_config.width = m_default_sample_size * vae_scale_factor;

        check_inputs(generation_config, initial_image);

        // encode_prompt
        std::string prompt_2_str =
            generation_config.prompt_2 != std::nullopt ? *generation_config.prompt_2 : positive_prompt;

        m_clip_text_encoder->infer(positive_prompt, "", false);
        size_t idx_pooler_output = 1;
        ov::Tensor pooled_prompt_embeds_out = m_clip_text_encoder->get_output_tensor(idx_pooler_output);

        ov::Tensor prompt_embeds_out = m_t5_text_encoder->infer(positive_prompt);

        ov::Tensor pooled_prompt_embeds, prompt_embeds;
        if (generation_config.num_images_per_prompt == 1) {
            pooled_prompt_embeds = pooled_prompt_embeds_out;
            prompt_embeds = prompt_embeds_out;
        } else {
            pooled_prompt_embeds =
                tensor_batch_copy(pooled_prompt_embeds_out, generation_config.num_images_per_prompt, 1);
            prompt_embeds = tensor_batch_copy(prompt_embeds_out, generation_config.num_images_per_prompt, 1);
        }

        // text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        ov::Shape text_ids_shape = {prompt_embeds.get_shape()[1], 3};
        ov::Tensor text_ids(ov::element::f32, text_ids_shape);
        std::fill_n(text_ids.data<float>(), text_ids_shape[0] * text_ids_shape[1], 0.0f);

        // std::cout << "pooled_prompt_embeds" << std::endl;
        // for (int i = 0; i<10; ++i){
        //     std::cout << pooled_prompt_embeds.data<float>()[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "prompt_embeds" << std::endl;
        // for (int i = 0; i<10; ++i){
        //     std::cout << prompt_embeds.data<float>()[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "text_ids" << std::endl;
        // for (int i = 0; i<10; ++i){
        //     std::cout << text_ids.data<float>()[i] << " ";
        // }
        // std::cout << std::endl;

        size_t num_channels_latents = m_transformer->get_config().in_channels / 4;
        size_t height = 2 * generation_config.height / vae_scale_factor;
        size_t width = 2 * generation_config.width / vae_scale_factor;

        ov::Tensor latents_inp = prepare_latents(initial_image, generation_config);
        ov::Tensor latents = pack_latents(latents_inp, generation_config.num_images_per_prompt, num_channels_latents, height, width);
        ov::Tensor latent_image_ids = prepare_latent_image_ids(generation_config.num_images_per_prompt, height / 2, width / 2);

        std::cout << "latents" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << latents.data<float>()[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "latent_image_ids" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << latent_image_ids.data<float>()[i] << " ";
        }
        std::cout << std::endl;

        m_transformer->set_hidden_states("pooled_projections", pooled_prompt_embeds);
        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds);
        m_transformer->set_hidden_states("txt_ids", text_ids);
        m_transformer->set_hidden_states("img_ids", latent_image_ids);

        // TODO: mu = calculate_shift(...)
        float mu = 0.63f;

        float linspace_end = 1.0f / generation_config.num_inference_steps;
        std::vector<float> sigmas = linspace<float>(1.0f, linspace_end, generation_config.num_inference_steps, true);

        m_scheduler->set_timesteps_with_sigma(sigmas, mu);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();
        size_t num_inference_steps = timesteps.size();

        // 6. Denoising loop
        size_t timestep_size = latents.get_shape()[0];
        ov::Tensor timestep(ov::element::f32, {timestep_size});
        float* timestep_data = timestep.data<float>();

        for (size_t inference_step = 0; inference_step < num_inference_steps; ++inference_step) {
            std::fill_n(timestep_data, timestep_size, (timesteps[inference_step] / 1000));

            std::cout << "latents input" << std::endl;
            for (int i = 0; i < 10; ++i) {
                std::cout << latents.data<float>()[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "timestep" << std::endl;
            for (int i = 0; i < timestep_size; ++i) {
                std::cout << timestep.data<float>()[i] << " ";
            }
            std::cout << std::endl;

            ov::Tensor noise_pred_tensor = m_transformer->infer(latents, timestep);

            std::cout << "noise_pred_tensor" << std::endl;
            for (int i = 0; i < 10; ++i) {
                std::cout << noise_pred_tensor.data<float>()[i] << " ";
            }
            std::cout << std::endl;

            auto scheduler_step_result = m_scheduler->step(noise_pred_tensor, latents, inference_step, generation_config.generator);
            latents = scheduler_step_result["latent"];

        }

        latents = unpack_latents(latents, generation_config.height, generation_config.width, vae_scale_factor);
        return m_vae_decoder->decode(latents);
    }

private:
    void initialize_generation_config(const std::string& class_name) override {
        assert(m_transformer != nullptr);
        assert(m_vae_decoder != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_transformer->get_vae_scale_factor();

        m_default_sample_size = 128;
        m_generation_config.height = m_default_sample_size * vae_scale_factor;
        m_generation_config.width = m_default_sample_size * vae_scale_factor;

        if (class_name == "FluxPipeline") {
            m_generation_config.guidance_scale = 3.5f;
            m_generation_config.num_inference_steps = 28;
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

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.width, generation_config.height);

        const char* const pipeline_name = "Flux";

        OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt,
                        "Negative prompt is not used by ",
                        pipeline_name);
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt,
                        "Negative prompt 2 is not used by ",
                        pipeline_name);
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt,
                        "Negative prompt 3 is not used by ",
                        pipeline_name);

        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by ", pipeline_name);
    }

    std::shared_ptr<FluxTransformer2DModel> m_transformer;
    std::shared_ptr<CLIPTextModel> m_clip_text_encoder;
    // TODO:
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder;
    std::shared_ptr<AutoencoderKL> m_vae_decoder;

private:
    size_t m_default_sample_size;
};

}  // namespace genai
}  // namespace ov
