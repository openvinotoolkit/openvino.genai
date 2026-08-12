// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/threaded_callback.hpp"

#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/qwen3_text_encoder.hpp"
#include "openvino/genai/image_generation/zimage_transformer_2d_model.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

class ZImagePipeline : public DiffusionPipeline {
public:
    ZImagePipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) : ZImagePipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "Qwen3ForCausalLM" || text_encoder == "Qwen3Model") {
            m_text_encoder = std::make_shared<Qwen3TextEncoder>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type for ZImagePipeline");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
            else {
                OPENVINO_THROW("ZImagePipeline only supports TEXT_2_IMAGE pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "ZImageTransformer2DModel") {
            m_transformer = std::make_shared<ZImageTransformer2DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        const std::string class_name = data["_class_name"].get<std::string>();
        initialize_generation_config(class_name);
    }

    ZImagePipeline(PipelineType pipeline_type,
                   const std::filesystem::path& root_dir,
                   const std::string& device,
                   const ov::AnyMap& properties)
        : ZImagePipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        auto updated_properties = update_adapters_in_properties(properties, &ZImagePipeline::derived_adapters);

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "Qwen3ForCausalLM" || text_encoder == "Qwen3Model") {
            m_text_encoder = std::make_shared<Qwen3TextEncoder>(root_dir / "text_encoder", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type for ZImagePipeline");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, *updated_properties);
            else {
                OPENVINO_THROW("ZImagePipeline only supports TEXT_2_IMAGE pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "ZImageTransformer2DModel") {
            m_transformer = std::make_shared<ZImageTransformer2DModel>(root_dir / "transformer", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        const std::string class_name = data["_class_name"].get<std::string>();
        initialize_generation_config(class_name);
        update_adapters_from_properties(properties, m_generation_config.adapters);
    }

    ZImagePipeline(PipelineType pipeline_type,
                   const Qwen3TextEncoder& text_encoder,
                   const ZImageTransformer2DModel& transformer,
                   const AutoencoderKL& vae)
        : ZImagePipeline(pipeline_type) {
        m_text_encoder = std::make_shared<Qwen3TextEncoder>(text_encoder);
        m_vae = std::make_shared<AutoencoderKL>(vae);
        m_transformer = std::make_shared<ZImageTransformer2DModel>(transformer);
        initialize_generation_config("ZImagePipeline");
    }

    ZImagePipeline(PipelineType pipeline_type, const ZImagePipeline& pipe)
        : ZImagePipeline(pipeline_type) {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::TEXT_2_IMAGE,
            "ZImagePipeline only supports TEXT_2_IMAGE pipeline type");

        m_root_dir = pipe.m_root_dir;
        m_text_encoder = std::make_shared<Qwen3TextEncoder>(*pipe.m_text_encoder);
        m_vae = std::make_shared<AutoencoderKL>(*pipe.m_vae);
        m_transformer = std::make_shared<ZImageTransformer2DModel>(*pipe.m_transformer);

        m_pipeline_type = pipeline_type;
        initialize_generation_config("ZImagePipeline");
    }

    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        check_image_size(height, width);

        m_text_encoder->reshape(1, m_generation_config.max_sequence_length);
        m_transformer->reshape(num_images_per_prompt, height, width, m_generation_config.max_sequence_length);
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);
        auto updated_properties = update_adapters_in_properties(properties, &ZImagePipeline::derived_adapters);
        m_text_encoder->compile(text_encode_device, *updated_properties);
        m_vae->compile(vae_device, *updated_properties);
        m_transformer->compile(denoise_device, *updated_properties);
    }

    std::shared_ptr<DiffusionPipeline> clone() override {
        OPENVINO_ASSERT(!m_root_dir.empty(), "Cannot clone pipeline without root directory");
        
        std::shared_ptr<AutoencoderKL> vae = std::make_shared<AutoencoderKL>(m_vae->clone());
        std::shared_ptr<Qwen3TextEncoder> text_encoder = m_text_encoder->clone();
        std::shared_ptr<ZImageTransformer2DModel> transformer = std::make_shared<ZImageTransformer2DModel>(m_transformer->clone());
        std::shared_ptr<ZImagePipeline> pipeline = std::make_shared<ZImagePipeline>(m_pipeline_type,
                                                                       *text_encoder,
                                                                       *transformer,
                                                                       *vae);

        pipeline->m_root_dir = m_root_dir;
        pipeline->set_scheduler(Scheduler::from_config(m_root_dir / "scheduler/scheduler_config.json"));
        pipeline->set_generation_config(m_generation_config);
        return pipeline;
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        auto infer_start = std::chrono::steady_clock::now();
        ov::Tensor prompt_embeds = m_text_encoder->infer(positive_prompt, "", false, generation_config.max_sequence_length);
        auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = infer_duration;

        prompt_embeds = numpy_utils::repeat(prompt_embeds, generation_config.num_images_per_prompt);
        m_prompt_embeds = prompt_embeds;
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        size_t num_channels_latents = m_transformer->get_config().in_channels;
        size_t height = generation_config.height / vae_scale_factor;
        size_t width = generation_config.width / vae_scale_factor;

        ov::Shape latent_shape{generation_config.num_images_per_prompt,
                               num_channels_latents,
                               height,
                               width};
        ov::Tensor latent, noise, proccesed_image, image_latents;

        noise = generation_config.generator->randn_tensor(latent_shape);
        latent = ov::Tensor(noise.get_element_type(), noise.get_shape());
        noise.copy_to(latent);

        return std::make_tuple(latent, proccesed_image, image_latents, noise);
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) override {
        if(adapters) {
            if(auto updated_adapters = derived_adapters(*adapters)) {
                adapters = updated_adapters;
            }
            m_text_encoder->set_adapters(adapters);
            m_transformer->set_adapters(adapters);
        }
    }

    std::tuple<ov::Tensor, ov::Tensor> prepare_mask_latents(ov::Tensor mask_image,
                                                            ov::Tensor processed_image,
                                                            const ImageGenerationConfig& generation_config,
                                                            const size_t batch_size_multiplier = 1) override {
        OPENVINO_THROW("prepare_mask_latents is not supported for ZImagePipeline");
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();
        ImageGenerationConfig custom_generation_config = m_generation_config;
        custom_generation_config.update_generation_config(properties);

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        custom_generation_config.height = custom_generation_config.height == 0 ? 512 : custom_generation_config.height;
        custom_generation_config.width = custom_generation_config.width == 0 ? 512 : custom_generation_config.width;

        OPENVINO_ASSERT(custom_generation_config.height == 512 && custom_generation_config.width == 512,
            "ZImagePipeline only supports fixed resolution of 512x512");

        std::shared_ptr<Generator> generator = custom_generation_config.generator;
        if (!generator) {
            OPENVINO_THROW("Generator must be provided for ZImagePipeline");
        }

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        const size_t image_seq_len = (custom_generation_config.height / vae_scale_factor / 2) *
                                     (custom_generation_config.width / vae_scale_factor / 2);
        m_scheduler->set_timesteps(image_seq_len,
                                   custom_generation_config.num_inference_steps,
                                   custom_generation_config.strength);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        compute_hidden_states(positive_prompt, custom_generation_config);

        auto [latent, processed_image, image_latents, noise] = prepare_latents(initial_image, custom_generation_config);

        // Denoising loop
        ov::Tensor timestep(ov::element::f32, {1});
        float* timestep_data = timestep.data<float>();

        for (size_t step_idx = 0; step_idx < timesteps.size(); ++step_idx) {
            auto step_start = std::chrono::steady_clock::now();
            timestep_data[0] = (1000.0f - timesteps[step_idx]) / 1000.0f;

            auto infer_start = std::chrono::steady_clock::now();
            ov::Tensor noise_pred = m_transformer->step(latent, timestep, m_prompt_embeds);
            auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

            float* noise_pred_data = noise_pred.data<float>();
            for (size_t i = 0; i < noise_pred.get_size(); ++i) {
                noise_pred_data[i] = -noise_pred_data[i];
            }

            auto scheduler_step_result = m_scheduler->step(noise_pred, latent, step_idx, generator);
            latent = scheduler_step_result["latent"];

            if (callback_ptr && callback_ptr->has_callback() && callback_ptr->write(step_idx, timesteps.size(), latent) == CallbackStatus::STOP) {
                callback_ptr->end();
                auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
                m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));

                auto image = ov::Tensor(ov::element::u8, {});
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
                return image;
            }

            auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
        }

        if (callback_ptr != nullptr) {
            callback_ptr->end();
        }

        const auto decode_start = std::chrono::steady_clock::now();
        auto image = m_vae->decode(latent);
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
        return m_perf_metrics;
    }

    void export_model(const std::filesystem::path& export_dir) override {
        OPENVINO_THROW("export_model is not yet implemented for ZImagePipeline");
    }

protected:
    void initialize_generation_config(const std::string& class_name) override {
        m_generation_config.max_sequence_length = 128;
        m_generation_config.num_images_per_prompt = 1;
        m_generation_config.height = 512;
        m_generation_config.width = 512;
        m_generation_config.num_inference_steps = 8;
        m_generation_config.guidance_scale = 7.5f;
        m_generation_config.negative_prompt = "";
    }

    void check_image_size(const int height, const int width) const override {
        OPENVINO_ASSERT(height == 512 && width == 512,
            "ZImagePipeline only supports fixed resolution of 512x512");
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
    }

    size_t get_config_in_channels() const override {
        return m_transformer->get_config().in_channels;
    }

private:
    std::shared_ptr<Qwen3TextEncoder> m_text_encoder;
    std::shared_ptr<ZImageTransformer2DModel> m_transformer;
    ov::Tensor m_prompt_embeds;

    ZImagePipeline(PipelineType pipeline_type) : DiffusionPipeline(pipeline_type) {}

    static std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters) {
        return std::nullopt;
    }
};

}  // namespace genai
}  // namespace ov
