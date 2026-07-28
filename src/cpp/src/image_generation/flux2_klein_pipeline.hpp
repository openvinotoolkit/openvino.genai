// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cmath>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/flux_latent_utils.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/threaded_callback.hpp"

#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/flux2_transformer_2d_model.hpp"
#include "openvino/genai/image_generation/qwen3_text_encoder.hpp"
#include "utils.hpp"

namespace {

// Prepare text IDs for Flux2: (batch_size, text_seq_len, 4) with format (T=0, H=0, W=0, L=0..seq_len-1)
inline ov::Tensor flux2_prepare_text_ids(const size_t batch_size, const size_t text_seq_len) {
    ov::Tensor text_ids(ov::element::f32, {batch_size, text_seq_len, 4});
    float* data = text_ids.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t l = 0; l < text_seq_len; ++l) {
            const size_t idx = (b * text_seq_len + l) * 4;
            data[idx + 0] = 0.0f;  // T
            data[idx + 1] = 0.0f;  // H
            data[idx + 2] = 0.0f;  // W
            data[idx + 3] = static_cast<float>(l);  // L
        }
    }

    return text_ids;
}

// Prepare latent IDs for Flux2: (batch_size, H*W, 4) with format (T=0, H, W, L=0)
inline ov::Tensor flux2_prepare_latent_ids(const size_t batch_size, const size_t height, const size_t width) {
    const size_t seq_len = height * width;
    ov::Tensor latent_ids(ov::element::f32, {batch_size, seq_len, 4});
    float* data = latent_ids.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                const size_t idx = (b * seq_len + h * width + w) * 4;
                data[idx + 0] = 0.0f;  // T
                data[idx + 1] = static_cast<float>(h);  // H
                data[idx + 2] = static_cast<float>(w);  // W
                data[idx + 3] = 0.0f;  // L
            }
        }
    }

    return latent_ids;
}

// Prepare image IDs for Flux2 reference conditioning: (batch_size, N*H*W, 4) with format (T=scale+scale*i, H, W, L=0)
inline ov::Tensor flux2_prepare_image_ids(const size_t batch_size, const std::vector<std::pair<size_t, size_t>>& image_dims, const float scale = 10.0f) {
    // Calculate total sequence length across all images
    size_t total_seq_len = 0;
    for (const auto& [h, w] : image_dims) {
        total_seq_len += h * w;
    }

    ov::Tensor image_ids(ov::element::f32, {batch_size, total_seq_len, 4});
    float* data = image_ids.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        size_t offset = 0;
        for (size_t img_idx = 0; img_idx < image_dims.size(); ++img_idx) {
            const float t_coord = scale + scale * static_cast<float>(img_idx);
            const size_t h = image_dims[img_idx].first;
            const size_t w = image_dims[img_idx].second;

            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    const size_t idx = (b * total_seq_len + offset + hi * w + wi) * 4;
                    data[idx + 0] = t_coord;  // T
                    data[idx + 1] = static_cast<float>(hi);  // H
                    data[idx + 2] = static_cast<float>(wi);  // W
                    data[idx + 3] = 0.0f;  // L
                }
            }
            offset += h * w;
        }
    }

    return image_ids;
}

}  // anonymous namespace

namespace ov {
namespace genai {

class Flux2KleinPipeline : public DiffusionPipeline {
public:
    Flux2KleinPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) : Flux2KleinPipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        if (data.contains("is_distilled")) {
            read_json_param(data, "is_distilled", m_is_distilled);
        }

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "Qwen3ForCausalLM") {
            m_text_encoder = std::make_shared<Qwen3TextEncoder>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKLFlux2" || vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
            } else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder");
            } else {
                OPENVINO_ASSERT(false, "Unsupported pipeline type for Flux2KleinPipeline");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "Flux2Transformer2DModel") {
            m_transformer = std::make_shared<Flux2Transformer2DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        // Load batch norm parameters for VAE latent normalization
        load_vae_batch_norm_params(root_dir / "vae_decoder" / "config.json");

        // initialize generation config
        initialize_generation_config("Flux2KleinPipeline");
    }

    Flux2KleinPipeline(PipelineType pipeline_type,
                       const std::filesystem::path& root_dir,
                       const std::string& device,
                       const ov::AnyMap& properties)
        : Flux2KleinPipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        if (data.contains("is_distilled")) {
            read_json_param(data, "is_distilled", m_is_distilled);
        }

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        auto updated_properties = update_adapters_in_properties(properties, &Flux2KleinPipeline::derived_adapters);

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "Qwen3ForCausalLM") {
            m_text_encoder = std::make_shared<Qwen3TextEncoder>(root_dir / "text_encoder", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKLFlux2" || vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, *updated_properties);
            } else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder", device, *updated_properties);
            } else {
                OPENVINO_ASSERT(false, "Unsupported pipeline type for Flux2KleinPipeline");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "Flux2Transformer2DModel") {
            m_transformer = std::make_shared<Flux2Transformer2DModel>(root_dir / "transformer", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        // Load batch norm parameters for VAE latent normalization
        load_vae_batch_norm_params(root_dir / "vae_decoder" / "config.json");

        // initialize generation config
        initialize_generation_config("Flux2KleinPipeline");
        update_adapters_from_properties(properties, m_generation_config.adapters);
    }

    Flux2KleinPipeline(PipelineType pipeline_type,
                       const Qwen3TextEncoder& text_encoder,
                       const Flux2Transformer2DModel& transformer,
                       const AutoencoderKL& vae)
        : Flux2KleinPipeline(pipeline_type) {
        m_text_encoder = std::make_shared<Qwen3TextEncoder>(text_encoder);
        m_vae = std::make_shared<AutoencoderKL>(vae);
        m_transformer = std::make_shared<Flux2Transformer2DModel>(transformer);
        initialize_generation_config("Flux2KleinPipeline");
    }

    Flux2KleinPipeline(PipelineType pipeline_type, const Flux2KleinPipeline& pipe)
        : Flux2KleinPipeline(pipeline_type) {
        m_root_dir = pipe.m_root_dir;

        m_text_encoder = std::make_shared<Qwen3TextEncoder>(*pipe.m_text_encoder);
        m_vae = std::make_shared<AutoencoderKL>(*pipe.m_vae);
        m_transformer = std::make_shared<Flux2Transformer2DModel>(*pipe.m_transformer);

        m_pipeline_type = pipeline_type;
        m_bn_mean = pipe.m_bn_mean;
        m_bn_std = pipe.m_bn_std;
        m_is_distilled = pipe.m_is_distilled;
        initialize_generation_config("Flux2KleinPipeline");
    }

    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        check_image_size(height, width);

        const int text_encoder_batch = do_classifier_free_guidance(guidance_scale) ? 2 : 1;
        m_text_encoder->reshape(text_encoder_batch, m_generation_config.max_sequence_length);

        // For img2img, hidden_states/img_ids seq_len is doubled (noise latents + reference latents)
        const int transformer_height = (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) ? height * 2 : height;
        m_transformer->reshape(num_images_per_prompt, transformer_height, width, m_generation_config.max_sequence_length);

        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);
        auto updated_properties = update_adapters_in_properties(properties, &Flux2KleinPipeline::derived_adapters);
        m_text_encoder->compile(text_encode_device, *updated_properties);
        m_vae->compile(vae_device, *updated_properties);
        m_transformer->compile(denoise_device, *updated_properties);
    }

    std::shared_ptr<DiffusionPipeline> clone() override {
        OPENVINO_ASSERT(!m_root_dir.empty(), "Cannot clone pipeline without root directory");

        std::shared_ptr<AutoencoderKL> vae = std::make_shared<AutoencoderKL>(m_vae->clone());
        std::shared_ptr<Flux2Transformer2DModel> transformer = std::make_shared<Flux2Transformer2DModel>(m_transformer->clone());
        std::shared_ptr<Qwen3TextEncoder> text_encoder = m_text_encoder->clone();
        std::shared_ptr<Flux2KleinPipeline> pipeline = std::make_shared<Flux2KleinPipeline>(m_pipeline_type,
                                                                                            *text_encoder,
                                                                                            *transformer,
                                                                                            *vae);

        pipeline->m_root_dir = m_root_dir;
        pipeline->m_bn_mean = m_bn_mean;
        pipeline->m_bn_std = m_bn_std;
        pipeline->m_is_distilled = m_is_distilled;
        pipeline->set_scheduler(Scheduler::from_config(m_root_dir / "scheduler/scheduler_config.json"));
        pipeline->set_generation_config(m_generation_config);
        return pipeline;
    }

    bool do_classifier_free_guidance(const float guidance_scale) const {
        return guidance_scale > 1.0f && !m_is_distilled;
    }

    bool do_classifier_free_guidance(const ImageGenerationConfig& generation_config) const {
        return do_classifier_free_guidance(generation_config.guidance_scale);
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        const bool do_cfg = do_classifier_free_guidance(generation_config);

        auto infer_start = std::chrono::steady_clock::now();
        ov::Tensor encoder_output = m_text_encoder->infer(positive_prompt, "", do_cfg, generation_config.max_sequence_length);
        auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = infer_duration;

        const size_t text_seq_len = encoder_output.get_shape()[1];
        const size_t output_dim = encoder_output.get_shape()[2];

        if (do_cfg) {
            // encoder_output shape: (2, seq_len, dim) where [0]=negative, [1]=positive
            ov::Tensor prompt_embeds(ov::element::f32, {1, text_seq_len, output_dim});
            std::memcpy(prompt_embeds.data<float>(),
                        encoder_output.data<float>() + text_seq_len * output_dim,
                        text_seq_len * output_dim * sizeof(float));

            ov::Tensor neg_embeds(ov::element::f32, {1, text_seq_len, output_dim});
            std::memcpy(neg_embeds.data<float>(),
                        encoder_output.data<float>(),
                        text_seq_len * output_dim * sizeof(float));

            prompt_embeds = numpy_utils::repeat(prompt_embeds, generation_config.num_images_per_prompt);
            neg_embeds = numpy_utils::repeat(neg_embeds, generation_config.num_images_per_prompt);

            m_positive_prompt_embeds = prompt_embeds;
            m_positive_text_ids = flux2_prepare_text_ids(generation_config.num_images_per_prompt, text_seq_len);
            m_negative_prompt_embeds = neg_embeds;
            m_negative_text_ids = flux2_prepare_text_ids(generation_config.num_images_per_prompt, text_seq_len);
        } else {
            // encoder_output shape: (1, seq_len, dim)
            ov::Tensor prompt_embeds = numpy_utils::repeat(encoder_output, generation_config.num_images_per_prompt);

            m_positive_prompt_embeds = prompt_embeds;
            m_positive_text_ids = flux2_prepare_text_ids(generation_config.num_images_per_prompt, text_seq_len);
            m_negative_prompt_embeds = ov::Tensor();
            m_negative_text_ids = ov::Tensor();
        }

        if (m_transformer->get_config().guidance_embeds) {
            ov::Tensor guidance(ov::element::f32, {generation_config.num_images_per_prompt});
            std::fill_n(guidance.data<float>(), guidance.get_size(), static_cast<float>(generation_config.guidance_scale));
            m_transformer->set_hidden_states("guidance", guidance);
        }

        m_transformer->set_hidden_states("encoder_hidden_states", m_positive_prompt_embeds);
        m_transformer->set_hidden_states("txt_ids", m_positive_text_ids);
    }

    // Prepare and set image IDs (latent_ids only for text2image, latent_ids + image_ids for img2img)
    void set_latent_and_image_ids(const ImageGenerationConfig& generation_config) {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t latent_height = generation_config.height / vae_scale_factor / 2;
        const size_t latent_width = generation_config.width / vae_scale_factor / 2;

        ov::Tensor latent_image_ids = flux2_prepare_latent_ids(generation_config.num_images_per_prompt, latent_height, latent_width);

        if (!m_ref_image_ids) {
            // Text2Image: only latent IDs
            m_transformer->set_hidden_states("img_ids", latent_image_ids);
        } else {
            // Image2Image: concatenate latent IDs + image IDs along sequence dimension (dim=1)
            const size_t batch_size = latent_image_ids.get_shape()[0];
            const size_t latent_seq = latent_image_ids.get_shape()[1];
            const size_t image_seq = m_ref_image_ids.get_shape()[1];
            const size_t total_seq = latent_seq + image_seq;

            ov::Tensor combined_ids(ov::element::f32, {batch_size, total_seq, 4});
            float* dst = combined_ids.data<float>();
            const float* latent_src = latent_image_ids.data<float>();
            const float* image_src = m_ref_image_ids.data<float>();

            for (size_t b = 0; b < batch_size; ++b) {
                std::memcpy(dst + b * total_seq * 4, latent_src + b * latent_seq * 4, latent_seq * 4 * sizeof(float));
                std::memcpy(dst + b * total_seq * 4 + latent_seq * 4, image_src + b * image_seq * 4, image_seq * 4 * sizeof(float));
            }

            m_transformer->set_hidden_states("img_ids", combined_ids);
        }
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t num_channels_latents = m_transformer->get_config().in_channels / 4;
        const size_t height = generation_config.height / vae_scale_factor;
        const size_t width = generation_config.width / vae_scale_factor;

        ov::Shape latent_shape{generation_config.num_images_per_prompt,
                               num_channels_latents,
                               height,
                               width};
        ov::Tensor latent, noise, processed_image, image_latents;

        // Reset reference image state
        m_ref_image_latents = ov::Tensor();
        m_ref_image_ids = ov::Tensor();

        if (initial_image) {
            processed_image = m_image_resizer->execute(initial_image, generation_config.height, generation_config.width);
            processed_image = m_image_processor->execute(processed_image);
            auto encode_start = std::chrono::steady_clock::now();
            image_latents = m_vae->encode(processed_image);
            m_perf_metrics.vae_encoder_inference_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - encode_start)
                    .count();

            // Undo VAE's scaling_factor (AutoencoderKLFlux2 doesn't use it, but our AutoencoderKL applies it)
            undo_vae_encode_scaling(image_latents);

            // Pack latents and normalize with batch norm
            m_ref_image_latents = pack_latents(image_latents, 1, num_channels_latents, height, width);
            apply_bn_normalize(m_ref_image_latents);

            // Prepare image IDs with T-offset
            const size_t img_h = height / 2;
            const size_t img_w = width / 2;
            std::vector<std::pair<size_t, size_t>> image_dims = {{img_h, img_w}};
            m_ref_image_ids = flux2_prepare_image_ids(generation_config.num_images_per_prompt, image_dims);

            // Repeat for batch if needed
            if (generation_config.num_images_per_prompt > 1) {
                m_ref_image_latents = numpy_utils::repeat(m_ref_image_latents, generation_config.num_images_per_prompt);
            }
        }

        // Generate random noise latents (same for both text2image and img2img)
        noise = generation_config.generator->randn_tensor(latent_shape);
        latent = pack_latents(noise, generation_config.num_images_per_prompt, num_channels_latents, height, width);

        return std::make_tuple(latent, processed_image, image_latents, noise);
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) override {
        if (adapters) {
            if (auto updated_adapters = derived_adapters(*adapters)) {
                adapters = updated_adapters;
            }
            m_transformer->set_adapters(adapters);
        }
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const ov::AnyMap& properties) override {
        OPENVINO_ASSERT(!mask_image, "Flux2KleinPipeline does not support mask_image/inpainting");
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();
        m_custom_generation_config = m_generation_config;
        m_custom_generation_config.update_generation_config(properties);

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t multiple_of = vae_scale_factor * 2;

        // Match diffusers behavior: if height/width not explicitly set and initial_image
        // is provided, infer output dimensions from the input image (rounded to multiple_of)
        bool height_set = properties.find(ov::genai::height.name()) != properties.end();
        bool width_set = properties.find(ov::genai::width.name()) != properties.end();

        if (initial_image && !height_set) {
            size_t img_h = initial_image.get_shape()[1];
            m_custom_generation_config.height = static_cast<int>((img_h / multiple_of) * multiple_of);
        } else if (m_custom_generation_config.height < 0) {
            m_custom_generation_config.height = m_transformer->get_config().default_sample_size * vae_scale_factor;
        }

        if (initial_image && !width_set) {
            size_t img_w = initial_image.get_shape()[2];
            m_custom_generation_config.width = static_cast<int>((img_w / multiple_of) * multiple_of);
        } else if (m_custom_generation_config.width < 0) {
            m_custom_generation_config.width = m_transformer->get_config().default_sample_size * vae_scale_factor;
        }

        check_inputs(m_custom_generation_config, initial_image);

        set_lora_adapters(m_custom_generation_config.adapters);

        // use callback if defined
        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        compute_hidden_states(positive_prompt, m_custom_generation_config);

        // Compute image_seq_len for timestep scheduling
        const size_t latent_height = m_custom_generation_config.height / vae_scale_factor / 2;
        const size_t latent_width = m_custom_generation_config.width / vae_scale_factor / 2;
        const size_t image_seq_len = latent_height * latent_width;

        // Flux2 Klein does not support variable strength:
        // - Text2Image: always denoises from pure noise, requires all steps
        // - Image2Image: uses reference conditioning (image latents as extra tokens),
        //   not noise blending, so partial denoising is not applicable
        // Aligned with diffusers FluxPipeline which does not accept 'strength':
        m_scheduler->set_timesteps(image_seq_len, m_custom_generation_config.num_inference_steps, 1.0f);

        // Prepare timesteps
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        // Prepare latent variables
        ov::Tensor latents;
        std::tie(latents, std::ignore, std::ignore, std::ignore) = prepare_latents(initial_image, m_custom_generation_config);

        // Set latent IDs (and image IDs if img2img)
        set_latent_and_image_ids(m_custom_generation_config);

        // Denoising loop
        ov::Tensor timestep(ov::element::f32, {m_custom_generation_config.num_images_per_prompt});
        float* timestep_data = timestep.data<float>();

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();

            // timestep / 1000 as per diffusers
            for (size_t i = 0; i < m_custom_generation_config.num_images_per_prompt; ++i) {
                timestep_data[i] = timesteps[inference_step] / 1000.0f;
            }

            // For img2img: concatenate latents with reference image latents along sequence dim
            ov::Tensor transformer_input = latents;
            if (m_ref_image_latents) {
                transformer_input = concat_along_seq_dim(latents, m_ref_image_latents);
            }

            auto infer_start = std::chrono::steady_clock::now();
            ov::Tensor noise_pred_tensor = m_transformer->infer(transformer_input, timestep);
            auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

            // Trim output to latent sequence length (discard reference tokens if any)
            noise_pred_tensor = trim_to_latent_seq_len(noise_pred_tensor, latents);

            // Classifier-free guidance: run transformer with negative prompt and combine
            if (do_classifier_free_guidance(m_custom_generation_config)) {
                // Copy positive prediction before negative inference overwrites the output buffer
                ov::Tensor pos_noise_pred(noise_pred_tensor.get_element_type(), noise_pred_tensor.get_shape());
                noise_pred_tensor.copy_to(pos_noise_pred);

                m_transformer->set_hidden_states("encoder_hidden_states", m_negative_prompt_embeds);
                m_transformer->set_hidden_states("txt_ids", m_negative_text_ids);

                auto neg_infer_start = std::chrono::steady_clock::now();
                ov::Tensor neg_noise_pred = m_transformer->infer(transformer_input, timestep);
                auto neg_infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - neg_infer_start);
                m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(neg_infer_duration));

                neg_noise_pred = trim_to_latent_seq_len(neg_noise_pred, latents);

                // noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                const float guidance_scale = m_custom_generation_config.guidance_scale;
                float* pred_data = pos_noise_pred.data<float>();
                const float* neg_data = neg_noise_pred.data<float>();
                for (size_t i = 0; i < pos_noise_pred.get_size(); ++i) {
                    pred_data[i] = neg_data[i] + guidance_scale * (pred_data[i] - neg_data[i]);
                }
                noise_pred_tensor = pos_noise_pred;

                // Restore positive prompt embeddings for the next step
                m_transformer->set_hidden_states("encoder_hidden_states", m_positive_prompt_embeds);
                m_transformer->set_hidden_states("txt_ids", m_positive_text_ids);
            }

            auto scheduler_step_result = m_scheduler->step(noise_pred_tensor, latents, inference_step, m_custom_generation_config.generator);
            latents = scheduler_step_result["latent"];

            if (callback_ptr && callback_ptr->has_callback() && callback_ptr->write(inference_step, timesteps.size(), latents) == CallbackStatus::STOP) {
                callback_ptr->end();
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

        if (callback_ptr != nullptr) {
            callback_ptr->end();
        }

        // Denormalize with batch norm in sequence form
        apply_bn_denormalize(latents);

        // Unpack latents: (B, seq_len, C*4) -> (B, C, H, W)
        ov::Tensor final_latents = unpack_latents(latents, m_custom_generation_config.height, m_custom_generation_config.width, vae_scale_factor);

        // Pre-apply VAE's scaling_factor so decode's internal division cancels out
        apply_vae_decode_scaling(final_latents);

        const auto decode_start = std::chrono::steady_clock::now();
        auto image = m_vae->decode(final_latents);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();
        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        return image;
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        ov::Tensor latent_copy(latent.get_element_type(), latent.get_shape());
        latent.copy_to(latent_copy);
        apply_bn_denormalize(latent_copy);
        ov::Tensor final_latents = unpack_latents(latent_copy, m_custom_generation_config.height, m_custom_generation_config.width, vae_scale_factor);
        apply_vae_decode_scaling(final_latents);
        return m_vae->decode(final_latents);
    }

    ImageGenerationPerfMetrics get_performance_metrics() override {
        m_perf_metrics.load_time = m_load_time_ms;
        return m_perf_metrics;
    }

protected:
    explicit Flux2KleinPipeline(PipelineType pipeline_type) :
        DiffusionPipeline(pipeline_type) {}

    void initialize_generation_config(const std::string& class_name) override {
        OPENVINO_ASSERT(m_transformer != nullptr);
        OPENVINO_ASSERT(m_vae != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();

        m_generation_config.height = transformer_config.default_sample_size * vae_scale_factor;
        m_generation_config.width = transformer_config.default_sample_size * vae_scale_factor;

        if (class_name == "Flux2KleinPipeline") {
            m_generation_config.guidance_scale = 4.0f;
            m_generation_config.num_inference_steps = 50;
            m_generation_config.max_sequence_length = 512;
            m_generation_config.strength = 1.0f;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        // Flux2 requires dimensions divisible by vae_scale_factor * 2 (due to patchification)
        OPENVINO_ASSERT((height % (vae_scale_factor * 2) == 0 || height < 0) &&
                        (width % (vae_scale_factor * 2) == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ",
                        vae_scale_factor * 2);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.height, generation_config.width);

        OPENVINO_ASSERT(generation_config.max_sequence_length <= 512, "'max_sequence_length' must be less or equal to 512");

        OPENVINO_ASSERT(generation_config.prompt_2 == std::nullopt, "Prompt 2 is not used by Flux2KleinPipeline");
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by Flux2KleinPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used by Flux2KleinPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by Flux2KleinPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by Flux2KleinPipeline");

        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            // Flux2 Klein does not support variable strength:
            // - Text2Image: always denoises from pure noise, requires all steps
            // - Image2Image: uses reference conditioning (image latents as extra tokens),
            //   not noise blending, so partial denoising is not applicable
            // Aligned with diffusers FluxPipeline which does not accept 'strength':
        } else {
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
        }
    }

    size_t get_config_in_channels() const override {
        OPENVINO_ASSERT(m_transformer != nullptr);
        return m_transformer->get_config().in_channels;
    }

    void blend_latents(ov::Tensor latents,
                       const ov::Tensor image_latent,
                       const ov::Tensor mask,
                       const ov::Tensor noise,
                       size_t inference_step) override {
        OPENVINO_THROW("blend_latents is not supported by Flux2KleinPipeline");
    }

    // Returns non-empty updated adapters if they are required to be updated
    static std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters) {
        return ov::genai::derived_adapters(adapters, flux_adapter_normalization);
    }

private:
    void load_vae_batch_norm_params(const std::filesystem::path& vae_config_path) {
        std::ifstream file(vae_config_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open VAE config for batch norm parameters: ", vae_config_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        size_t latent_channels = 32;
        float batch_norm_eps = 1e-4f;
        std::vector<int> patch_size = {2, 2};

        if (data.contains("latent_channels")) {
            read_json_param(data, "latent_channels", latent_channels);
        }
        if (data.contains("batch_norm_eps")) {
            read_json_param(data, "batch_norm_eps", batch_norm_eps);
        }
        if (data.contains("patch_size")) {
            read_json_param(data, "patch_size", patch_size);
        }
        OPENVINO_ASSERT(patch_size.size() == 2, "patch_size must have exactly 2 elements, got ", patch_size.size());

        m_batch_norm_eps = batch_norm_eps;
        m_latent_channels = latent_channels;
        m_patch_size = patch_size;

        const size_t num_bn_channels = latent_channels * patch_size[0] * patch_size[1];

        OPENVINO_ASSERT(data.contains("bn_running_mean_data") && data.contains("bn_running_var_data"),
                        "VAE config missing required 'bn_running_mean_data' and/or 'bn_running_var_data' for Flux2KleinPipeline");

        std::vector<float> bn_mean_data = data["bn_running_mean_data"].get<std::vector<float>>();
        std::vector<float> bn_var_data = data["bn_running_var_data"].get<std::vector<float>>();

        OPENVINO_ASSERT(bn_mean_data.size() == num_bn_channels,
                        "BN running mean size mismatch: expected ", num_bn_channels, ", got ", bn_mean_data.size());
        OPENVINO_ASSERT(bn_var_data.size() == num_bn_channels,
                        "BN running var size mismatch: expected ", num_bn_channels, ", got ", bn_var_data.size());

        m_bn_mean = std::move(bn_mean_data);
        m_bn_std.resize(num_bn_channels);

        // Convert variance to std: std = sqrt(var + eps)
        for (size_t i = 0; i < num_bn_channels; ++i) {
            m_bn_std[i] = std::sqrt(bn_var_data[i] + m_batch_norm_eps);
        }
    }

    // Apply batch norm normalization: (latents - mean) / std
    // Operates on sequence form (B, seq_len, C)
    void apply_bn_normalize(ov::Tensor& latents) const {
        if (m_bn_mean.empty() || m_bn_std.empty()) {
            return;
        }

        const ov::Shape& shape = latents.get_shape();
        OPENVINO_ASSERT(shape.size() == 3, "Expected 3D tensor (B, seq, C) for batch norm");
        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t channels = shape[2];

        float* data = latents.data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                float* token_data = data + (b * seq_len + s) * channels;
                for (size_t c = 0; c < channels; ++c) {
                    token_data[c] = (token_data[c] - m_bn_mean[c]) / m_bn_std[c];
                }
            }
        }
    }

    // Apply batch norm denormalization: latents * std + mean
    // Operates on sequence form (B, seq_len, C)
    void apply_bn_denormalize(ov::Tensor& latents) const {
        if (m_bn_mean.empty() || m_bn_std.empty()) {
            return;
        }

        const ov::Shape& shape = latents.get_shape();
        OPENVINO_ASSERT(shape.size() == 3, "Expected 3D tensor (B, seq, C) for batch norm denormalization");
        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t channels = shape[2];

        float* data = latents.data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                float* token_data = data + (b * seq_len + s) * channels;
                for (size_t c = 0; c < channels; ++c) {
                    token_data[c] = token_data[c] * m_bn_std[c] + m_bn_mean[c];
                }
            }
        }
    }

    // Trim transformer output to match the latent sequence length
    ov::Tensor trim_to_latent_seq_len(const ov::Tensor& noise_pred, const ov::Tensor& latents) const {
        const ov::Shape& pred_shape = noise_pred.get_shape();
        const ov::Shape& latent_shape = latents.get_shape();

        if (pred_shape[1] == latent_shape[1]) {
            return noise_pred;
        }

        // noise_pred[:, :latents.shape[1], :]
        const size_t batch_size = pred_shape[0];
        const size_t target_seq_len = latent_shape[1];
        const size_t channels = pred_shape[2];

        ov::Shape trimmed_shape = {batch_size, target_seq_len, channels};
        ov::Tensor trimmed(noise_pred.get_element_type(), trimmed_shape);

        const float* src = noise_pred.data<float>();
        float* dst = trimmed.data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            std::memcpy(dst + b * target_seq_len * channels,
                        src + b * pred_shape[1] * channels,
                        target_seq_len * channels * sizeof(float));
        }

        return trimmed;
    }

    // Concatenate two tensors along the sequence dimension (dim=1)
    // Input shapes: (B, S1, C) and (B, S2, C) -> Output: (B, S1+S2, C)
    ov::Tensor concat_along_seq_dim(const ov::Tensor& a, const ov::Tensor& b) const {
        const ov::Shape& shape_a = a.get_shape();
        const ov::Shape& shape_b = b.get_shape();

        OPENVINO_ASSERT(shape_a[0] == shape_b[0] && shape_a[2] == shape_b[2],
                        "Batch and channel dimensions must match for concatenation");

        const size_t batch_size = shape_a[0];
        const size_t seq_a = shape_a[1];
        const size_t seq_b = shape_b[1];
        const size_t channels = shape_a[2];

        ov::Shape out_shape = {batch_size, seq_a + seq_b, channels};
        ov::Tensor result(a.get_element_type(), out_shape);

        const float* src_a = a.data<float>();
        const float* src_b = b.data<float>();
        float* dst = result.data<float>();

        for (size_t bx = 0; bx < batch_size; ++bx) {
            std::memcpy(dst + bx * (seq_a + seq_b) * channels,
                        src_a + bx * seq_a * channels,
                        seq_a * channels * sizeof(float));
            std::memcpy(dst + bx * (seq_a + seq_b) * channels + seq_a * channels,
                        src_b + bx * seq_b * channels,
                        seq_b * channels * sizeof(float));
        }

        return result;
    }

    // Undo VAE encode scaling: AutoencoderKLFlux2 doesn't use scaling_factor,
    // but our AutoencoderKL applies (latent - shift) * scale in encode()
    void undo_vae_encode_scaling(ov::Tensor& latents) const {
        const auto& vae_config = m_vae->get_config();
        if (vae_config.scaling_factor == 1.0f && vae_config.shift_factor == 0.0f) {
            return;
        }
        float* data = latents.data<float>();
        for (size_t i = 0; i < latents.get_size(); ++i) {
            data[i] = data[i] / vae_config.scaling_factor + vae_config.shift_factor;
        }
    }

    // Pre-apply VAE decode scaling: the decoder internally divides by scaling_factor
    // and adds shift_factor. We multiply/subtract to cancel that out.
    void apply_vae_decode_scaling(ov::Tensor& latents) const {
        const auto& vae_config = m_vae->get_config();
        if (vae_config.scaling_factor == 1.0f && vae_config.shift_factor == 0.0f) {
            return;
        }
        float* data = latents.data<float>();
        for (size_t i = 0; i < latents.get_size(); ++i) {
            data[i] = (data[i] - vae_config.shift_factor) * vae_config.scaling_factor;
        }
    }

    std::shared_ptr<Flux2Transformer2DModel> m_transformer = nullptr;
    std::shared_ptr<Qwen3TextEncoder> m_text_encoder = nullptr;

    ImageGenerationConfig m_custom_generation_config;

    // Reference image conditioning for img2img
    ov::Tensor m_ref_image_latents;  // Packed image latents: (B, img_seq_len, C)
    ov::Tensor m_ref_image_ids;      // Image position IDs: (img_seq_len, 4)

    // Classifier-free guidance: positive and negative prompt embeddings
    ov::Tensor m_positive_prompt_embeds;
    ov::Tensor m_positive_text_ids;
    ov::Tensor m_negative_prompt_embeds;
    ov::Tensor m_negative_text_ids;

    // Model configuration
    bool m_is_distilled = false;

    // Batch norm parameters for VAE latent normalization
    std::vector<float> m_bn_mean;
    std::vector<float> m_bn_std;
    float m_batch_norm_eps = 1e-4f;
    size_t m_latent_channels = 32;
    std::vector<int> m_patch_size = {2, 2};
};

}  // namespace genai
}  // namespace ov
