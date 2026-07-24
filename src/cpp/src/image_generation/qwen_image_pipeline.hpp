// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/flux_latent_utils.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/threaded_callback.hpp"

#include "openvino/genai/image_generation/autoencoder_kl_qwen_image.hpp"
#include "openvino/genai/image_generation/qwen_image_transformer_2d_model.hpp"
#include "openvino/genai/image_generation/qwen2_5_vl_for_conditional_generation.hpp"
#include "utils.hpp"

namespace {

inline double qwen_image_calculate_shift(
    size_t image_seq_len,
    size_t base_seq_len = 256,
    size_t max_seq_len = 4096,
    double base_shift = 0.5,
    double max_shift = 1.15) {
    double m = (max_shift - base_shift) / (static_cast<double>(max_seq_len) - static_cast<double>(base_seq_len));
    double b = base_shift - m * static_cast<double>(base_seq_len);
    return static_cast<double>(image_seq_len) * m + b;
}

// Compute complex rotary frequency parameters
inline std::vector<std::complex<float>> compute_rope_params(const std::vector<int64_t>& indices, size_t dim, float theta = 10000.0f) {
    const size_t half_dim = dim / 2;
    std::vector<std::complex<float>> result(indices.size() * half_dim);

    for (size_t i = 0; i < indices.size(); ++i) {
        for (size_t d = 0; d < half_dim; ++d) {
            float freq = static_cast<float>(indices[i]) / std::pow(theta, static_cast<float>(2 * d) / static_cast<float>(dim));
            result[i * half_dim + d] = std::complex<float>(std::cos(freq), std::sin(freq));
        }
    }

    return result;
}

// Compute video rotary embeddings for QwenImage
// Returns (cos, sin) each of shape (frame*height*width, total_rotary_dim/2)
inline std::pair<ov::Tensor, ov::Tensor> qwen_image_compute_rotary_embeddings(
    size_t frame, size_t height, size_t width,
    const std::vector<size_t>& axes_dims_rope,
    float theta = 10000.0f) {

    // Build positive and negative index arrays (like diffusers: pos_index = arange(4096), neg_index = arange(4096).flip(0) * -1 - 1)
    const size_t max_index = 4096;

    // Compute per-axis frequencies for positive and negative indices
    auto compute_axis_freqs = [&](size_t dim, const std::vector<int64_t>& indices) {
        return compute_rope_params(indices, dim, theta);
    };

    // Positive indices for frame dimension
    std::vector<int64_t> pos_indices(max_index);
    for (size_t i = 0; i < max_index; ++i) pos_indices[i] = static_cast<int64_t>(i);

    // Negative indices
    std::vector<int64_t> neg_indices(max_index);
    for (size_t i = 0; i < max_index; ++i) neg_indices[i] = -static_cast<int64_t>(max_index - i);

    // Compute frequencies for each axis
    auto pos_freqs_0 = compute_rope_params(pos_indices, axes_dims_rope[0], theta); // frame axis
    auto pos_freqs_1 = compute_rope_params(pos_indices, axes_dims_rope[1], theta); // height axis
    auto neg_freqs_1 = compute_rope_params(neg_indices, axes_dims_rope[1], theta); // height axis negative
    auto pos_freqs_2 = compute_rope_params(pos_indices, axes_dims_rope[2], theta); // width axis
    auto neg_freqs_2 = compute_rope_params(neg_indices, axes_dims_rope[2], theta); // width axis negative

    const size_t half_dim_0 = axes_dims_rope[0] / 2;
    const size_t half_dim_1 = axes_dims_rope[1] / 2;
    const size_t half_dim_2 = axes_dims_rope[2] / 2;
    const size_t total_half_dim = half_dim_0 + half_dim_1 + half_dim_2;

    const size_t total_seq = frame * height * width;

    ov::Tensor cos_tensor(ov::element::f32, {total_seq, total_half_dim});
    ov::Tensor sin_tensor(ov::element::f32, {total_seq, total_half_dim});
    float* cos_data = cos_tensor.data<float>();
    float* sin_data = sin_tensor.data<float>();

    for (size_t f = 0; f < frame; ++f) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                const size_t idx = (f * height * width + h * width + w);
                float* cos_row = cos_data + idx * total_half_dim;
                float* sin_row = sin_data + idx * total_half_dim;

                // Frame axis: positive indices
                for (size_t d = 0; d < half_dim_0; ++d) {
                    cos_row[d] = pos_freqs_0[f * half_dim_0 + d].real();
                    sin_row[d] = pos_freqs_0[f * half_dim_0 + d].imag();
                }

                // Height axis: negative for first half, positive for second half
                // freqs_height = cat([neg[-(height - height//2):], pos[:height//2]])
                const size_t h_half = height / 2;
                size_t h_freq_idx;
                if (h < (height - h_half)) {
                    // from negative: neg[-(height - h_half) + h] = neg[max_index - (height - h_half) + h]
                    h_freq_idx = max_index - (height - h_half) + h;
                    for (size_t d = 0; d < half_dim_1; ++d) {
                        cos_row[half_dim_0 + d] = neg_freqs_1[h_freq_idx * half_dim_1 + d].real();
                        sin_row[half_dim_0 + d] = neg_freqs_1[h_freq_idx * half_dim_1 + d].imag();
                    }
                } else {
                    // from positive: pos[h - (height - h_half)]
                    h_freq_idx = h - (height - h_half);
                    for (size_t d = 0; d < half_dim_1; ++d) {
                        cos_row[half_dim_0 + d] = pos_freqs_1[h_freq_idx * half_dim_1 + d].real();
                        sin_row[half_dim_0 + d] = pos_freqs_1[h_freq_idx * half_dim_1 + d].imag();
                    }
                }

                // Width axis: negative for first half, positive for second half
                // freqs_width = cat([neg[-(width - width//2):], pos[:width//2]])
                const size_t w_half = width / 2;
                size_t w_freq_idx;
                if (w < (width - w_half)) {
                    w_freq_idx = max_index - (width - w_half) + w;
                    for (size_t d = 0; d < half_dim_2; ++d) {
                        cos_row[half_dim_0 + half_dim_1 + d] = neg_freqs_2[w_freq_idx * half_dim_2 + d].real();
                        sin_row[half_dim_0 + half_dim_1 + d] = neg_freqs_2[w_freq_idx * half_dim_2 + d].imag();
                    }
                } else {
                    w_freq_idx = w - (width - w_half);
                    for (size_t d = 0; d < half_dim_2; ++d) {
                        cos_row[half_dim_0 + half_dim_1 + d] = pos_freqs_2[w_freq_idx * half_dim_2 + d].real();
                        sin_row[half_dim_0 + half_dim_1 + d] = pos_freqs_2[w_freq_idx * half_dim_2 + d].imag();
                    }
                }
            }
        }
    }

    return {cos_tensor, sin_tensor};
}

// Compute text rotary embeddings for QwenImage
// Text positions start at max(height//2, width//2) — matching diffusers
inline std::pair<ov::Tensor, ov::Tensor> qwen_image_compute_text_rotary_embeddings(
    size_t text_seq_len,
    size_t height, size_t width,
    const std::vector<size_t>& axes_dims_rope,
    float theta = 10000.0f) {

    const size_t max_vid_index = std::max(height / 2, width / 2);
    const size_t half_dim_0 = axes_dims_rope[0] / 2;
    const size_t half_dim_1 = axes_dims_rope[1] / 2;
    const size_t half_dim_2 = axes_dims_rope[2] / 2;
    const size_t total_half_dim = half_dim_0 + half_dim_1 + half_dim_2;

    // Text frequencies use positive indices starting from max_vid_index
    std::vector<int64_t> text_indices(text_seq_len);
    for (size_t i = 0; i < text_seq_len; ++i) {
        text_indices[i] = static_cast<int64_t>(max_vid_index + i);
    }

    auto freqs_0 = compute_rope_params(text_indices, axes_dims_rope[0], theta);
    auto freqs_1 = compute_rope_params(text_indices, axes_dims_rope[1], theta);
    auto freqs_2 = compute_rope_params(text_indices, axes_dims_rope[2], theta);

    ov::Tensor cos_tensor(ov::element::f32, {text_seq_len, total_half_dim});
    ov::Tensor sin_tensor(ov::element::f32, {text_seq_len, total_half_dim});
    float* cos_data = cos_tensor.data<float>();
    float* sin_data = sin_tensor.data<float>();

    for (size_t i = 0; i < text_seq_len; ++i) {
        float* cos_row = cos_data + i * total_half_dim;
        float* sin_row = sin_data + i * total_half_dim;

        for (size_t d = 0; d < half_dim_0; ++d) {
            cos_row[d] = freqs_0[i * half_dim_0 + d].real();
            sin_row[d] = freqs_0[i * half_dim_0 + d].imag();
        }
        for (size_t d = 0; d < half_dim_1; ++d) {
            cos_row[half_dim_0 + d] = freqs_1[i * half_dim_1 + d].real();
            sin_row[half_dim_0 + d] = freqs_1[i * half_dim_1 + d].imag();
        }
        for (size_t d = 0; d < half_dim_2; ++d) {
            cos_row[half_dim_0 + half_dim_1 + d] = freqs_2[i * half_dim_2 + d].real();
            sin_row[half_dim_0 + half_dim_1 + d] = freqs_2[i * half_dim_2 + d].imag();
        }
    }

    return {cos_tensor, sin_tensor};
}

}  // anonymous namespace

namespace ov {
namespace genai {

class QwenImagePipeline : public DiffusionPipeline {
public:
    QwenImagePipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) : QwenImagePipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "Qwen2_5_VLForConditionalGeneration") {
            m_text_encoder = std::make_shared<Qwen2_5_VLForConditionalGeneration>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKLQwenImage" || vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKLQwenImage>(root_dir / "vae_decoder");
            } else {
                OPENVINO_ASSERT(false, "Unsupported pipeline type for QwenImagePipeline");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "QwenImageTransformer2DModel") {
            m_transformer = std::make_shared<QwenImageTransformer2DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        load_scheduler_shift_params(root_dir / "scheduler/scheduler_config.json");

        // initialize generation config
        initialize_generation_config("QwenImagePipeline");
    }

    QwenImagePipeline(PipelineType pipeline_type,
                      const std::filesystem::path& root_dir,
                      const std::string& device,
                      const ov::AnyMap& properties)
        : QwenImagePipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        auto updated_properties = update_adapters_in_properties(properties, &QwenImagePipeline::derived_adapters);

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "Qwen2_5_VLForConditionalGeneration") {
            m_text_encoder = std::make_shared<Qwen2_5_VLForConditionalGeneration>(root_dir / "text_encoder", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKLQwenImage" || vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKLQwenImage>(root_dir / "vae_decoder", device, *updated_properties);
            } else {
                OPENVINO_ASSERT(false, "Unsupported pipeline type for QwenImagePipeline");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "QwenImageTransformer2DModel") {
            m_transformer = std::make_shared<QwenImageTransformer2DModel>(root_dir / "transformer", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        // Load scheduler config for shift parameters
        load_scheduler_shift_params(root_dir / "scheduler/scheduler_config.json");

        // initialize generation config
        initialize_generation_config("QwenImagePipeline");
        update_adapters_from_properties(properties, m_generation_config.adapters);
    }

    QwenImagePipeline(PipelineType pipeline_type,
                      const Qwen2_5_VLForConditionalGeneration& text_encoder,
                      const QwenImageTransformer2DModel& transformer,
                      const AutoencoderKLQwenImage& vae)
        : QwenImagePipeline(pipeline_type) {
        m_text_encoder = std::make_shared<Qwen2_5_VLForConditionalGeneration>(text_encoder);
        m_vae = std::make_shared<AutoencoderKLQwenImage>(vae);
        m_transformer = std::make_shared<QwenImageTransformer2DModel>(transformer);
        initialize_generation_config("QwenImagePipeline");
    }

    QwenImagePipeline(PipelineType pipeline_type, const QwenImagePipeline& pipe)
        : QwenImagePipeline(pipeline_type) {
        m_root_dir = pipe.m_root_dir;

        m_text_encoder = std::make_shared<Qwen2_5_VLForConditionalGeneration>(*pipe.m_text_encoder);
        m_vae = std::make_shared<AutoencoderKLQwenImage>(*pipe.m_vae);
        m_transformer = std::make_shared<QwenImageTransformer2DModel>(*pipe.m_transformer);

        m_pipeline_type = pipeline_type;

        m_base_seq_len = pipe.m_base_seq_len;
        m_max_seq_len = pipe.m_max_seq_len;
        m_base_shift = pipe.m_base_shift;
        m_max_shift = pipe.m_max_shift;
        initialize_generation_config("QwenImagePipeline");
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
        auto updated_properties = update_adapters_in_properties(properties, &QwenImagePipeline::derived_adapters);
        m_text_encoder->compile(text_encode_device, *updated_properties);
        m_vae->compile(vae_device, *updated_properties);
        m_transformer->compile(denoise_device, *updated_properties);
    }

    std::shared_ptr<DiffusionPipeline> clone() override {
        OPENVINO_ASSERT(!m_root_dir.empty(), "Cannot clone pipeline without root directory");

        std::shared_ptr<AutoencoderKLQwenImage> vae = std::make_shared<AutoencoderKLQwenImage>(m_vae->clone());
        std::shared_ptr<QwenImageTransformer2DModel> transformer = std::make_shared<QwenImageTransformer2DModel>(m_transformer->clone());
        std::shared_ptr<Qwen2_5_VLForConditionalGeneration> text_encoder = m_text_encoder->clone();
        std::shared_ptr<QwenImagePipeline> pipeline = std::make_shared<QwenImagePipeline>(m_pipeline_type,
                                                                                          *text_encoder,
                                                                                          *transformer,
                                                                                          *vae);

        pipeline->m_root_dir = m_root_dir;
        pipeline->m_base_seq_len = m_base_seq_len;
        pipeline->m_max_seq_len = m_max_seq_len;
        pipeline->m_base_shift = m_base_shift;
        pipeline->m_max_shift = m_max_shift;
        pipeline->set_scheduler(Scheduler::from_config(m_root_dir / "scheduler/scheduler_config.json"));
        pipeline->set_generation_config(m_generation_config);
        return pipeline;
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        auto infer_start = std::chrono::steady_clock::now();
        auto [prompt_embeds, encoder_mask] = m_text_encoder->infer(positive_prompt, generation_config.max_sequence_length);
        auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = infer_duration;

        // Repeat for num_images_per_prompt
        m_positive_prompt_embeds = numpy_utils::repeat(prompt_embeds, generation_config.num_images_per_prompt);
        m_positive_encoder_mask = numpy_utils::repeat(encoder_mask, generation_config.num_images_per_prompt);

        // Set text encoder outputs to transformer
        m_transformer->set_hidden_states("encoder_hidden_states", m_positive_prompt_embeds);
        m_transformer->set_hidden_states("encoder_hidden_states_mask", m_positive_encoder_mask);

        if (m_transformer->get_config().guidance_embeds) {
            ov::Tensor guidance(ov::element::f32, {generation_config.num_images_per_prompt});
            std::fill_n(guidance.data<float>(), guidance.get_size(), static_cast<float>(generation_config.guidance_scale));
            m_transformer->set_hidden_states("guidance", guidance);
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

        // Generate random noise latents
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
        OPENVINO_ASSERT(!mask_image, "QwenImagePipeline does not support mask_image/inpainting");
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();
        m_custom_generation_config = m_generation_config;
        m_custom_generation_config.update_generation_config(properties);

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        if (m_custom_generation_config.height < 0) {
            m_custom_generation_config.height = m_transformer->get_config().default_sample_size * vae_scale_factor;
        }
        if (m_custom_generation_config.width < 0) {
            m_custom_generation_config.width = m_transformer->get_config().default_sample_size * vae_scale_factor;
        }

        check_inputs(m_custom_generation_config, initial_image);

        set_lora_adapters(m_custom_generation_config.adapters);

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        compute_hidden_states(positive_prompt, m_custom_generation_config);

        // Encode negative prompt if true_cfg is active
        const float true_cfg_scale = m_custom_generation_config.guidance_scale;
        const bool do_true_cfg = true_cfg_scale > 1.0f && m_custom_generation_config.negative_prompt.has_value();

        ov::Tensor negative_prompt_embeds, negative_encoder_mask;
        if (do_true_cfg) {
            auto neg_infer_start = std::chrono::steady_clock::now();
            auto [neg_embeds, neg_mask] = m_text_encoder->infer(
                m_custom_generation_config.negative_prompt.value(),
                m_custom_generation_config.max_sequence_length);
            auto neg_infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - neg_infer_start).count();
            m_perf_metrics.encoder_inference_duration["text_encoder"] += neg_infer_duration;

            negative_prompt_embeds = numpy_utils::repeat(neg_embeds, m_custom_generation_config.num_images_per_prompt);
            negative_encoder_mask = numpy_utils::repeat(neg_mask, m_custom_generation_config.num_images_per_prompt);
        }

        ov::Tensor latents;
        std::tie(latents, std::ignore, std::ignore, std::ignore) = prepare_latents(initial_image, m_custom_generation_config);

        // Compute image_seq_len for timestep scheduling
        const size_t latent_height = m_custom_generation_config.height / vae_scale_factor / 2;
        const size_t latent_width = m_custom_generation_config.width / vae_scale_factor / 2;
        const size_t image_seq_len = latent_height * latent_width;

        // Compute rotary embeddings on the host side.
        // TODO: for diffusers this is done inside QwenImageTransformer2DModel.forward()
        const auto& axes_dims = m_transformer->get_config().axes_dims_rope;
        const size_t text_seq_len = m_positive_prompt_embeds.get_shape()[1];

        auto [img_cos, img_sin] = qwen_image_compute_rotary_embeddings(1, latent_height, latent_width, axes_dims);
        auto [txt_cos, txt_sin] = qwen_image_compute_text_rotary_embeddings(text_seq_len, latent_height, latent_width, axes_dims);

        m_transformer->set_hidden_states("img_cos", img_cos);
        m_transformer->set_hidden_states("img_sin", img_sin);
        m_transformer->set_hidden_states("txt_cos", txt_cos);
        m_transformer->set_hidden_states("txt_sin", txt_sin);

        const double mu = qwen_image_calculate_shift(image_seq_len, m_base_seq_len, m_max_seq_len, m_base_shift, m_max_shift);
        m_scheduler->set_timesteps_with_mu(mu, m_custom_generation_config.num_inference_steps, 1.0f);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        // Denoising loop
        ov::Tensor timestep_tensor(ov::element::f32, {m_custom_generation_config.num_images_per_prompt});
        float* timestep_data = timestep_tensor.data<float>();

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();

            for (size_t i = 0; i < m_custom_generation_config.num_images_per_prompt; ++i) {
                timestep_data[i] = timesteps[inference_step] / 1000.0f;
            }

            auto infer_start = std::chrono::steady_clock::now();
            ov::Tensor noise_pred = m_transformer->infer(latents, timestep_tensor);
            auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

            // classifier-free guidance
            if (do_true_cfg) {
                ov::Tensor pos_noise_pred(noise_pred.get_element_type(), noise_pred.get_shape());
                noise_pred.copy_to(pos_noise_pred);

                const ov::Shape& pred_shape = pos_noise_pred.get_shape();
                const size_t batch_seq = pred_shape[0] * pred_shape[1];
                const size_t channels = pred_shape[2];

                std::vector<float> cond_norms(batch_seq);
                const float* orig_pos_data = pos_noise_pred.data<float>();
                for (size_t t = 0; t < batch_seq; ++t) {
                    float sum_sq = 0.0f;
                    for (size_t c = 0; c < channels; ++c) {
                        float val = orig_pos_data[t * channels + c];
                        sum_sq += val * val;
                    }
                    cond_norms[t] = std::sqrt(sum_sq);
                }

                m_transformer->set_hidden_states("encoder_hidden_states", negative_prompt_embeds);
                m_transformer->set_hidden_states("encoder_hidden_states_mask", negative_encoder_mask);

                // Compute and set text rotary embeddings for negative prompt
                const size_t neg_text_seq_len = negative_prompt_embeds.get_shape()[1];
                if (neg_text_seq_len != text_seq_len) {
                    auto [neg_txt_cos, neg_txt_sin] = qwen_image_compute_text_rotary_embeddings(
                        neg_text_seq_len, latent_height, latent_width, axes_dims);
                    m_transformer->set_hidden_states("txt_cos", neg_txt_cos);
                    m_transformer->set_hidden_states("txt_sin", neg_txt_sin);
                }

                auto neg_infer_start = std::chrono::steady_clock::now();
                ov::Tensor neg_noise_pred = m_transformer->infer(latents, timestep_tensor);
                auto neg_infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - neg_infer_start);
                m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(neg_infer_duration));

                float* pos_data = pos_noise_pred.data<float>();
                const float* neg_data = neg_noise_pred.data<float>();

                for (size_t i = 0; i < pos_noise_pred.get_size(); ++i) {
                    pos_data[i] = neg_data[i] + true_cfg_scale * (pos_data[i] - neg_data[i]);
                }

                // Norm rescaling: noise_pred = comb_pred * (cond_norm / noise_norm)
                for (size_t t = 0; t < batch_seq; ++t) {
                    float noise_norm_sq = 0.0f;
                    for (size_t c = 0; c < channels; ++c) {
                        float val = pos_data[t * channels + c];
                        noise_norm_sq += val * val;
                    }
                    float noise_norm = std::sqrt(noise_norm_sq);
                    float scale = (noise_norm > 1e-8f) ? (cond_norms[t] / noise_norm) : 1.0f;
                    for (size_t c = 0; c < channels; ++c) {
                        pos_data[t * channels + c] *= scale;
                    }
                }

                noise_pred = pos_noise_pred;

                // Restore positive prompt embeddings for the next step
                m_transformer->set_hidden_states("encoder_hidden_states", m_positive_prompt_embeds);
                m_transformer->set_hidden_states("encoder_hidden_states_mask", m_positive_encoder_mask);
                if (neg_text_seq_len != text_seq_len) {
                    m_transformer->set_hidden_states("txt_cos", txt_cos);
                    m_transformer->set_hidden_states("txt_sin", txt_sin);
                }
            }

            auto scheduler_step_result = m_scheduler->step(noise_pred, latents, inference_step, m_custom_generation_config.generator);
            latents = scheduler_step_result["latent"];

            if (callback_ptr && callback_ptr->has_callback() && callback_ptr->write(inference_step, timesteps.size(), latents) == CallbackStatus::STOP) {
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

        // Unpack latents: (B, seq_len, C*4) -> (B, C, H, W)
        ov::Tensor final_latents = unpack_latents(latents, m_custom_generation_config.height, m_custom_generation_config.width, vae_scale_factor);

        // Apply VAE latent denormalization: latents = latents / latents_std + latents_mean
        // The final_latents shape is (B, C, H, W), need to add temporal dim for 3D VAE: (B, C, 1, H, W)
        apply_latent_denormalization(final_latents);

        // Reshape to 5D for the 3D VAE: (B, C, H, W) -> (B, C, 1, H, W)
        const ov::Shape& lat_shape = final_latents.get_shape();
        ov::Tensor vae_input(final_latents.get_element_type(), {lat_shape[0], lat_shape[1], 1, lat_shape[2], lat_shape[3]});
        std::memcpy(vae_input.data<float>(), final_latents.data<float>(), final_latents.get_byte_size());

        const auto decode_start = std::chrono::steady_clock::now();
        auto image = m_vae->decode(vae_input);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start).count();

        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        return image;
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        ov::Tensor latent_copy(latent.get_element_type(), latent.get_shape());
        latent.copy_to(latent_copy);

        ov::Tensor final_latents = unpack_latents(latent_copy, m_custom_generation_config.height, m_custom_generation_config.width, vae_scale_factor);
        apply_latent_denormalization(final_latents);

        // Reshape to 5D: (B, C, H, W) -> (B, C, 1, H, W)
        const ov::Shape& lat_shape = final_latents.get_shape();
        ov::Tensor vae_input(final_latents.get_element_type(), {lat_shape[0], lat_shape[1], 1, lat_shape[2], lat_shape[3]});
        std::memcpy(vae_input.data<float>(), final_latents.data<float>(), final_latents.get_byte_size());

        return m_vae->decode(vae_input);
    }

    ImageGenerationPerfMetrics get_performance_metrics() override {
        m_perf_metrics.load_time = m_load_time_ms;
        return m_perf_metrics;
    }

protected:
    explicit QwenImagePipeline(PipelineType pipeline_type) :
        DiffusionPipeline(pipeline_type) {}

    void initialize_generation_config(const std::string& class_name) override {
        OPENVINO_ASSERT(m_transformer != nullptr);
        OPENVINO_ASSERT(m_vae != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();

        m_generation_config.height = transformer_config.default_sample_size * vae_scale_factor;
        m_generation_config.width = transformer_config.default_sample_size * vae_scale_factor;

        if (class_name == "QwenImagePipeline") {
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
        OPENVINO_ASSERT((height % (vae_scale_factor * 2) == 0 || height < 0) &&
                        (width % (vae_scale_factor * 2) == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ",
                        vae_scale_factor * 2);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.height, generation_config.width);

        OPENVINO_ASSERT(generation_config.max_sequence_length <= 1024,
                        "'max_sequence_length' must be less or equal to 1024");

        OPENVINO_ASSERT(generation_config.prompt_2 == std::nullopt, "Prompt 2 is not used by QwenImagePipeline");
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by QwenImagePipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by QwenImagePipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by QwenImagePipeline");
        OPENVINO_ASSERT(!initial_image, "QwenImagePipeline does not support initial_image");
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
        OPENVINO_THROW("blend_latents is not supported by QwenImagePipeline");
    }

    static std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters) {
        return std::nullopt;
    }

private:
    void load_scheduler_shift_params(const std::filesystem::path& scheduler_config_path) {
        std::ifstream file(scheduler_config_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open scheduler config: ", scheduler_config_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        if (data.contains("base_image_seq_len"))
            m_base_seq_len = data["base_image_seq_len"].get<size_t>();
        if (data.contains("max_image_seq_len"))
            m_max_seq_len = data["max_image_seq_len"].get<size_t>();
        if (data.contains("base_shift"))
            m_base_shift = data["base_shift"].get<double>();
        if (data.contains("max_shift"))
            m_max_shift = data["max_shift"].get<double>();
    }

    // Apply VAE latent denormalization: latents = latents * latents_std + latents_mean
    // (matching diffusers: latents_std_inv = 1.0 / latents_std; latents = latents / latents_std_inv + latents_mean)
    // latents shape: (B, C, H, W) where C == z_dim
    void apply_latent_denormalization(ov::Tensor& latents) const {
        const auto& vae_config = m_vae->get_config();
        if (vae_config.latents_mean.empty() || vae_config.latents_std.empty()) {
            return;
        }

        const ov::Shape& shape = latents.get_shape();
        const size_t batch_size = shape[0];
        const size_t channels = shape[1];
        const size_t spatial = shape[2] * shape[3];

        OPENVINO_ASSERT(channels <= vae_config.latents_mean.size() && channels <= vae_config.latents_std.size(),
                        "Latent channels (", channels, ") exceed latents_mean/std size");

        float* data = latents.data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                const float std_val = vae_config.latents_std[c];
                const float mean = vae_config.latents_mean[c];
                float* channel_data = data + (b * channels + c) * spatial;
                for (size_t i = 0; i < spatial; ++i) {
                    channel_data[i] = channel_data[i] * std_val + mean;
                }
            }
        }
    }

    std::shared_ptr<Qwen2_5_VLForConditionalGeneration> m_text_encoder;
    std::shared_ptr<AutoencoderKLQwenImage> m_vae;
    std::shared_ptr<QwenImageTransformer2DModel> m_transformer;

    ov::Tensor m_positive_prompt_embeds;
    ov::Tensor m_positive_encoder_mask;

    ImageGenerationConfig m_custom_generation_config;
    ImageGenerationPerfMetrics m_perf_metrics;

    size_t m_base_seq_len = 256;
    size_t m_max_seq_len = 4096;
    double m_base_shift = 0.5;
    double m_max_shift = 1.15;
};

}  // namespace genai
}  // namespace ov
