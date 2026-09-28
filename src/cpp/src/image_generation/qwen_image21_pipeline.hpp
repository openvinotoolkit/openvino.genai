// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/threaded_callback.hpp"

#include "openvino/genai/image_generation/autoencoder_kl_qwen_image.hpp"
#include "openvino/genai/image_generation/qwen3_vl_for_conditional_generation.hpp"
#include "openvino/genai/image_generation/qwen_image21_transformer_2d_model.hpp"
#include "utils.hpp"

namespace {

// Qwen-Image 2.1 consumes latents unpatched, so packing is a plain spatial flatten:
// (B, C, ...spatial) -> (B, spatial, C).
inline ov::Tensor qwen_image21_pack_latents(const ov::Tensor latents) {
    const ov::Shape& shape = latents.get_shape();
    const size_t batch_size = shape[0], channels = shape[1];
    const size_t spatial = std::accumulate(shape.begin() + 2, shape.end(), size_t{1}, std::multiplies<size_t>());

    ov::Tensor packed(latents.get_element_type(), {batch_size, spatial, channels});
    const float* src_data = latents.data<const float>();
    float* dst_data = packed.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            const float* src_channel = src_data + (b * channels + c) * spatial;
            float* dst_channel = dst_data + b * spatial * channels + c;
            for (size_t s = 0; s < spatial; ++s) {
                dst_channel[s * channels] = src_channel[s];
            }
        }
    }

    return packed;
}

// Inverse of qwen_image21_pack_latents, with the temporal axis the 3D VAE expects:
// (B, H * W, C) -> (B, C, 1, H, W).
inline ov::Tensor qwen_image21_unpack_latents(const ov::Tensor latents, const size_t height, const size_t width) {
    const ov::Shape& shape = latents.get_shape();
    const size_t batch_size = shape[0], spatial = shape[1], channels = shape[2];
    OPENVINO_ASSERT(spatial == height * width,
                    "Packed latent sequence length (", spatial, ") does not match ", height, "x", width);

    ov::Tensor unpacked(latents.get_element_type(), {batch_size, channels, 1, height, width});
    const float* src_data = latents.data<const float>();
    float* dst_data = unpacked.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            const float* src_channel = src_data + b * spatial * channels + c;
            float* dst_channel = dst_data + (b * channels + c) * spatial;
            for (size_t s = 0; s < spatial; ++s) {
                dst_channel[s] = src_channel[s * channels];
            }
        }
    }

    return unpacked;
}

// latents = latents * latents_std + latents_mean, over the channel axis of a (B, C, 1, H, W) tensor.
inline void qwen_image21_denormalize_latents(ov::Tensor& latents,
                                             const std::vector<float>& latents_mean,
                                             const std::vector<float>& latents_std) {
    const ov::Shape& shape = latents.get_shape();
    const size_t batch_size = shape[0], channels = shape[1], spatial = shape[2] * shape[3] * shape[4];

    OPENVINO_ASSERT(channels <= latents_mean.size() && channels <= latents_std.size(),
                    "Latent channels (", channels, ") exceed latents_mean/latents_std size");

    float* data = latents.data<float>();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            const float mean = latents_mean[c], std_value = latents_std[c];
            float* channel_data = data + (b * channels + c) * spatial;
            for (size_t i = 0; i < spatial; ++i) {
                channel_data[i] = channel_data[i] * std_value + mean;
            }
        }
    }
}

// Inverse of qwen_image21_denormalize_latents, applied to freshly encoded condition latents.
inline void qwen_image21_normalize_latents(ov::Tensor& latents,
                                           const std::vector<float>& latents_mean,
                                           const std::vector<float>& latents_std) {
    const ov::Shape& shape = latents.get_shape();
    const size_t batch_size = shape[0], channels = shape[1], spatial = shape[2] * shape[3] * shape[4];

    OPENVINO_ASSERT(channels <= latents_mean.size() && channels <= latents_std.size(),
                    "Latent channels (", channels, ") exceed latents_mean/latents_std size");

    float* data = latents.data<float>();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            const float mean = latents_mean[c], std_inverse = 1.0f / latents_std[c];
            float* channel_data = data + (b * channels + c) * spatial;
            for (size_t i = 0; i < spatial; ++i) {
                channel_data[i] = (channel_data[i] - mean) * std_inverse;
            }
        }
    }
}

}  // anonymous namespace

namespace ov {
namespace genai {

class QwenImage21Pipeline : public DiffusionPipeline {
public:
    QwenImage21Pipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir)
        : QwenImage21Pipeline(pipeline_type) {
        m_root_dir = root_dir;
        const nlohmann::json data = read_model_index(root_dir);

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        m_transformer = std::make_shared<QwenImage21Transformer2DModel>(root_dir / transformer_subfolder(data));

        const std::filesystem::path text_encoder_dir = root_dir / text_encoder_subfolder(data);
        const std::filesystem::path vae_decoder_dir = root_dir / vae_subfolder(data);
        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            m_text_encoder = std::make_shared<Qwen3VLForConditionalGeneration>(text_encoder_dir,
                                                                              root_dir / VISION_ENCODER_SUBFOLDER,
                                                                              root_dir / TEXT_ENCODER_I2I_SUBFOLDER);
            m_vae = std::make_shared<AutoencoderKLQwenImage>(root_dir / VAE_ENCODER_SUBFOLDER, vae_decoder_dir);
        } else {
            m_text_encoder = std::make_shared<Qwen3VLForConditionalGeneration>(text_encoder_dir);
            m_vae = std::make_shared<AutoencoderKLQwenImage>(vae_decoder_dir);
        }

        initialize_generation_config("QwenImage21Pipeline");
    }

    QwenImage21Pipeline(PipelineType pipeline_type,
                        const std::filesystem::path& root_dir,
                        const std::string& device,
                        const ov::AnyMap& properties)
        : QwenImage21Pipeline(pipeline_type) {
        m_root_dir = root_dir;
        const nlohmann::json data = read_model_index(root_dir);

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        auto updated_properties = update_adapters_in_properties(properties, &QwenImage21Pipeline::derived_adapters);

        m_transformer = std::make_shared<QwenImage21Transformer2DModel>(root_dir / transformer_subfolder(data),
                                                                       device, *updated_properties);

        const std::filesystem::path text_encoder_dir = root_dir / text_encoder_subfolder(data);
        const std::filesystem::path vae_decoder_dir = root_dir / vae_subfolder(data);
        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            m_text_encoder = std::make_shared<Qwen3VLForConditionalGeneration>(text_encoder_dir,
                                                                              root_dir / VISION_ENCODER_SUBFOLDER,
                                                                              root_dir / TEXT_ENCODER_I2I_SUBFOLDER);
            m_vae = std::make_shared<AutoencoderKLQwenImage>(root_dir / VAE_ENCODER_SUBFOLDER, vae_decoder_dir,
                                                            device, *updated_properties);
        } else {
            m_text_encoder = std::make_shared<Qwen3VLForConditionalGeneration>(text_encoder_dir);
            m_vae = std::make_shared<AutoencoderKLQwenImage>(vae_decoder_dir, device, *updated_properties);
        }
        m_text_encoder->compile(device, *updated_properties);

        initialize_generation_config("QwenImage21Pipeline");
        update_adapters_from_properties(properties, m_generation_config.adapters);
    }

    QwenImage21Pipeline(PipelineType pipeline_type,
                        const Qwen3VLForConditionalGeneration& text_encoder,
                        const QwenImage21Transformer2DModel& transformer,
                        const AutoencoderKLQwenImage& vae)
        : QwenImage21Pipeline(pipeline_type) {
        m_text_encoder = std::make_shared<Qwen3VLForConditionalGeneration>(text_encoder);
        m_vae = std::make_shared<AutoencoderKLQwenImage>(vae);
        m_transformer = std::make_shared<QwenImage21Transformer2DModel>(transformer);
        initialize_generation_config("QwenImage21Pipeline");
    }

    // The text encoder and the transformer consume sequences whose length depends on the prompt, so only the VAE
    // is reshaped to static shapes.
    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        check_image_size(height, width);
        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            // The condition image is encoded once, at the resolution derived from the requested aspect ratio rather
            // than from the requested output size.
            const Qwen3VLForConditionalGeneration::ImageSize condition_size =
                Qwen3VLForConditionalGeneration::calculate_dimensions(
                    DEFAULT_OUTPUT_RESOLUTION * DEFAULT_OUTPUT_RESOLUTION,
                    static_cast<double>(width) / static_cast<double>(height));
            m_vae->reshape_encoder(1, static_cast<int>(condition_size.height), static_cast<int>(condition_size.width));
        }
        m_vae->reshape_decoder(num_images_per_prompt, height, width);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);
        auto updated_properties = update_adapters_in_properties(properties, &QwenImage21Pipeline::derived_adapters);
        m_text_encoder->compile(text_encode_device, *updated_properties);
        m_vae->compile(vae_device, *updated_properties);
        m_transformer->compile(denoise_device, *updated_properties);
    }

    std::shared_ptr<DiffusionPipeline> clone() override {
        OPENVINO_ASSERT(!m_root_dir.empty(), "Cannot clone pipeline without root directory");

        AutoencoderKLQwenImage vae = m_vae->clone();
        QwenImage21Transformer2DModel transformer = m_transformer->clone();
        std::shared_ptr<Qwen3VLForConditionalGeneration> text_encoder = m_text_encoder->clone();

        std::shared_ptr<QwenImage21Pipeline> pipeline =
            std::make_shared<QwenImage21Pipeline>(m_pipeline_type, *text_encoder, transformer, vae);

        pipeline->m_root_dir = m_root_dir;
        pipeline->set_scheduler(Scheduler::from_config(m_root_dir / "scheduler/scheduler_config.json"));
        pipeline->set_generation_config(m_generation_config);
        return pipeline;
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        const auto infer_start = std::chrono::steady_clock::now();

        // The condition image is preprocessed by generate() before this call because the base interface passes
        // the prompt only.
        const auto encode = [&](const std::string& prompt) {
            return numpy_utils::repeat(m_condition_image
                                           ? m_text_encoder->infer(prompt, m_condition_image, generation_config.max_sequence_length)
                                           : m_text_encoder->infer(prompt, generation_config.max_sequence_length),
                                       generation_config.num_images_per_prompt);
        };

        m_positive_prompt_embeds = encode(positive_prompt);
        m_positive_image_pad_mask = m_text_encoder->get_image_pad_mask();

        if (do_true_cfg(generation_config)) {
            m_negative_prompt_embeds = encode(*generation_config.negative_prompt);
            m_negative_image_pad_mask = m_text_encoder->get_image_pad_mask();
        }

        m_perf_metrics.encoder_inference_duration["text_encoder"] =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const ov::Shape latent_shape{generation_config.num_images_per_prompt,
                                     m_transformer->get_config().in_channels,
                                     generation_config.height / vae_scale_factor,
                                     generation_config.width / vae_scale_factor};

        const ov::Tensor noise = generation_config.generator->randn_tensor(latent_shape);

        ov::Tensor condition_latents;
        if (m_condition_image) {
            const auto encode_start = std::chrono::steady_clock::now();
            condition_latents = m_vae->encode(to_vae_input(m_condition_image));
            m_perf_metrics.vae_encoder_inference_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - encode_start).count();

            const auto& vae_config = m_vae->get_config();
            qwen_image21_normalize_latents(condition_latents, vae_config.latents_mean, vae_config.latents_std);
            condition_latents = numpy_utils::repeat(qwen_image21_pack_latents(condition_latents),
                                                    generation_config.num_images_per_prompt);
        }

        return std::make_tuple(qwen_image21_pack_latents(noise), ov::Tensor(), condition_latents, noise);
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
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();
        m_custom_generation_config = m_generation_config;
        m_custom_generation_config.update_generation_config(properties);

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        m_condition_image = ov::Tensor();

        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            OPENVINO_ASSERT(initial_image, "'initial_image' must not be empty for Image 2 image pipeline");
            const ov::Shape& image_shape = initial_image.get_shape();
            OPENVINO_ASSERT(image_shape.size() == 4 && image_shape[0] == 1 && image_shape[1] > 0 &&
                                image_shape[2] > 0 && image_shape[3] == 3,
                            "'initial_image' must have [1, height, width, 3] shape with non-zero spatial dimensions, got ",
                            image_shape);
            const Qwen3VLForConditionalGeneration::ImageSize condition_size =
                Qwen3VLForConditionalGeneration::calculate_dimensions(
                    DEFAULT_OUTPUT_RESOLUTION * DEFAULT_OUTPUT_RESOLUTION,
                    static_cast<double>(image_shape[2]) / static_cast<double>(image_shape[1]));

            // Without an explicit resolution the target follows the condition image's aspect ratio.
            if (m_custom_generation_config.height < 0) {
                m_custom_generation_config.height = static_cast<int64_t>(condition_size.height);
            }
            if (m_custom_generation_config.width < 0) {
                m_custom_generation_config.width = static_cast<int64_t>(condition_size.width);
            }

            // One resize and normalization feed both the vision tower and the VAE: Qwen3-VL's processor uses
            // image_mean = image_std = 0.5, which reduces to the (pixel / 127.5) - 1 the VAE expects.
            m_condition_image = m_image_processor->execute(
                m_image_resizer->execute(initial_image, condition_size.height, condition_size.width));
            m_condition_block = {condition_size.height / vae_scale_factor, condition_size.width / vae_scale_factor};
        }

        if (m_custom_generation_config.height < 0) {
            m_custom_generation_config.height = DEFAULT_OUTPUT_RESOLUTION;
        }
        if (m_custom_generation_config.width < 0) {
            m_custom_generation_config.width = DEFAULT_OUTPUT_RESOLUTION;
        }

        check_inputs(m_custom_generation_config, initial_image);
        OPENVINO_ASSERT(!mask_image, "QwenImage21Pipeline does not support mask_image/inpainting");

        set_lora_adapters(m_custom_generation_config.adapters);

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        if (auto callback_iter = properties.find(ov::genai::callback.name()); callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(
                callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        compute_hidden_states(positive_prompt, m_custom_generation_config);

        const size_t latent_height = m_custom_generation_config.height / vae_scale_factor;
        const size_t latent_width = m_custom_generation_config.width / vae_scale_factor;
        const size_t image_seq_len = latent_height * latent_width;

        m_scheduler->set_timesteps_with_mu(m_scheduler->calculate_shift(image_seq_len),
                                           m_custom_generation_config.num_inference_steps,
                                           1.0f);
        const std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        ov::Tensor latents, condition_latents;
        std::tie(latents, std::ignore, condition_latents, std::ignore) = prepare_latents(initial_image, m_custom_generation_config);

        std::vector<LatentBlock> blocks;
        if (condition_latents) {
            blocks.push_back(m_condition_block);
        }
        blocks.push_back({latent_height, latent_width});

        const JointSequenceInputs positive_inputs =
            build_joint_sequence_inputs(m_positive_prompt_embeds, m_positive_image_pad_mask, blocks);
        const bool true_cfg = do_true_cfg(m_custom_generation_config);
        const JointSequenceInputs negative_inputs =
            true_cfg ? build_joint_sequence_inputs(m_negative_prompt_embeds, m_negative_image_pad_mask, blocks)
                     : JointSequenceInputs();

        ov::Tensor timestep_tensor(ov::element::f32, {m_custom_generation_config.num_images_per_prompt});

        // Condition latents are static context tokens prepended to the denoised ones, so the joint input buffer is
        // built once and every step refreshes only its trailing target region.
        ov::Tensor joint_input;
        size_t joint_row = 0, target_row = 0;
        if (condition_latents) {
            joint_input = numpy_utils::concat(condition_latents, latents, 1);
            target_row = latents.get_shape()[1] * latents.get_shape()[2];
            joint_row = joint_input.get_shape()[1] * joint_input.get_shape()[2];
        }

        // The joint-sequence inputs do not depend on the timestep, so they are bound once. Only CFG has to swap
        // them between the two passes of a step.
        set_transformer_inputs(m_positive_prompt_embeds, positive_inputs);
        const float true_cfg_scale = m_custom_generation_config.guidance_scale;

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            const auto step_start = std::chrono::steady_clock::now();

            std::fill_n(timestep_tensor.data<float>(), timestep_tensor.get_size(), timesteps[inference_step] / 1000.0f);

            if (joint_input) {
                float* target_data = joint_input.data<float>() + joint_row - target_row;
                const float* latent_data = latents.data<const float>();
                for (size_t batch = 0; batch < m_custom_generation_config.num_images_per_prompt; ++batch) {
                    std::copy_n(latent_data + batch * target_row, target_row, target_data + batch * joint_row);
                }
            }
            const ov::Tensor model_input = joint_input ? joint_input : latents;

            ov::Tensor noise_pred = infer_transformer(model_input, timestep_tensor, image_seq_len);

            if (true_cfg) {
                set_transformer_inputs(m_negative_prompt_embeds, negative_inputs);
                const ov::Tensor negative_noise_pred = infer_transformer(model_input, timestep_tensor, image_seq_len);
                set_transformer_inputs(m_positive_prompt_embeds, positive_inputs);

                float* positive_data = noise_pred.data<float>();
                const float* negative_data = negative_noise_pred.data<const float>();
                for (size_t i = 0; i < noise_pred.get_size(); ++i) {
                    positive_data[i] = negative_data[i] + true_cfg_scale * (positive_data[i] - negative_data[i]);
                }
            }

            latents = m_scheduler->step(noise_pred, latents, inference_step, m_custom_generation_config.generator)["latent"];

            const bool stop_requested = callback_ptr && callback_ptr->has_callback() &&
                callback_ptr->write(inference_step, timesteps.size(), latents) == CallbackStatus::STOP;

            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(
                MicroSeconds(ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start)));

            if (stop_requested) {
                callback_ptr->end();
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
                return ov::Tensor(ov::element::u8, {});
            }
        }

        if (callback_ptr != nullptr) {
            callback_ptr->end();
        }

        ov::Tensor image = decode(latents);
        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        return image;
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        ov::Tensor vae_input = qwen_image21_unpack_latents(latent,
                                                           m_custom_generation_config.height / vae_scale_factor,
                                                           m_custom_generation_config.width / vae_scale_factor);

        const auto& vae_config = m_vae->get_config();
        qwen_image21_denormalize_latents(vae_input, vae_config.latents_mean, vae_config.latents_std);

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor image = m_vae->decode(vae_input);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start).count();
        return image;
    }

    ImageGenerationPerfMetrics get_performance_metrics() override {
        m_perf_metrics.load_time = m_load_time_ms;
        return m_perf_metrics;
    }

protected:
    explicit QwenImage21Pipeline(PipelineType pipeline_type) : DiffusionPipeline(pipeline_type) {
        OPENVINO_ASSERT(pipeline_type != PipelineType::INPAINTING,
                        "QwenImage21Pipeline does not support inpainting");
    }

    void initialize_generation_config(const std::string& class_name) override {
        OPENVINO_ASSERT(class_name == "QwenImage21Pipeline",
                        "Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");

        // Height and width stay unset so that generate() can derive them: from the condition image's aspect
        // ratio for image-to-image, and from DEFAULT_OUTPUT_RESOLUTION otherwise.
        m_generation_config = ImageGenerationConfig();
        m_generation_config.guidance_scale = 4.0f;
        m_generation_config.num_inference_steps = 50;
        m_generation_config.max_sequence_length = 8192;
        m_generation_config.strength = 1.0f;
    }

    void check_image_size(const int height, const int width) const override {
        const int64_t multiple_of = static_cast<int64_t>(m_vae->get_vae_scale_factor()) * 2;
        OPENVINO_ASSERT((height % multiple_of == 0 || height < 0) && (width % multiple_of == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ", multiple_of);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.height, generation_config.width);

        OPENVINO_ASSERT(generation_config.prompt_2 == std::nullopt, "Prompt 2 is not used by QwenImage21Pipeline");
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by QwenImage21Pipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by QwenImage21Pipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by QwenImage21Pipeline");

        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            OPENVINO_ASSERT(initial_image, "'initial_image' must not be empty for Image 2 image pipeline");
            OPENVINO_ASSERT(m_text_encoder->has_vision_tower(),
                            "Image 2 image generation requires the '", VISION_ENCODER_SUBFOLDER, "' and '",
                            TEXT_ENCODER_I2I_SUBFOLDER, "' submodels");
            // Qwen-Image 2.1 does not support variable strength: the condition image enters the joint sequence as
            // its own token block instead of being blended with noise, so partial denoising is not applicable.
            // Aligned with diffusers QwenImage21Pipeline which does not accept 'strength'.
        } else {
            OPENVINO_ASSERT(generation_config.strength == 1.0f,
                            "'strength' generation parameter must be 1.0f for Text 2 image pipeline");
            OPENVINO_ASSERT(!initial_image, "'initial_image' must be empty for Text 2 image pipeline");
        }
    }

    size_t get_config_in_channels() const override {
        return m_transformer->get_config().in_channels;
    }

    void blend_latents(ov::Tensor latents,
                       const ov::Tensor image_latent,
                       const ov::Tensor mask,
                       const ov::Tensor noise,
                       size_t inference_step) override {
        OPENVINO_THROW("blend_latents is not supported by QwenImage21Pipeline");
    }

    static std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters) {
        return std::nullopt;
    }

private:
    // Host-precomputed transformer graph inputs. They depend on the prompt length and the target resolution
    // only, so they are built once per prompt and reused across denoising steps.
    struct JointSequenceInputs {
        ov::Tensor cos, sin, gather_idx, attn_mask, modulation_mask;
    };

    // Latent extent of one image block of the joint sequence.
    struct LatentBlock {
        size_t height = 0;
        size_t width = 0;

        size_t area() const {
            return height * width;
        }
    };

    static constexpr int64_t DEFAULT_OUTPUT_RESOLUTION = 1024;
    static constexpr double ROPE_THETA = 10000.0;
    // The transformer expands one joint-sequence image slot into a 2x2 group of latent tokens.
    static constexpr size_t LATENTS_PER_IMAGE_SLOT = 4;
    static constexpr const char* VISION_ENCODER_SUBFOLDER = "vision_encoder";
    static constexpr const char* TEXT_ENCODER_I2I_SUBFOLDER = "text_encoder_i2i";
    static constexpr const char* VAE_ENCODER_SUBFOLDER = "vae_encoder";
    // Qwen-Image 2.1 encodes and decodes RGBA; a fully opaque alpha channel normalizes to 1.0.
    static constexpr size_t VAE_IMAGE_CHANNELS = 4;

    static nlohmann::json read_model_index(const std::filesystem::path& root_dir) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);
        return nlohmann::json::parse(file);
    }

    static std::filesystem::path text_encoder_subfolder(const nlohmann::json& model_index) {
        const std::string text_encoder = model_index["text_encoder"][1].get<std::string>();
        OPENVINO_ASSERT(text_encoder == "Qwen3VLForConditionalGeneration",
                        "Unsupported '", text_encoder, "' text encoder type");
        return "text_encoder";
    }

    static std::filesystem::path vae_subfolder(const nlohmann::json& model_index) {
        const std::string vae = model_index["vae"][1].get<std::string>();
        OPENVINO_ASSERT(vae == "AutoencoderKLQwenImage21", "Unsupported '", vae, "' VAE decoder type");
        return "vae_decoder";
    }

    static std::filesystem::path transformer_subfolder(const nlohmann::json& model_index) {
        const std::string transformer = model_index["transformer"][1].get<std::string>();
        OPENVINO_ASSERT(transformer == "QwenImage21Transformer2DModel",
                        "Unsupported '", transformer, "' Transformer type");
        return "transformer";
    }

    static bool do_true_cfg(const ImageGenerationConfig& generation_config) {
        return generation_config.guidance_scale > 1.0f && generation_config.negative_prompt.has_value();
    }

    // (1, 3, H, W) normalized RGB -> (1, 4, 1, H, W) RGBA with the temporal axis the 3D VAE expects.
    static ov::Tensor to_vae_input(const ov::Tensor normalized_image) {
        const ov::Shape& shape = normalized_image.get_shape();
        const size_t batch_size = shape[0], channels = shape[1], spatial = shape[2] * shape[3];
        OPENVINO_ASSERT(channels < VAE_IMAGE_CHANNELS,
                        "Condition image must have fewer than ", VAE_IMAGE_CHANNELS, " channels, got ", channels);

        ov::Tensor vae_input(ov::element::f32, {batch_size, VAE_IMAGE_CHANNELS, 1, shape[2], shape[3]});
        const float* src_data = normalized_image.data<const float>();
        float* dst_data = vae_input.data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            float* image_data = dst_data + b * VAE_IMAGE_CHANNELS * spatial;
            std::copy_n(src_data + b * channels * spatial, channels * spatial, image_data);
            std::fill_n(image_data + channels * spatial, (VAE_IMAGE_CHANNELS - channels) * spatial, 1.0f);
        }

        return vae_input;
    }

    // Builds the rotary table, the joint-sequence gather index, the dense block-causal attention mask and the
    // modulation mask. The prompt reserves one slot per contiguous run of vision placeholders, and the target
    // image appends its own run; every slot expands into the latent tokens of the matching image block.
    JointSequenceInputs build_joint_sequence_inputs(const ov::Tensor prompt_embeds,
                                                    const ov::Tensor prompt_image_pad_mask,
                                                    const std::vector<LatentBlock>& blocks) const {
        OPENVINO_ASSERT(!blocks.empty(), "Joint sequence must contain at least the target image block");
        const size_t batch_size = prompt_embeds.get_shape()[0];
        const size_t text_length = prompt_embeds.get_shape()[1];

        std::vector<char> slots(text_length + blocks.back().area() / LATENTS_PER_IMAGE_SLOT, 1);
        std::copy_n(prompt_image_pad_mask.data<const bool>(), text_length, slots.begin());

        std::vector<size_t> repeats(slots.size(), 1);
        std::vector<int32_t> slot_block(slots.size(), -1);
        size_t block_index = 0;
        for (size_t slot = 0; slot < slots.size();) {
            if (slots[slot] == 0) {
                ++slot;
                continue;
            }
            const size_t run_start = slot;
            while (slot < slots.size() && slots[slot] != 0) {
                ++slot;
            }
            OPENVINO_ASSERT(block_index < blocks.size(),
                            "Prompt reserves more image slots than there are image blocks");
            const size_t run_length = slot - run_start;
            const size_t block_area = blocks[block_index].area();
            OPENVINO_ASSERT(block_area % run_length == 0,
                            "Image block of ", block_area, " latent tokens does not fit ", run_length, " prompt slots");
            std::fill(repeats.begin() + run_start, repeats.begin() + slot, block_area / run_length);
            std::fill(slot_block.begin() + run_start, slot_block.begin() + slot, static_cast<int32_t>(block_index));
            ++block_index;
        }
        OPENVINO_ASSERT(block_index == blocks.size(),
                        "Prompt reserves fewer image slots than there are image blocks");

        // Each joint position reads either a text token or an image token of cat([text, image_blocks]).
        const size_t seq_len = std::accumulate(repeats.begin(), repeats.end(), size_t{0});
        std::vector<int32_t> token_block;
        token_block.reserve(seq_len);
        JointSequenceInputs inputs;
        inputs.gather_idx = ov::Tensor(ov::element::i64, {seq_len});
        int64_t* gather_data = inputs.gather_idx.data<int64_t>();

        size_t image_token_count = 0;
        for (size_t slot = 0; slot < slots.size(); ++slot) {
            for (size_t repeat = 0; repeat < repeats[slot]; ++repeat) {
                *gather_data++ = slots[slot] != 0 ? static_cast<int64_t>(text_length + image_token_count++)
                                                  : static_cast<int64_t>(slot);
                token_block.push_back(slot_block[slot]);
            }
        }

        std::tie(inputs.cos, inputs.sin) = build_rotary_embeddings(token_block, blocks);

        // Attention follows '(q_idx >= kv_idx) or same_image_block': text rows stay causal, while an image row
        // additionally attends to the rest of its own block.
        std::vector<size_t> block_start(blocks.size(), 0);
        for (size_t token = seq_len; token-- > 0;) {
            if (token_block[token] >= 0) {
                block_start[token_block[token]] = token;
            }
        }

        inputs.attn_mask = ov::Tensor(ov::element::f32, {batch_size, 1, seq_len, seq_len});
        float* attn_data = inputs.attn_mask.data<float>();
        for (size_t row = 0; row < seq_len; ++row) {
            float* row_data = attn_data + row * seq_len;
            std::fill_n(row_data, row + 1, 0.0f);
            std::fill_n(row_data + row + 1, seq_len - row - 1, -std::numeric_limits<float>::infinity());
            if (token_block[row] >= 0) {
                std::fill_n(row_data + block_start[token_block[row]], blocks[token_block[row]].area(), 0.0f);
            }
        }
        for (size_t batch = 1; batch < batch_size; ++batch) {
            std::copy_n(attn_data, seq_len * seq_len, attn_data + batch * seq_len * seq_len);
        }

        inputs.modulation_mask = ov::Tensor(ov::element::boolean, {seq_len});
        bool* modulation_data = inputs.modulation_mask.data<bool>();
        const int32_t target_block = static_cast<int32_t>(blocks.size() - 1);
        for (size_t token = 0; token < seq_len; ++token) {
            modulation_data[token] = token_block[token] == target_block;
        }

        return inputs;
    }

    // 3-axis (frame, height, width) rotary embedding. Text tokens advance a shared position on all three axes;
    // every image block freezes the frame axis at the position reached by the preceding text and lays its tokens
    // out on a height/width grid centred on zero. The half-width table is duplicated onto both halves.
    std::pair<ov::Tensor, ov::Tensor> build_rotary_embeddings(const std::vector<int32_t>& token_block,
                                                              const std::vector<LatentBlock>& blocks) const {
        const std::vector<size_t>& axes_dims_rope = m_transformer->get_config().axes_dims_rope;
        const size_t half_dim = std::accumulate(axes_dims_rope.begin(), axes_dims_rope.end(), size_t{0}) / 2;
        const size_t head_dim = half_dim * 2;
        const size_t seq_len = token_block.size();

        std::vector<std::array<int64_t, 3>> positions(seq_len);
        int64_t position = 0;
        for (size_t token = 0; token < seq_len;) {
            if (token_block[token] < 0) {
                positions[token] = {position, position, position};
                ++position;
                ++token;
                continue;
            }
            const LatentBlock& block = blocks[token_block[token]];
            const int64_t height_origin = -static_cast<int64_t>(block.height - block.height / 2);
            const int64_t width_origin = -static_cast<int64_t>(block.width - block.width / 2);
            for (size_t index = 0; index < block.area(); ++index, ++token) {
                positions[token] = {position,
                                    height_origin + static_cast<int64_t>(index / block.width),
                                    width_origin + static_cast<int64_t>(index % block.width)};
            }
            position += static_cast<int64_t>(std::max(block.height, block.width));
        }

        ov::Tensor cos(ov::element::f32, {1, seq_len, 1, head_dim});
        ov::Tensor sin(ov::element::f32, {1, seq_len, 1, head_dim});
        float* cos_data = cos.data<float>();
        float* sin_data = sin.data<float>();

        for (size_t token = 0; token < seq_len; ++token) {
            float* token_cos = cos_data + token * head_dim;
            float* token_sin = sin_data + token * head_dim;
            size_t offset = 0;
            for (size_t axis = 0; axis < axes_dims_rope.size(); ++axis) {
                const size_t axis_half = axes_dims_rope[axis] / 2;
                for (size_t i = 0; i < axis_half; ++i) {
                    const double inv_freq =
                        1.0 / std::pow(ROPE_THETA, 2.0 * static_cast<double>(i) / static_cast<double>(axes_dims_rope[axis]));
                    const double angle = static_cast<double>(positions[token][axis]) * inv_freq;
                    token_cos[offset + i] = token_cos[half_dim + offset + i] = static_cast<float>(std::cos(angle));
                    token_sin[offset + i] = token_sin[half_dim + offset + i] = static_cast<float>(std::sin(angle));
                }
                offset += axis_half;
            }
        }

        return {cos, sin};
    }

    void set_transformer_inputs(const ov::Tensor prompt_embeds, const JointSequenceInputs& inputs) {
        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds);
        m_transformer->set_hidden_states("cos", inputs.cos);
        m_transformer->set_hidden_states("sin", inputs.sin);
        m_transformer->set_hidden_states("gather_idx", inputs.gather_idx);
        m_transformer->set_hidden_states("attn_mask", inputs.attn_mask);
        m_transformer->set_hidden_states("modulation_mask", inputs.modulation_mask);
    }

    // The transformer returns the whole joint sequence; only the trailing image tokens are the noise prediction.
    ov::Tensor infer_transformer(const ov::Tensor& latents, const ov::Tensor& timestep, const size_t image_seq_len) {
        const auto infer_start = std::chrono::steady_clock::now();
        const ov::Tensor joint_output = m_transformer->infer(latents, timestep);
        m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(
            MicroSeconds(ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start)));

        const ov::Shape& shape = joint_output.get_shape();
        const ov::Coordinate start{0, shape[1] - image_seq_len, 0}, end{shape[0], shape[1], shape[2]};

        ov::Tensor noise_pred(joint_output.get_element_type(), {shape[0], image_seq_len, shape[2]});
        ov::Tensor(joint_output, start, end).copy_to(noise_pred);
        return noise_pred;
    }

    std::shared_ptr<Qwen3VLForConditionalGeneration> m_text_encoder;
    std::shared_ptr<AutoencoderKLQwenImage> m_vae;
    std::shared_ptr<QwenImage21Transformer2DModel> m_transformer;

    ov::Tensor m_positive_prompt_embeds, m_positive_image_pad_mask;
    ov::Tensor m_negative_prompt_embeds, m_negative_image_pad_mask;

    ov::Tensor m_condition_image;
    LatentBlock m_condition_block;

    ImageGenerationConfig m_custom_generation_config;
    ImageGenerationPerfMetrics m_perf_metrics;
};

}  // namespace genai
}  // namespace ov
