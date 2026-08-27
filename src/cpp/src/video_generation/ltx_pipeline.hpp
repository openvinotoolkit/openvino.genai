// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>

#include "image_generation/image_processor.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/schedulers/ischeduler.hpp"
#include "image_generation/threaded_callback.hpp"
#include "diffusion_caching/taylorseer_lite.hpp"
#include "lora/helper.hpp"
#include "logger.hpp"
#include "openvino/genai/video_generation/ltx_video_transformer_3d_model.hpp"
#include "generation_config_utils.hpp"
#include "video_generation/video_generation_utils.hpp"
#include "video_generation/video_pipeline.hpp"

#include "utils.hpp"

using namespace ov::genai;

namespace {

const VideoGenerationConfig LTX_VIDEO_DEFAULT_CONFIG = VideoGenerationConfig{
    std::nullopt,            // negative_prompt
    1,                       // num_videos_per_prompt
    nullptr,                 // generator
    7.5f,                    // guidance_scale
    512,                     // height
    704,                     // width
    50,                      // num_inference_steps
    128,                     // max_sequence_length
    0.0,                     // guidance_rescale
    161,                     // num_frames
    25.0f,                   // frame_rate
    std::nullopt             // taylorseer_config
};

void check_inputs(const VideoGenerationConfig& generation_config, size_t vae_scale_factor) {
    utils::validate_generation_config(generation_config);
    OPENVINO_ASSERT(generation_config.height > 0, "Height must be positive");
    OPENVINO_ASSERT(generation_config.height % 32 == 0,
                    "Height have to be divisible by 32 but got ",
                    generation_config.height);
    OPENVINO_ASSERT(generation_config.width > 0, "Width must be positive");
    OPENVINO_ASSERT(generation_config.width % 32 == 0,
                    "Width have to be divisible by 32 but got ",
                    generation_config.width);

    OPENVINO_ASSERT(generation_config.max_sequence_length <= 512,
                    "T5's 'max_sequence_length' must be less or equal to 512");
    OPENVINO_ASSERT((generation_config.height % vae_scale_factor == 0 || generation_config.height < 0) &&
                        (generation_config.width % vae_scale_factor == 0 || generation_config.width < 0),
                    "Both 'width' and 'height' must be divisible by ",
                    vae_scale_factor);
}

}  // anonymous namespace

namespace ov::genai {

class LTXPipeline : public VideoPipeline {
    std::shared_ptr<IScheduler> m_scheduler;
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder;
    std::shared_ptr<LTXVideoTransformer3DModel> m_transformer;
    std::shared_ptr<AutoencoderKLLTXVideo> m_vae;

    size_t m_latent_num_frames = 0;
    size_t m_latent_height = 0;
    size_t m_latent_width = 0;
    // Batch size multiplier from the last reshape() call (0 = not set, 1 = no CFG, 2 = CFG enabled)
    size_t m_reshape_batch_size_multiplier = 0;
    // Batch size multiplier used when model was compiled (0 = not compiled, 1 = no CFG, 2 = CFG enabled)
    size_t m_compiled_batch_size_multiplier = 0;
    bool m_is_compiled = false;
    std::filesystem::path m_models_dir;
    std::string m_text_encode_device;
    std::string m_denoise_device;
    std::string m_vae_device;
    ov::AnyMap m_compile_properties;
    VideoPipelineType m_pipeline_type = VideoPipelineType::TEXT_2_VIDEO;
    std::shared_ptr<ImageResizer> m_image_resizer = nullptr;
    std::shared_ptr<ImageProcessor> m_image_processor = nullptr;

    // Builds the initial packed latent noise. For image-to-video, pass the encoded
    // conditioning image to anchor the first frame; leave empty for text-to-video.
    ov::Tensor prepare_latents(const ov::genai::VideoGenerationConfig& generation_config,
                               size_t num_channels_latents,
                               size_t transformer_spatial_patch_size,
                               size_t transformer_temporal_patch_size,
                               const ov::Tensor& image_latent_packed = ov::Tensor()) {
        OPENVINO_ASSERT(generation_config.generator, "Generator must not be null");
        OPENVINO_ASSERT(m_latent_num_frames > 0 && m_latent_height > 0 && m_latent_width > 0,
                        "Latent sizes must be > 0 (got num_frames=",
                        m_latent_num_frames,
                        ", height=",
                        m_latent_height,
                        ", width=",
                        m_latent_width,
                        ").");

        ov::Shape shape{generation_config.num_videos_per_prompt,
                        num_channels_latents,
                        m_latent_num_frames,
                        m_latent_height,
                        m_latent_width};
        ov::Tensor noise = generation_config.generator->randn_tensor(shape);
        ov::Tensor latents =
            video_generation_utils::pack_latents(noise, transformer_spatial_patch_size, transformer_temporal_patch_size);

        // Image-to-video: pin the first-frame tokens to the encoded conditioning image.
        if (image_latent_packed) {
            const size_t tokens_per_frame =
                (m_latent_height / transformer_spatial_patch_size) * (m_latent_width / transformer_spatial_patch_size);
            const size_t S = latents.get_shape()[1];
            const size_t D = latents.get_shape()[2];
            for (size_t b = 0; b < generation_config.num_videos_per_prompt; ++b) {
                float* dst       = latents.data<float>()             + b * S * D;
                const float* src = image_latent_packed.data<float>() + b * tokens_per_frame * D;
                std::memcpy(dst, src, tokens_per_frame * D * sizeof(float));
            }
        }
        return latents;
    }

    ov::Tensor preprocess_and_encode_image(const ov::Tensor& image, const VideoGenerationConfig& config) {
        OPENVINO_ASSERT(m_pipeline_type == VideoPipelineType::IMAGE_2_VIDEO,
                        "Image2VideoPipeline requires a VAE encoder. "
                        "Ensure 'vae_encoder' exists in the model directory.");

        // ov::Tensor copies share the underlying memory, so set_shape() here would promote the
        // caller's rank-3 tensor in place. Wrap the same memory in a rank-4 view instead.
        ov::Tensor img = image;
        if (image.get_shape().size() == 3) {
            const auto s = image.get_shape();
            img = ov::Tensor(image.get_element_type(), ov::Shape{1, s[0], s[1], s[2]}, image.data());
        }

        const auto& img_shape = img.get_shape();
        OPENVINO_ASSERT(img_shape.size() == 4,
                        "Conditioning image must have shape [H, W, 3] or [1, H, W, 3] (NHWC), got rank ",
                        img_shape.size());
        OPENVINO_ASSERT(img.get_element_type() == ov::element::u8,
                        "Conditioning image must have element type u8 (uint8), got ",
                        img.get_element_type());
        OPENVINO_ASSERT(img_shape[3] == 3,
                        "Conditioning image must have 3 channels in the last dimension (NHWC), got ",
                        img_shape[3]);

        ov::Tensor resized = (img_shape[1] == static_cast<size_t>(config.height) &&
                              img_shape[2] == static_cast<size_t>(config.width))
                                 ? img
                                 : m_image_resizer->execute(img, config.height, config.width);
        ov::Tensor processed = m_image_processor->execute(resized);

        OPENVINO_ASSERT(processed.get_element_type() == ov::element::f32,
                        "ImageProcessor must return f32, got ", processed.get_element_type());
        OPENVINO_ASSERT(processed.get_shape().size() == 4,
                        "ImageProcessor must return rank-4 [N,C,H,W], got rank ", processed.get_shape().size());
        const auto& proc_shape = processed.get_shape();
        ov::Tensor encoder_input(ov::element::f32, {proc_shape[0], proc_shape[1], 1, proc_shape[2], proc_shape[3]});
        std::memcpy(encoder_input.data<float>(), processed.data<const float>(),
                    proc_shape[0] * proc_shape[1] * proc_shape[2] * proc_shape[3] * sizeof(float));

        ov::Tensor latent = m_vae->encode(encoder_input, config.generator);

        const size_t ps   = m_transformer->get_config().patch_size;
        const size_t ps_t = m_transformer->get_config().patch_size_t;
        OPENVINO_ASSERT(ps_t == 1,
                        "Image2VideoPipeline requires patch_size_t=1; the conditioning latent has a single frame "
                        "and cannot be temporally packed with patch_size_t=", ps_t);
        ov::Tensor packed = video_generation_utils::pack_latents(latent, ps, ps_t);

        if (config.num_videos_per_prompt > 1)
            packed = numpy_utils::repeat(packed, config.num_videos_per_prompt);

        return packed;
    }

    ov::Tensor postprocess_latents(const ov::Tensor& latent) {
        OPENVINO_ASSERT(m_latent_num_frames > 0 && m_latent_height > 0 && m_latent_width > 0,
                        "Latent sizes must be > 0 (got num_frames=",
                        m_latent_num_frames,
                        ", height=",
                        m_latent_height,
                        ", width=",
                        m_latent_width,
                        ").");

        ov::Tensor decoded = video_generation_utils::unpack_latents(latent,
                                                                    m_latent_num_frames,
                                                                    m_latent_height,
                                                                    m_latent_width,
                                                                    m_transformer->get_config().patch_size,
                                                                    m_transformer->get_config().patch_size_t);

        decoded = video_generation_utils::denormalize_latents(
            decoded,
            video_generation_utils::tensor_from_vector(m_vae->get_config().latents_mean_data),
            video_generation_utils::tensor_from_vector(m_vae->get_config().latents_std_data),
            m_vae->get_config().scaling_factor);

        return decoded;
    }

    void compute_hidden_states(const std::string& positive_prompt,
                               const std::string& negative_prompt,
                               const VideoGenerationConfig& generation_config,
                               bool do_classifier_free_guidance) {
        OPENVINO_ASSERT(m_latent_num_frames > 0 && m_latent_height > 0 && m_latent_width > 0,
                        "Latent sizes must be > 0 (got num_frames=",
                        m_latent_num_frames,
                        ", height=",
                        m_latent_height,
                        ", width=",
                        m_latent_width,
                        ").");

        auto infer_start = std::chrono::steady_clock::now();
        // torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        // genai m_t5_text_encoder->infer retuns the same tensor [negative_prompt_embeds, prompt_embeds]
        infer_start = std::chrono::steady_clock::now();
        ov::Tensor prompt_embeds =
            m_t5_text_encoder->infer(positive_prompt,
                                     negative_prompt,
                                     do_classifier_free_guidance,
                                     generation_config.max_sequence_length,
                                     {ov::genai::pad_to_max_length(true),
                                      ov::genai::max_length(generation_config.max_sequence_length),
                                      ov::genai::add_special_tokens(true)});

        auto infer_end = std::chrono::steady_clock::now();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = Ms{infer_end - infer_start}.count();

        ov::Tensor prompt_attention_mask = m_t5_text_encoder->get_prompt_attention_mask();

        prompt_embeds = numpy_utils::repeat(prompt_embeds, generation_config.num_videos_per_prompt);
        prompt_attention_mask = numpy_utils::repeat(prompt_attention_mask, generation_config.num_videos_per_prompt);

        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds);
        m_transformer->set_hidden_states("encoder_attention_mask", prompt_attention_mask);

        auto make_scalar_tensor = [](size_t value) {
            ov::Tensor scalar(ov::element::i64, {});
            scalar.data<int64_t>()[0] = value;
            return scalar;
        };
        m_transformer->set_hidden_states("num_frames", make_scalar_tensor(m_latent_num_frames));
        m_transformer->set_hidden_states("height", make_scalar_tensor(m_latent_height));
        m_transformer->set_hidden_states("width", make_scalar_tensor(m_latent_width));
    }

public:
    LTXPipeline(VideoPipelineType pipeline_type,
                const std::filesystem::path& root_dir,
                std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()) {
        m_models_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";

        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);

        m_scheduler = video_generation_utils::cast_scheduler(
            Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string t5_text_encoder = data["text_encoder"][1].get<std::string>();
        if (t5_text_encoder == "T5EncoderModel") {
            m_t5_text_encoder = std::make_shared<T5EncoderModel>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", t5_text_encoder, "' text encoder type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKLLTXVideo") {
            if (pipeline_type == VideoPipelineType::IMAGE_2_VIDEO) {
                m_vae = std::make_shared<AutoencoderKLLTXVideo>(root_dir / "vae_encoder", root_dir / "vae_decoder");
            } else {
                m_vae = std::make_shared<AutoencoderKLLTXVideo>(root_dir / "vae_decoder");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "LTXVideoTransformer3DModel") {
            m_transformer = std::make_shared<LTXVideoTransformer3DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        m_generation_config = LTX_VIDEO_DEFAULT_CONFIG;
        m_pipeline_type = pipeline_type;

        if (pipeline_type == VideoPipelineType::IMAGE_2_VIDEO) {
            m_image_resizer = std::make_shared<ImageResizer>(
                "CPU", ov::element::u8, "NHWC",
                ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW);
            m_image_processor = std::make_shared<ImageProcessor>("CPU", true);
        }

        m_load_time = Ms{std::chrono::steady_clock::now() - start_time};
    }

    LTXPipeline(VideoPipelineType pipeline_type,
                const std::filesystem::path& models_dir,
                const std::string& device,
                const ov::AnyMap& properties,
                std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now())
        : m_scheduler{video_generation_utils::cast_scheduler(
              Scheduler::from_config(models_dir / "scheduler/scheduler_config.json"))},
          m_t5_text_encoder{std::make_shared<T5EncoderModel>(models_dir / "text_encoder", device, properties)},
          m_transformer{std::make_shared<LTXVideoTransformer3DModel>(models_dir / "transformer", device, properties)},
          m_pipeline_type{pipeline_type} {
        m_generation_config = LTX_VIDEO_DEFAULT_CONFIG;
        if (pipeline_type == VideoPipelineType::IMAGE_2_VIDEO) {
            m_vae = std::make_shared<AutoencoderKLLTXVideo>(
                models_dir / "vae_encoder", models_dir / "vae_decoder", device, properties);
            m_image_resizer = std::make_shared<ImageResizer>(
                device, ov::element::u8, "NHWC",
                ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW);
            m_image_processor = std::make_shared<ImageProcessor>(device, true);
        } else {
            m_vae = std::make_shared<AutoencoderKLLTXVideo>(models_dir / "vae_decoder", device, properties);
        }
        m_models_dir = models_dir;
        m_text_encode_device = device;
        m_denoise_device = device;
        m_vae_device = device;
        m_compile_properties = properties;
        m_is_compiled = true;
        update_adapters_from_properties(properties, m_generation_config.adapters);
        m_load_time = Ms{std::chrono::steady_clock::now() - start_time};
    }

    std::shared_ptr<VideoPipeline> clone() override {
        OPENVINO_ASSERT(m_is_compiled, "Cannot clone an uncompiled LTXPipeline");
        auto cloned = std::make_shared<LTXPipeline>(*this);
        cloned->m_generation_config.generator.reset();
        cloned->m_scheduler = video_generation_utils::cast_scheduler(
            Scheduler::from_config(m_models_dir / "scheduler/scheduler_config.json"));
        cloned->m_t5_text_encoder = m_t5_text_encoder->clone();
        cloned->m_transformer = std::make_shared<LTXVideoTransformer3DModel>(m_transformer->clone());
        cloned->m_vae = std::make_shared<AutoencoderKLLTXVideo>(m_vae->clone());
        if (m_pipeline_type == VideoPipelineType::IMAGE_2_VIDEO) {
            cloned->m_image_resizer = std::make_shared<ImageResizer>(
                m_vae_device, ov::element::u8, "NHWC",
                ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW);
            cloned->m_image_processor = std::make_shared<ImageProcessor>(m_vae_device, true);
        }
        return cloned;
    }

    bool do_classifier_free_guidance(float guidance_scale) const {
        return guidance_scale > 1.0;
    }

    void rebuild_models() {
        m_t5_text_encoder = std::make_shared<T5EncoderModel>(m_models_dir / "text_encoder");
        m_transformer = std::make_shared<LTXVideoTransformer3DModel>(m_models_dir / "transformer");
        if (m_pipeline_type == VideoPipelineType::IMAGE_2_VIDEO)
            m_vae = std::make_shared<AutoencoderKLLTXVideo>(m_models_dir / "vae_encoder", m_models_dir / "vae_decoder");
        else
            m_vae = std::make_shared<AutoencoderKLLTXVideo>(m_models_dir / "vae_decoder");
    }

    void reshape_models(const VideoGenerationConfig& generation_config, size_t batch_size_multiplier) {
        m_reshape_batch_size_multiplier = batch_size_multiplier;
        m_t5_text_encoder->reshape(batch_size_multiplier, generation_config.max_sequence_length);
        m_transformer->reshape(generation_config.num_videos_per_prompt * batch_size_multiplier,
                               generation_config.num_frames,
                               generation_config.height,
                               generation_config.width,
                               generation_config.max_sequence_length);
        m_vae->reshape(generation_config.num_videos_per_prompt,
                       generation_config.num_frames,
                       generation_config.height,
                       generation_config.width);
    }

    void reconfigure_for_guidance_scale(const VideoGenerationConfig& generation_config, size_t batch_size_multiplier) {
        rebuild_models();
        reshape_models(generation_config, batch_size_multiplier);
        if (m_is_compiled) {
            compile(m_text_encode_device, m_denoise_device, m_vae_device, m_compile_properties);
        }
    }

    VideoGenerationResult generate(const ov::Tensor& image,
                                   const std::string& positive_prompt,
                                   const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();

        VideoGenerationConfig merged_generation_config = m_generation_config;
        utils::update_generation_config(merged_generation_config, properties);
        replace_defaults(merged_generation_config);
        const float requested_guidance_scale = merged_generation_config.guidance_scale;

        size_t requested_batch_size_multiplier =
            do_classifier_free_guidance(merged_generation_config.guidance_scale) ? 2 : 1;
        if (m_is_compiled) {
            const size_t expected_batch_size = m_transformer->get_expected_batch_size();
            if (expected_batch_size > 0) {
                OPENVINO_ASSERT(expected_batch_size % merged_generation_config.num_videos_per_prompt == 0,
                                "Compiled batch size must be divisible by num_videos_per_prompt");
                requested_batch_size_multiplier =
                    expected_batch_size / merged_generation_config.num_videos_per_prompt;
            } else if (m_compiled_batch_size_multiplier > 0) {
                requested_batch_size_multiplier = m_compiled_batch_size_multiplier;
            }
            OPENVINO_ASSERT(!(requested_batch_size_multiplier > 1 && merged_generation_config.guidance_scale <= 1.0f),
                            "guidance_scale <= 1 requested, but the compiled model expects CFG (batch size multiplier = ",
                            requested_batch_size_multiplier, "). "
                            "Either set guidance_scale > 1, or reshape/compile the model with guidance_scale <= 1.");
        }
        size_t batch_size_multiplier = std::max({requested_batch_size_multiplier,
                                                  m_reshape_batch_size_multiplier,
                                                  m_compiled_batch_size_multiplier});

        if (!m_is_compiled) {
            if (m_reshape_batch_size_multiplier == 0) {
                m_reshape_batch_size_multiplier = batch_size_multiplier;
            } else if (m_reshape_batch_size_multiplier < batch_size_multiplier) {
                reconfigure_for_guidance_scale(merged_generation_config, batch_size_multiplier);
            }
        }

        const bool use_classifier_free_guidance = batch_size_multiplier > 1;
        if (m_is_compiled && requested_guidance_scale > 1.0f && !use_classifier_free_guidance) {
            GENAI_WARN("guidance_scale > 1 requested, but the compiled model batch size does not allow CFG. "
                       "Run reshape/compile with guidance_scale > 1 to enable guidance.");
        }

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();
        check_inputs(merged_generation_config, vae_scale_factor);

        m_transformer->set_adapters(merged_generation_config.adapters);

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        const size_t num_channels_latents = transformer_config.in_channels;
        const size_t spatial_compression_ratio =
            m_vae->get_config().patch_size * std::pow(2,
                                                      std::accumulate(m_vae->get_config().spatio_temporal_scaling.begin(),
                                                                  m_vae->get_config().spatio_temporal_scaling.end(),
                                                                  0));
        const size_t temporal_compression_ratio =
            m_vae->get_config().patch_size_t * std::pow(2,
                                                        std::accumulate(m_vae->get_config().spatio_temporal_scaling.begin(),
                                                                    m_vae->get_config().spatio_temporal_scaling.end(),
                                                                    0));
        const size_t transformer_spatial_patch_size  = transformer_config.patch_size;
        const size_t transformer_temporal_patch_size = transformer_config.patch_size_t;

        m_latent_num_frames = (merged_generation_config.num_frames - 1) / temporal_compression_ratio + 1;
        m_latent_height = merged_generation_config.height / spatial_compression_ratio;
        m_latent_width  = merged_generation_config.width  / spatial_compression_ratio;

        compute_hidden_states(positive_prompt,
                              merged_generation_config.negative_prompt.value_or(""),
                              merged_generation_config,
                              use_classifier_free_guidance);

        ov::Tensor image_latent_packed = preprocess_and_encode_image(image, merged_generation_config);

        ov::Tensor latent = prepare_latents(merged_generation_config,
                                            num_channels_latents,
                                            transformer_spatial_patch_size,
                                            transformer_temporal_patch_size,
                                            image_latent_packed);

        const size_t video_sequence_length = latent.get_shape().at(1);
        // mu must come from calculate_shift(), matching t2v and HF's reference,
        // not the time_shift_type-branching set_timesteps() overload (which
        // picks a different, unrelated formula for "exponential" configs).
        const double mu = m_scheduler->calculate_shift(video_sequence_length);
        m_scheduler->set_timesteps_with_mu(mu,
                                           merged_generation_config.num_inference_steps,
                                           1.0f);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        ov::Tensor rope_interpolation_scale(ov::element::f32, {3});
        const float frame_rate =
            merged_generation_config.frame_rate.value_or(LTX_VIDEO_DEFAULT_CONFIG.frame_rate.value());
        rope_interpolation_scale.data<float>()[0] =
            static_cast<float>(temporal_compression_ratio) / frame_rate;
        rope_interpolation_scale.data<float>()[1] = spatial_compression_ratio;
        rope_interpolation_scale.data<float>()[2] = spatial_compression_ratio;
        m_transformer->set_hidden_states("rope_interpolation_scale", rope_interpolation_scale);

        // Frame-0 tokens are first in the packed [B, S, D] layout (pack_latents is F-outermost).
        const size_t tokens_per_frame =
            (m_latent_height / transformer_spatial_patch_size) *
            (m_latent_width  / transformer_spatial_patch_size);
        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);

        TaylorSeerState ts_state(merged_generation_config.taylorseer_config, timesteps.size());

        const size_t B_ts = latent_shape_cfg[0];
        // Frame-0 conditioning needs a per-token timestep. Legacy exports have a rank-1 timestep.
        OPENVINO_ASSERT(m_transformer->get_timestep_rank() == 2,
                        "Image-to-video requires a rank-2 [B, S] timestep input, but this model has a "
                        "legacy rank-1 timestep. Please re-export the model.");
        ov::Tensor timestep(ov::element::f32, {B_ts, video_sequence_length});
        float* timestep_data = timestep.data<float>();
        for (size_t b = 0; b < B_ts; ++b) {
            std::fill_n(timestep_data + b * video_sequence_length, tokens_per_frame, 0.0f);
        }

        ov::Tensor noisy_residual_tensor(ov::element::f32, {});
        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();
            if (batch_size_multiplier > 1) {
                numpy_utils::batch_copy(latent, latent_cfg, 0, 0, merged_generation_config.num_videos_per_prompt);
                numpy_utils::batch_copy(latent,
                                        latent_cfg,
                                        0,
                                        merged_generation_config.num_videos_per_prompt,
                                        merged_generation_config.num_videos_per_prompt);
            } else {
                latent_cfg = latent;
            }
            // batch_size_multiplier already incorporates m_transformer->get_expected_batch_size()
            // (see the override above), so latent_cfg's batch can never fall short of it here.
            // Tripwire: latent_cfg must stay in sync with the B_ts-sized timestep tensor.
            OPENVINO_ASSERT(latent_cfg.get_shape()[0] == B_ts,
                            "latent batch (", latent_cfg.get_shape()[0],
                            ") must match timestep batch (", B_ts, ")");

            const float t = timesteps[inference_step];
            for (size_t b = 0; b < B_ts; ++b) {
                std::fill_n(timestep_data + b * video_sequence_length + tokens_per_frame,
                            video_sequence_length - tokens_per_frame, t);
            }

            // Use TaylorSeer if enabled and caching is appropriate
            ov::Tensor noise_pred_tensor;
            if (ts_state.is_active() && !ts_state.should_compute(inference_step)) {
                noise_pred_tensor = ts_state.predict(inference_step);
            } else {
                auto infer_start = std::chrono::steady_clock::now();
                noise_pred_tensor = m_transformer->infer(latent_cfg, timestep);
                auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
                m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));
                if (ts_state.is_active()) {
                    ts_state.update(inference_step, noise_pred_tensor);
                }
            }

            ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
            noise_pred_shape[0] /= batch_size_multiplier;

            if (batch_size_multiplier > 1) {
                noisy_residual_tensor.set_shape(noise_pred_shape);
                float* noisy_residual = noisy_residual_tensor.data<float>();
                const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
                const float* noise_pred_text = noise_pred_uncond + noisy_residual_tensor.get_size();

                for (size_t i = 0; i < noisy_residual_tensor.get_size(); ++i) {
                    noisy_residual[i] = noise_pred_uncond[i] + merged_generation_config.guidance_scale *
                                                                   (noise_pred_text[i] - noise_pred_uncond[i]);
                }
            } else {
                noisy_residual_tensor = noise_pred_tensor;
            }

            if (batch_size_multiplier > 1 && *merged_generation_config.guidance_rescale > 0.0f) {
                OPENVINO_ASSERT(noise_pred_shape[0] > 0,
                                "Expected positive batch dimension in noise_pred_shape[0] before rescaling noise.");
                video_generation_utils::rescale_noise_cfg(noisy_residual_tensor.data<float>(),
                                  noise_pred_tensor.data<const float>() + noisy_residual_tensor.get_size(),
                                  noise_pred_shape[0],
                                  noisy_residual_tensor.get_size() / noise_pred_shape[0],
                                  *merged_generation_config.guidance_rescale);
            }

            auto scheduler_step_result =
                m_scheduler->step(noisy_residual_tensor, latent, inference_step, merged_generation_config.generator);
            latent = scheduler_step_result["latent"];

            // Frame-0 tokens are pinned to init_latents; discard the scheduler's update to them.
            {
                const size_t D = latent.get_shape()[2];
                const size_t B_l = latent.get_shape()[0];
                for (size_t b = 0; b < B_l; ++b) {
                    float* dst       = latent.data<float>() + b * video_sequence_length * D;
                    const float* src = image_latent_packed.data<float>() +
                                       (b % merged_generation_config.num_videos_per_prompt) * tokens_per_frame * D;
                    std::memcpy(dst, src, tokens_per_frame * D * sizeof(float));
                }
            }

            if (callback_ptr && callback_ptr->has_callback() && callback_ptr->write(inference_step, timesteps.size(), latent) == CallbackStatus::STOP) {
                callback_ptr->end();
                auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
                m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
                auto video = ov::Tensor(ov::element::u8, {});
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start)
                        .count();
                return {video, m_perf_metrics};
            }

            auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
        }

        if (callback_ptr != nullptr) {
            callback_ptr->end();
        }

        latent = postprocess_latents(latent);

        OPENVINO_ASSERT(!m_vae->get_config().timestep_conditioning,
                            "Parameter 'timestep_conditioning' is not currently supported by AutoencoderKLLTX. Please, contact OpenVINO GenAI developers.");

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = m_vae->decode(latent);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();

        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();

        return VideoGenerationResult{video, m_perf_metrics};
    }

    VideoGenerationResult generate(const std::string& positive_prompt, const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();

        VideoGenerationConfig merged_generation_config = m_generation_config;
        utils::update_generation_config(merged_generation_config, properties);
        replace_defaults(merged_generation_config);
        const float requested_guidance_scale = merged_generation_config.guidance_scale;

        size_t requested_batch_size_multiplier =
            do_classifier_free_guidance(merged_generation_config.guidance_scale) ? 2 : 1;
        if (m_is_compiled) {
            const size_t expected_batch_size = m_transformer->get_expected_batch_size();
            if (expected_batch_size > 0) {
                OPENVINO_ASSERT(expected_batch_size % merged_generation_config.num_videos_per_prompt == 0,
                                "Compiled batch size must be divisible by num_videos_per_prompt");
                requested_batch_size_multiplier =
                    expected_batch_size / merged_generation_config.num_videos_per_prompt;
            } else if (m_compiled_batch_size_multiplier > 0) {
                requested_batch_size_multiplier = m_compiled_batch_size_multiplier;
            }
            OPENVINO_ASSERT(!(requested_batch_size_multiplier > 1 && merged_generation_config.guidance_scale <= 1.0f),
                            "guidance_scale <= 1 requested, but the compiled model expects CFG (batch size multiplier = ",
                            requested_batch_size_multiplier, "). "
                            "Either set guidance_scale > 1, or reshape/compile the model with guidance_scale <= 1.");
        }
        // Use maximum of all multipliers to ensure model can handle requested batch size
        size_t batch_size_multiplier = std::max({requested_batch_size_multiplier,
                                                  m_reshape_batch_size_multiplier,
                                                  m_compiled_batch_size_multiplier});

        // Before compilation: track and upgrade reshape multiplier if CFG is needed
        if (!m_is_compiled) {
            if (m_reshape_batch_size_multiplier == 0) {
                m_reshape_batch_size_multiplier = batch_size_multiplier;
            } else if (m_reshape_batch_size_multiplier < batch_size_multiplier) {
                reconfigure_for_guidance_scale(merged_generation_config, batch_size_multiplier);
            }
        }

        const bool use_classifier_free_guidance = batch_size_multiplier > 1;
        if (m_is_compiled && requested_guidance_scale > 1.0f && !use_classifier_free_guidance) {
            GENAI_WARN("guidance_scale > 1 requested, but the compiled model batch size does not allow CFG. "
                       "Run reshape/compile with guidance_scale > 1 to enable guidance.");
        }

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();
        check_inputs(merged_generation_config, vae_scale_factor);

        m_transformer->set_adapters(merged_generation_config.adapters);

        // use callback if defined
        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        size_t num_channels_latents = transformer_config.in_channels;
        size_t spatial_compression_ratio =
            m_vae->get_config().patch_size * std::pow(2,
                                                      std::accumulate(m_vae->get_config().spatio_temporal_scaling.begin(),
                                                                  m_vae->get_config().spatio_temporal_scaling.end(),
                                                                  0));
        size_t temporal_compression_ratio =
            m_vae->get_config().patch_size_t * std::pow(2,
                                                        std::accumulate(m_vae->get_config().spatio_temporal_scaling.begin(),
                                                                    m_vae->get_config().spatio_temporal_scaling.end(),
                                                                    0));
        size_t transformer_spatial_patch_size = transformer_config.patch_size;
        size_t transformer_temporal_patch_size = transformer_config.patch_size_t;

        m_latent_num_frames = (merged_generation_config.num_frames - 1) / temporal_compression_ratio + 1;
        m_latent_height = merged_generation_config.height / spatial_compression_ratio;
        m_latent_width = merged_generation_config.width / spatial_compression_ratio;

        compute_hidden_states(positive_prompt,
                              merged_generation_config.negative_prompt.value_or(""),
                              merged_generation_config,
                              use_classifier_free_guidance);

        ov::Tensor latent = prepare_latents(merged_generation_config,
                                            num_channels_latents,
                                            transformer_spatial_patch_size,
                                            transformer_temporal_patch_size);

        // Prepare timesteps
        size_t video_sequence_length = m_latent_num_frames * m_latent_height * m_latent_width;
        const double mu = m_scheduler->calculate_shift(video_sequence_length);
        m_scheduler->set_timesteps_with_mu(mu,
                                           merged_generation_config.num_inference_steps,
                                           1.0f);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        // Prepare micro-conditions
        // TODO: move to compute_hidden_states
        ov::Tensor rope_interpolation_scale(ov::element::f32, {3});
        const float frame_rate =
            merged_generation_config.frame_rate.value_or(LTX_VIDEO_DEFAULT_CONFIG.frame_rate.value());
        rope_interpolation_scale.data<float>()[0] =
            static_cast<float>(temporal_compression_ratio) / frame_rate;
        rope_interpolation_scale.data<float>()[1] = spatial_compression_ratio;
        rope_interpolation_scale.data<float>()[2] = spatial_compression_ratio;
        m_transformer->set_hidden_states("rope_interpolation_scale", rope_interpolation_scale);

        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);

        // Initialize TaylorSeer if configured
        TaylorSeerState ts_state(merged_generation_config.taylorseer_config, timesteps.size());

        // Denoising loop
        ov::Tensor noisy_residual_tensor(ov::element::f32, {});
        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();
            // concat the same latent twice along a batch dimension in case of CFG
            if (batch_size_multiplier > 1) {
                numpy_utils::batch_copy(latent, latent_cfg, 0, 0, merged_generation_config.num_videos_per_prompt);
                numpy_utils::batch_copy(latent,
                                        latent_cfg,
                                        0,
                                        merged_generation_config.num_videos_per_prompt,
                                        merged_generation_config.num_videos_per_prompt);
            } else {
                // just assign to save memory copy
                latent_cfg = latent;
            }
            // Match compiled model's expected batch size by repeating latent if needed
            // (e.g., when model was compiled with CFG but current config doesn't require it)
            const size_t request_input_batch = m_transformer->get_request_input_batch();
            if (request_input_batch > latent_cfg.get_shape()[0]) {
                OPENVINO_ASSERT(request_input_batch % latent_cfg.get_shape()[0] == 0,
                                "Transformer input batch must be divisible by latent batch");
                latent_cfg = numpy_utils::repeat(latent_cfg, request_input_batch / latent_cfg.get_shape()[0]);
            }

            ov::Tensor noise_pred_tensor;
            // Use TaylorSeer if enabled and caching is appropriate
            if (ts_state.is_active() && !ts_state.should_compute(inference_step)) {
                noise_pred_tensor = ts_state.predict(inference_step);
            } else {
                auto infer_start = std::chrono::steady_clock::now();
                noise_pred_tensor = m_transformer->infer(latent_cfg, timesteps[inference_step]);
                auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
                m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));
                if (ts_state.is_active()) {
                    ts_state.update(inference_step, noise_pred_tensor);
                }
            }

            ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
            noise_pred_shape[0] /= batch_size_multiplier;

            if (batch_size_multiplier > 1) {
                noisy_residual_tensor.set_shape(noise_pred_shape);

                // perform guidance
                float* noisy_residual = noisy_residual_tensor.data<float>();
                const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
                const float* noise_pred_text = noise_pred_uncond + noisy_residual_tensor.get_size();

                for (size_t i = 0; i < noisy_residual_tensor.get_size(); ++i) {
                    noisy_residual[i] = noise_pred_uncond[i] + merged_generation_config.guidance_scale *
                                                                   (noise_pred_text[i] - noise_pred_uncond[i]);
                }
            } else {
                noisy_residual_tensor = noise_pred_tensor;
            }

            if (batch_size_multiplier > 1 && *merged_generation_config.guidance_rescale > 0.0f) {
                OPENVINO_ASSERT(noise_pred_shape[0] > 0,
                                "Expected positive batch dimension in noise_pred_shape[0] before rescaling noise.");
                video_generation_utils::rescale_noise_cfg(noisy_residual_tensor.data<float>(),
                                  noise_pred_tensor.data<const float>() + noisy_residual_tensor.get_size(),
                                  noise_pred_shape[0],
                                  noisy_residual_tensor.get_size() / noise_pred_shape[0],
                                  *merged_generation_config.guidance_rescale);
            }

            auto scheduler_step_result =
                m_scheduler->step(noisy_residual_tensor, latent, inference_step, merged_generation_config.generator);
            latent = scheduler_step_result["latent"];

            if (callback_ptr && callback_ptr->has_callback() && callback_ptr->write(inference_step, timesteps.size(), latent) == CallbackStatus::STOP) {
                callback_ptr->end();
                auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
                m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));

                auto video = ov::Tensor(ov::element::u8, {});
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start)
                        .count();
                return {video, m_perf_metrics};
            }

            auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
        }

        if (callback_ptr != nullptr) {
            callback_ptr->end();
        }

        latent = postprocess_latents(latent);

        // TODO: support timestep_conditioning for AutoencoderKLLTX
        OPENVINO_ASSERT(!m_vae->get_config().timestep_conditioning,
                            "Parameter 'timestep_conditioning' is not currently supported by AutoencoderKLLTX. Please, contact OpenVINO GenAI developers.");

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = m_vae->decode(latent);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();

        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();

        return VideoGenerationResult{video, m_perf_metrics};
    }

    VideoGenerationResult decode(const ov::Tensor& latent) override {
        ov::Tensor postprocessed = postprocess_latents(latent);

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = m_vae->decode(postprocessed);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();

        return VideoGenerationResult{video, m_perf_metrics};
    }

    void reshape(int64_t num_videos_per_prompt,
                 int64_t num_frames,
                 int64_t height,
                 int64_t width,
                 float guidance_scale) override {
        check_video_size(height, width);

        VideoGenerationConfig reshaped_config = m_generation_config;
        reshaped_config.num_videos_per_prompt = num_videos_per_prompt;
        reshaped_config.num_frames = num_frames;
        reshaped_config.height = height;
        reshaped_config.width = width;
        reshaped_config.guidance_scale = guidance_scale;
        const size_t batch_size_multiplier =
            do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Transformer accepts 2x batch in case of CFG
        reshape_models(reshaped_config, batch_size_multiplier);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);
        m_t5_text_encoder->compile(text_encode_device, properties);
        m_vae->compile(vae_device, properties);
        if (m_pipeline_type == VideoPipelineType::IMAGE_2_VIDEO) {
            m_image_resizer = std::make_shared<ImageResizer>(
                vae_device, ov::element::u8, "NHWC",
                ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW);
            m_image_processor = std::make_shared<ImageProcessor>(vae_device, true);
        }

        m_transformer->compile(denoise_device, properties);
        m_text_encode_device = text_encode_device;
        m_denoise_device = denoise_device;
        m_vae_device = vae_device;
        m_compile_properties = properties;
        m_is_compiled = true;
        m_compiled_batch_size_multiplier = m_reshape_batch_size_multiplier;
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        compile(device, device, device, properties);
    }

protected:
    void replace_defaults(VideoGenerationConfig& config) const override {
        // Some defaults aren't special values so it's not possible to distinguish
        // whether user set them or not. Replace only special values.
        if (-1 == config.height) {
            config.height = LTX_VIDEO_DEFAULT_CONFIG.height;
        }
        if (-1 == config.width) {
            config.width = LTX_VIDEO_DEFAULT_CONFIG.width;
        }
        if (-1 == config.num_inference_steps) {
            config.num_inference_steps = LTX_VIDEO_DEFAULT_CONFIG.num_inference_steps;
        }
        if (-1 == config.max_sequence_length) {
            config.max_sequence_length = LTX_VIDEO_DEFAULT_CONFIG.max_sequence_length;
        }
        if (!config.guidance_rescale.has_value()) {
            config.guidance_rescale = LTX_VIDEO_DEFAULT_CONFIG.guidance_rescale;
        }
        if (0 == config.num_frames) {
            config.num_frames = LTX_VIDEO_DEFAULT_CONFIG.num_frames;
        }
        if (!config.frame_rate.has_value()) {
            config.frame_rate = LTX_VIDEO_DEFAULT_CONFIG.frame_rate;
        }
    }

private:
    void check_video_size(const int height, const int width) const {
        OPENVINO_ASSERT(m_transformer != nullptr);
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) && (width % vae_scale_factor == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ",
                        vae_scale_factor);

        OPENVINO_ASSERT(height > 0, "Height must be positive");
        OPENVINO_ASSERT(height % 32 == 0, "Height have to be divisible by 32 but got ", height);
        OPENVINO_ASSERT(width > 0, "Width must be positive");
        OPENVINO_ASSERT(width % 32 == 0, "Width have to be divisible by 32 but got ", width);
    }

};

}  // namespace ov::genai
