// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>

#include <openvino/op/convert.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/round.hpp>
#include <openvino/op/transpose.hpp>
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"

#include "image_generation/image_processor.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/schedulers/ischeduler.hpp"
#include "image_generation/threaded_callback.hpp"
#include "diffusion_caching/taylorseer_lite.hpp"
#include "lora/helper.hpp"
#include "logger.hpp"
#include "openvino/genai/video_generation/ltx_video_transformer_3d_model.hpp"
#include "generation_config_utils.hpp"

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
    std::nullopt,            // strength
    TaylorSeerCacheConfig{}  // taylorseer_config
};

// Some defaults aren't special values so it's not possible to distinguish
// whether user set them or not. Replace only special values.
void replace_defaults(VideoGenerationConfig& config) {
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

std::shared_ptr<IScheduler> cast_scheduler(std::shared_ptr<Scheduler>&& scheduler) {
    auto casted = std::dynamic_pointer_cast<IScheduler>(std::move(scheduler));
    OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
    return casted;
}

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

// Rescales the CFG noise prediction to fix overexposure when using zero terminal SNR
// noise_cfg and noise_pred_text each contain batch_size * elements_per_sample consecutive floats.
void rescale_noise_cfg(float* noise_cfg,
                       const float* noise_pred_text,
                       size_t batch_size,
                       size_t elements_per_sample,
                       float guidance_rescale) {
    if (elements_per_sample == 0) {
        return;
    }
    for (size_t b = 0; b < batch_size; ++b) {
        float* cfg_sample = noise_cfg + b * elements_per_sample;
        const float* text_sample = noise_pred_text + b * elements_per_sample;

        double text_mean = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            text_mean += text_sample[i];
        }
        text_mean /= static_cast<double>(elements_per_sample);

        double text_var = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            const double diff = text_sample[i] - text_mean;
            text_var += diff * diff;
        }
        const float std_text = static_cast<float>(std::sqrt(text_var / static_cast<double>(elements_per_sample)));

        double cfg_mean = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            cfg_mean += cfg_sample[i];
        }
        cfg_mean /= static_cast<double>(elements_per_sample);

        double cfg_var = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            const double diff = cfg_sample[i] - cfg_mean;
            cfg_var += diff * diff;
        }
        const float std_cfg = static_cast<float>(std::sqrt(cfg_var / static_cast<double>(elements_per_sample)));

        const float scale = std_cfg > 0.0f ? std_text / std_cfg : 1.0f;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            const float rescaled = cfg_sample[i] * scale;
            cfg_sample[i] = guidance_rescale * rescaled + (1.0f - guidance_rescale) * cfg_sample[i];
        }
    }
}

// Unpacked latents of shape [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p,
// p]. The patch dimensions are then permuted and collapsed into the channel dimension of shape: [B, F // p_t * H // p *
// W // p, C * p_t * p * p] (a 3 dimensional tensor). dim=0 is the batch size, dim=1 is the effective video sequence
// length, dim=2 is the effective number of input features
ov::Tensor pack_latents(ov::Tensor& latents, size_t patch_size, size_t patch_size_t) {
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape.at(0), num_channels = latents_shape.at(1), num_frames = latents_shape.at(2),
           height = latents_shape.at(3), width = latents_shape.at(4);
    size_t post_patch_num_frames = num_frames / patch_size_t;
    size_t post_patch_height = height / patch_size;
    size_t post_patch_width = width / patch_size;
    latents.set_shape({batch_size,
                       num_channels,
                       post_patch_num_frames,
                       patch_size_t,
                       post_patch_height,
                       patch_size,
                       post_patch_width,
                       patch_size});
    std::array<int64_t, 8> order = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<ov::Tensor> outputs{ov::Tensor(ov::element::f32, {})};
    ov::op::v1::Transpose{}.evaluate(outputs,
                                     {latents, ov::Tensor(ov::element::i64, ov::Shape{order.size()}, order.data())});
    ov::Shape permuted_shape = outputs.at(0).get_shape();
    outputs.at(0).set_shape({permuted_shape.at(0),
                             permuted_shape.at(1) * permuted_shape.at(2) * permuted_shape.at(3),
                             permuted_shape.at(4) * permuted_shape.at(5) * permuted_shape.at(6)});
    return outputs.at(0);
}

// Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
// are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of what happens
// in the `_pack_latents` method.
ov::Tensor unpack_latents(const ov::Tensor& latents,
                          size_t num_frames,
                          size_t height,
                          size_t width,
                          size_t patch_size = 1,
                          size_t patch_size_t = 1) {
    const ov::Shape in_shape = latents.get_shape();
    OPENVINO_ASSERT(in_shape.size() == 3, "unpack_latents expects [B, S, D] input shape");
    const size_t batch_size = in_shape.at(0), sequence_length = in_shape.at(1), feature_dimensions = in_shape.at(2);

    const size_t patch_volume = patch_size_t * patch_size * patch_size;
    OPENVINO_ASSERT(feature_dimensions % patch_volume == 0, "D must be divisible by patch_size_t * patch_size * patch_size");
    const size_t num_channels = feature_dimensions / patch_volume;

    ov::Tensor reshaped{latents.get_element_type(), latents.get_shape()};
    latents.copy_to(reshaped);
    reshaped.set_shape({batch_size, num_frames, height, width, num_channels, patch_size_t, patch_size, patch_size});

    // permute(0, 4, 1, 5, 2, 6, 3, 7) -> [B, C, F//patch_size_t, patch_size_t, H//patch_size, patch_size, W//patch_size, patch_size]
    const std::array<int64_t, 8> order = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<ov::Tensor> outputs{ov::Tensor(reshaped.get_element_type(), {})};
    ov::op::v1::Transpose{}.evaluate(
        outputs,
        {reshaped, ov::Tensor(ov::element::i64, ov::Shape{order.size()}, const_cast<int64_t*>(order.data()))}
    );

    // (F//patch_size_t, patch_size_t) -> F, (H//patch_size, patch_size) -> H, (W//patch_size, patch_size) -> W
    const ov::Shape perm = outputs[0].get_shape(); // [B, C, F//patch_size_t, patch_size_t, H//patch_size, patch_size, W//patch_size, patch_size]
    OPENVINO_ASSERT(perm.size() == 8, "Unexpected rank after transpose");

    const size_t F = perm[2] * perm[3]; // (F//patch_size_t) * patch_size_t
    const size_t H = perm[4] * perm[5]; // (H//patch_size) * patch_size
    const size_t W = perm[6] * perm[7]; // (W//patch_size) * patch_size

    outputs[0].set_shape({perm[0], perm[1], F, H, W}); // [B, C, F, H, W]
    return outputs[0];
}

inline void reshape_to_1C111(ov::Tensor& t, size_t C) {
    size_t elems = 1;
    for (auto d : t.get_shape())
        elems *= d;

    OPENVINO_ASSERT(elems == C, "latents_mean/std must contain exactly C elements (got ", elems, ", expected ", C, ")");

    t.set_shape({1, C, 1, 1, 1});
}

inline ov::Tensor make_scalar(const ov::element::Type& et, float v) {
    ov::Tensor s(et, {});
    if (et == ov::element::f32) {
        *s.data<float>() = v;
    } else if (et == ov::element::f16) {
        *s.data<ov::float16>() = static_cast<ov::float16>(v);
    } else if (et == ov::element::bf16) {
        *s.data<ov::bfloat16>() = static_cast<ov::bfloat16>(v);
    } else {
        OPENVINO_ASSERT(false, "Unsupported element type for scalar scaling_factor");
    }
    return s;
}

// Denormalize latents across channel dim: [B, C, F, H, W]
// latents = latents * latents_std / scaling_factor + latents_mean
ov::Tensor denormalize_latents(const ov::Tensor& latents,
                               ov::Tensor latents_mean,
                               ov::Tensor latents_std,
                               float scaling_factor = 1.0f) {
    const ov::Shape latents_shape = latents.get_shape();
    OPENVINO_ASSERT(latents_shape.size() == 5, "denormalize_latents expects [B, C, F, H, W]");
    const size_t num_channels = latents_shape[1];

    // .view(1, -1, 1, 1, 1)
    reshape_to_1C111(latents_mean, num_channels);
    reshape_to_1C111(latents_std, num_channels);

    const auto latents_type = latents.get_element_type();
    ov::Tensor scale = make_scalar(latents_type, scaling_factor);

    // latents * latents_std
    std::vector<ov::Tensor> tmp{ov::Tensor(latents_type, {})};
    ov::op::v1::Multiply{}.evaluate(tmp, {latents, latents_std});  // NUMPY broadcast

    // (...) / scaling_factor
    std::vector<ov::Tensor> tmp2{ov::Tensor(latents_type, {})};
    ov::op::v1::Divide{}.evaluate(tmp2, {tmp[0], scale});

    // (...) + latents_mean
    std::vector<ov::Tensor> result{ov::Tensor(latents_type, {})};
    ov::op::v1::Add{}.evaluate(result, {tmp2[0], latents_mean});

    return result[0];  // [B, C, F, H, W]
}

inline ov::Tensor tensor_from_vector(const std::vector<float>& data) {
    ov::Tensor t{ov::element::f32, ov::Shape{data.size()}};
    if (!data.empty()) {
        std::memcpy(t.data<float>(), data.data(), data.size() * sizeof(float));
    }
    return t;
}

}  // anonymous namespace


namespace ov::genai {

enum class VideoPipelineType {
    TEXT_2_VIDEO = 0,
    IMAGE_2_VIDEO = 1,
};

class LTXPipeline {
    using Ms = std::chrono::duration<float, std::ratio<1, 1000>>;

    std::shared_ptr<IScheduler> m_scheduler;
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder;
    std::shared_ptr<LTXVideoTransformer3DModel> m_transformer;
    std::shared_ptr<AutoencoderKLLTXVideo> m_vae;
    VideoGenerationPerfMetrics m_perf_metrics;
    Ms m_load_time;

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

    ov::Tensor prepare_latents(const ov::genai::VideoGenerationConfig& generation_config,
                               size_t num_channels_latents,
                               size_t transformer_spatial_patch_size,
                               size_t transformer_temporal_patch_size) {
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
        ov::Tensor latents = generation_config.generator->randn_tensor(shape);
        return pack_latents(latents, transformer_spatial_patch_size, transformer_temporal_patch_size);
    }

    ov::Tensor prepare_latents(const ov::Tensor& image_latent_packed,
                               const VideoGenerationConfig& config,
                               size_t num_channels_latents,
                               size_t ps,
                               size_t ps_t) {
        OPENVINO_ASSERT(m_latent_num_frames > 0 && m_latent_height > 0 && m_latent_width > 0,
                        "Latent sizes must be > 0 (got num_frames=",
                        m_latent_num_frames,
                        ", height=",
                        m_latent_height,
                        ", width=",
                        m_latent_width,
                        ").");

        ov::Shape shape{config.num_videos_per_prompt,
                        num_channels_latents,
                        m_latent_num_frames,
                        m_latent_height,
                        m_latent_width};
        ov::Tensor noise = config.generator->randn_tensor(shape);
        ov::Tensor latents = pack_latents(noise, ps, ps_t);

        const size_t tokens_per_frame = m_latent_height * m_latent_width;
        const size_t D = latents.get_shape()[2];
        for (size_t b = 0; b < config.num_videos_per_prompt; ++b) {
            float* dst       = latents.data<float>()             + b * m_latent_num_frames * tokens_per_frame * D;
            const float* src = image_latent_packed.data<float>() + b * tokens_per_frame * D;
            std::memcpy(dst, src, tokens_per_frame * D * sizeof(float));
        }
        return latents;
    }

    ov::Tensor preprocess_and_encode_image(const ov::Tensor& image, const VideoGenerationConfig& config) {
        OPENVINO_ASSERT(m_pipeline_type == VideoPipelineType::IMAGE_2_VIDEO,
                        "Image2VideoPipeline requires a VAE encoder. "
                        "Ensure 'vae_encoder' exists in the model directory.");

        ov::Tensor img = image;
        if (img.get_shape().size() == 3) {
            auto s = img.get_shape();
            img.set_shape({1, s[0], s[1], s[2]});
        }

        ov::Tensor resized = m_image_resizer->execute(img, config.height, config.width);
        ov::Tensor processed = m_image_processor->execute(resized);

        // Copy to an owned tensor (with temporal dim added) before passing to the encoder.
        // get_output_tensor() aliases the ImageProcessor's internal buffer; passing that aliased
        // tensor directly to a second set_input_tensor() / infer() call causes a hang in OpenVINO.
        const auto& proc_shape = processed.get_shape();
        ov::Tensor encoder_input(ov::element::f32, {proc_shape[0], proc_shape[1], 1, proc_shape[2], proc_shape[3]});
        std::memcpy(encoder_input.data<float>(), processed.data<const float>(),
                    proc_shape[0] * proc_shape[1] * proc_shape[2] * proc_shape[3] * sizeof(float));

        ov::Tensor latent = m_vae->encode(encoder_input, config.generator);

        const size_t ps   = m_transformer->get_config().patch_size;
        const size_t ps_t = m_transformer->get_config().patch_size_t;
        ov::Tensor packed = pack_latents(latent, ps, ps_t);

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

        ov::Tensor decoded = unpack_latents(latent,
                                            m_latent_num_frames,
                                            m_latent_height,
                                            m_latent_width,
                                            m_transformer->get_config().patch_size,
                                            m_transformer->get_config().patch_size_t);

        decoded = denormalize_latents(decoded,
                                      tensor_from_vector(m_vae->get_config().latents_mean_data),
                                      tensor_from_vector(m_vae->get_config().latents_std_data),
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
    VideoGenerationConfig m_generation_config;

    VideoGenerationPerfMetrics get_performance_metrics() const {
        return m_perf_metrics;
    }

    LTXPipeline(const std::filesystem::path& root_dir,
                VideoPipelineType pipeline_type = VideoPipelineType::TEXT_2_VIDEO,
                std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()) {
        m_models_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";

        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);

        OPENVINO_ASSERT("LTXPipeline" == data["_class_name"].get<std::string>());

        m_scheduler = cast_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

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

    LTXPipeline(const std::filesystem::path& models_dir,
                const std::string& device,
                const ov::AnyMap& properties,
                VideoPipelineType pipeline_type = VideoPipelineType::TEXT_2_VIDEO,
                std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now())
        : m_scheduler{cast_scheduler(Scheduler::from_config(models_dir / "scheduler/scheduler_config.json"))},
          m_t5_text_encoder{std::make_shared<T5EncoderModel>(models_dir / "text_encoder", device, properties)},
          m_transformer{std::make_shared<LTXVideoTransformer3DModel>(models_dir / "transformer", device, properties)},
          m_generation_config{LTX_VIDEO_DEFAULT_CONFIG},
          m_pipeline_type{pipeline_type} {
        if (pipeline_type == VideoPipelineType::IMAGE_2_VIDEO) {
            m_vae = std::make_shared<AutoencoderKLLTXVideo>(
                models_dir / "vae_encoder", models_dir / "vae_decoder", device, properties);
            m_image_resizer = std::make_shared<ImageResizer>(
                "CPU", ov::element::u8, "NHWC",
                ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW);
            m_image_processor = std::make_shared<ImageProcessor>("CPU", true);
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
        const std::filesystem::path model_index_path = models_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);
        OPENVINO_ASSERT("LTXPipeline" == nlohmann::json::parse(file)["_class_name"].get<std::string>());
        m_load_time = Ms{std::chrono::steady_clock::now() - start_time};
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
                                   const ov::AnyMap& properties = {}) {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();

        VideoGenerationConfig merged_generation_config = m_generation_config;
        utils::update_generation_config(merged_generation_config, properties);
        replace_defaults(merged_generation_config);
        const float requested_guidance_scale = merged_generation_config.guidance_scale;

        const float strength = merged_generation_config.strength.value_or(1.0f);
        OPENVINO_ASSERT(strength >= 0.0f && strength <= 1.0f,
                        "'strength' must be in [0.0, 1.0], got ", strength);

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

        ov::Tensor latent = prepare_latents(image_latent_packed,
                                            merged_generation_config,
                                            num_channels_latents,
                                            transformer_spatial_patch_size,
                                            transformer_temporal_patch_size);

        if (strength == 0.0f) {
            latent = postprocess_latents(latent);
            OPENVINO_ASSERT(!m_vae->get_config().timestep_conditioning,
                            "Parameter 'timestep_conditioning' is not currently supported by AutoencoderKLLTX. Please, contact OpenVINO GenAI developers.");
            ov::Tensor video = m_vae->decode(latent);
            m_perf_metrics.generate_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
            return VideoGenerationResult{video, m_perf_metrics};
        }

        const size_t video_sequence_length = m_latent_num_frames * m_latent_height * m_latent_width;
        m_scheduler->set_timesteps(video_sequence_length,
                                   merged_generation_config.num_inference_steps,
                                   strength);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        ov::Tensor rope_interpolation_scale(ov::element::f32, {3});
        const float frame_rate =
            merged_generation_config.frame_rate.value_or(LTX_VIDEO_DEFAULT_CONFIG.frame_rate.value());
        rope_interpolation_scale.data<float>()[0] =
            static_cast<float>(temporal_compression_ratio) / frame_rate;
        rope_interpolation_scale.data<float>()[1] = spatial_compression_ratio;
        rope_interpolation_scale.data<float>()[2] = spatial_compression_ratio;
        m_transformer->set_hidden_states("rope_interpolation_scale", rope_interpolation_scale);

        // Per-token conditioning mask (matches diffusers): 1.0 at frame-0 tokens, 0.0 elsewhere.
        // Frame-0 occupies the first `tokens_per_frame` positions in the packed [B, S, D] layout
        // (pack_latents permutes (F, H, W) outermost-to-innermost). With patch_size>1 the frame-0
        // span is post-patch H*W; here patch_size=patch_size_t=1 for LTX-Video so it's lH*lW.
        const size_t tokens_per_frame =
            (m_latent_height / transformer_spatial_patch_size) *
            (m_latent_width  / transformer_spatial_patch_size);
        std::vector<float> conditioning_mask(video_sequence_length, 0.0f);
        std::fill_n(conditioning_mask.begin(), tokens_per_frame, 1.0f);

        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);

        TaylorSeerState ts_state(merged_generation_config.taylorseer_config, timesteps.size());

        ov::Tensor noisy_residual_tensor(ov::element::f32, {});
        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();

            // Upstream LTX-Video `add_noise_to_image_conditioning_latents`
            // (pipeline_ltx_video.py): before each transformer call, replace frame-0 tokens
            // with `init_latent + scale * randn * (t/1000)^2`. The model sees a slightly noised
            // anchor each step, which breaks the static-attractor collapse that pure clean-pin
            // produces in our setup. Default 0.15 (upstream default); override via
            // I2V_NOISE_SCALE env var (0 disables).
            {
                float image_cond_noise_scale = 0.15f;
                if (const char* env = std::getenv("I2V_NOISE_SCALE")) {
                    image_cond_noise_scale = std::atof(env);
                }
                if (image_cond_noise_scale > 0.0f) {
                    const float t_norm = timesteps[inference_step] / 1000.0f;
                    const float noise_coeff = image_cond_noise_scale * t_norm * t_norm;
                    const size_t D_pre = latent.get_shape()[2];
                    const ov::Shape f0_shape{merged_generation_config.num_videos_per_prompt,
                                              tokens_per_frame, D_pre};
                    ov::Tensor n = merged_generation_config.generator->randn_tensor(f0_shape);
                    const float* nd = n.data<const float>();
                    for (size_t b = 0; b < merged_generation_config.num_videos_per_prompt; ++b) {
                        float* dst       = latent.data<float>()             + b * video_sequence_length * D_pre;
                        const float* src = image_latent_packed.data<float>() + b * tokens_per_frame * D_pre;
                        const float* nn  = nd                                + b * tokens_per_frame * D_pre;
                        for (size_t i = 0; i < tokens_per_frame * D_pre; ++i) {
                            dst[i] = src[i] + noise_coeff * nn[i];
                        }
                    }
                    if (inference_step == 0) {
                        std::cerr << "[I2V] image_cond_noise_scale=" << image_cond_noise_scale
                                  << " (override via I2V_NOISE_SCALE)\n" << std::flush;
                    }
                }
            }

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
            const size_t request_input_batch = m_transformer->get_request_input_batch();
            if (request_input_batch > latent_cfg.get_shape()[0]) {
                OPENVINO_ASSERT(request_input_batch % latent_cfg.get_shape()[0] == 0,
                                "Transformer input batch must be divisible by latent batch");
                latent_cfg = numpy_utils::repeat(latent_cfg, request_input_batch / latent_cfg.get_shape()[0]);
            }

            // Per-token timestep: timestep = t * (1 - conditioning_mask)
            //   frame-0 tokens (mask=1)  -> t=0 (clean image-conditioning frame)
            //   other tokens   (mask=0)  -> t=current
            // Matches diffusers LTXImageToVideoPipeline.__call__:
            //   timestep = t.expand(B).unsqueeze(-1) * (1 - conditioning_mask)
            const float t = timesteps[inference_step];
            const size_t B_ts = latent_cfg.get_shape()[0];
            ov::Tensor timestep(ov::element::f32, {B_ts, video_sequence_length});
            float* timestep_data = timestep.data<float>();
            for (size_t b = 0; b < B_ts; ++b) {
                float* row = timestep_data + b * video_sequence_length;
                for (size_t i = 0; i < video_sequence_length; ++i) {
                    row[i] = t * (1.0f - conditioning_mask[i]);
                }
            }

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
                rescale_noise_cfg(noisy_residual_tensor.data<float>(),
                                  noise_pred_tensor.data<const float>() + noisy_residual_tensor.get_size(),
                                  noise_pred_shape[0],
                                  noisy_residual_tensor.get_size() / noise_pred_shape[0],
                                  *merged_generation_config.guidance_rescale);
            }

            // Diffusers I2V step ordering (pipeline_ltx_image2video.py:864-889):
            //   noise_pred = noise_pred[:, :, 1:]          # drop frame-0 from prediction
            //   noise_latents = latents[:, :, 1:]          # drop frame-0 from current latent
            //   pred_latents = scheduler.step(noise_pred, t, noise_latents)
            //   latents = cat([latents[:, :, :1], pred_latents], dim=2)
            // Run scheduler.step on frames 1+ only; preserve frame-0 verbatim from prior step
            // (which itself was pinned to clean image_latent_packed).
            const size_t B_l = latent.get_shape()[0];
            const size_t D = latent.get_shape()[2];
            const size_t tail_tokens = video_sequence_length - tokens_per_frame;
            ov::Tensor tail_latent(ov::element::f32, {B_l, tail_tokens, D});
            ov::Tensor tail_pred(ov::element::f32, {noisy_residual_tensor.get_shape()[0], tail_tokens, D});
            for (size_t b = 0; b < B_l; ++b) {
                std::memcpy(tail_latent.data<float>() + b * tail_tokens * D,
                            latent.data<float>() + b * video_sequence_length * D + tokens_per_frame * D,
                            tail_tokens * D * sizeof(float));
            }
            for (size_t b = 0; b < noisy_residual_tensor.get_shape()[0]; ++b) {
                std::memcpy(tail_pred.data<float>() + b * tail_tokens * D,
                            noisy_residual_tensor.data<float>() + b * video_sequence_length * D + tokens_per_frame * D,
                            tail_tokens * D * sizeof(float));
            }

            // Per-step noise_pred stats (full prediction, before slicing) — flag model collapse.
            if (inference_step % 10 == 0 || inference_step == timesteps.size() - 1) {
                const float* p = noisy_residual_tensor.data<const float>();
                const size_t per_b = video_sequence_length * D;
                auto stats = [&](size_t tok_idx) {
                    const float* row = p + tok_idx * D;
                    float amax = 0.0f, asum = 0.0f;
                    for (size_t i = 0; i < D; ++i) { float v = std::abs(row[i]); amax = std::max(amax, v); asum += v; }
                    return std::make_pair(amax, asum / D);
                };
                auto [f0_max, f0_mean] = stats(0);
                auto [f1_max, f1_mean] = stats(tokens_per_frame);
                auto [fL_max, fL_mean] = stats(video_sequence_length - 1);
                std::cerr << "[I2V step " << inference_step << " t=" << timesteps[inference_step]
                          << "] pred f0(amax=" << f0_max << ",amean=" << f0_mean << ")"
                          << " f1(amax=" << f1_max << ",amean=" << f1_mean << ")"
                          << " fLast(amax=" << fL_max << ",amean=" << fL_mean << ")\n" << std::flush;
                (void)per_b;
            }

            auto scheduler_step_result =
                m_scheduler->step(tail_pred, tail_latent, inference_step, merged_generation_config.generator);
            ov::Tensor tail_out = scheduler_step_result["latent"];
            OPENVINO_ASSERT(tail_out.get_shape() == ov::Shape({B_l, tail_tokens, D}),
                            "I2V scheduler.step returned unexpected shape");

            // Reassemble: frame-0 = clean image_latent_packed, frames 1+ = scheduler output.
            for (size_t b = 0; b < B_l; ++b) {
                float* dst       = latent.data<float>() + b * video_sequence_length * D;
                const float* src0 = image_latent_packed.data<float>() +
                                    (b % merged_generation_config.num_videos_per_prompt) * tokens_per_frame * D;
                std::memcpy(dst, src0, tokens_per_frame * D * sizeof(float));
                std::memcpy(dst + tokens_per_frame * D,
                            tail_out.data<float>() + b * tail_tokens * D,
                            tail_tokens * D * sizeof(float));
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

    VideoGenerationResult generate(const std::string& positive_prompt, const ov::AnyMap& properties = {}) {
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
        m_scheduler->set_timesteps(video_sequence_length,
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

        // Timestep tensor. The exported optimum-intel ltx-i2v-support transformer has rank-2
        // timestep [B, S]; the default (main) export has rank-1 [B]. We allocate based on the
        // compiled model's expected rank so this path works for both exports.
        ov::Shape timestep_shape;
        const auto& timestep_partial = m_transformer->get_timestep_partial_shape();
        if (timestep_partial.size() == 2) {
            timestep_shape = {batch_size_multiplier * merged_generation_config.num_videos_per_prompt,
                              video_sequence_length};
        } else {
            timestep_shape = {batch_size_multiplier * merged_generation_config.num_videos_per_prompt};
        }
        ov::Tensor timestep(ov::element::f32, timestep_shape);
        float* timestep_data = timestep.data<float>();

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

            std::fill_n(timestep_data, timestep.get_size(), timesteps[inference_step]);

            ov::Tensor noise_pred_tensor;
            // Use TaylorSeer if enabled and caching is appropriate
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
                rescale_noise_cfg(noisy_residual_tensor.data<float>(),
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

    VideoGenerationResult decode(const ov::Tensor& latent) {
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
                 float guidance_scale) {
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

    void save_load_time(std::chrono::steady_clock::time_point start_time) {
        m_load_time += Ms{std::chrono::steady_clock::now() - start_time};
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) {
        update_adapters_from_properties(properties, m_generation_config.adapters);
        m_t5_text_encoder->compile(text_encode_device, properties);
        m_vae->compile(vae_device, properties);
        m_transformer->compile(denoise_device, properties);
        m_text_encode_device = text_encode_device;
        m_denoise_device = denoise_device;
        m_vae_device = vae_device;
        m_compile_properties = properties;
        m_is_compiled = true;
        m_compiled_batch_size_multiplier = m_reshape_batch_size_multiplier;
    }

    void compile(const std::string& device, const ov::AnyMap& properties) {
        compile(device, device, device, properties);
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
