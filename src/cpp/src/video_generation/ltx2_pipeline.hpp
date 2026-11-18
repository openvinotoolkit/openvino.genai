// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "image_generation/numpy_utils.hpp"
#include "image_generation/schedulers/flow_match_euler_discrete.hpp"
#include "image_generation/schedulers/ischeduler.hpp"
#include "image_generation/threaded_callback.hpp"
#include "logger.hpp"
#include "lora/helper.hpp"
#include "generation_config_utils.hpp"
#include "openvino/genai/image_generation/gemma3_text_encoder.hpp"
#include "openvino/genai/video_generation/autoencoder_kl_ltx2_audio.hpp"
#include "openvino/genai/video_generation/autoencoder_kl_ltx2_video.hpp"
#include "openvino/genai/video_generation/ltx2_duration_head.hpp"
#include "openvino/genai/video_generation/ltx2_text_connectors.hpp"
#include "openvino/genai/video_generation/ltx2_video_transformer_3d_model.hpp"
#include "openvino/genai/video_generation/ltx2_vocoder.hpp"
#include "openvino/genai/video_generation/video_condition.hpp"
#include "video_generation/ltx2_conditioning.hpp"
#include "video_generation/ltx2_prompt_enhancer.hpp"
#include "video_generation/video_generation_utils.hpp"
#include "video_generation/video_pipeline.hpp"

#include "utils.hpp"

using namespace ov::genai;

namespace {

const VideoGenerationConfig LTX2_DEFAULT_CONFIG = VideoGenerationConfig{
    std::nullopt,            // negative_prompt
    1,                       // num_videos_per_prompt
    nullptr,                 // generator
    4.0f,                    // guidance_scale
    512,                     // height
    768,                     // width
    40,                      // num_inference_steps
    1024,                    // max_sequence_length
    0.0f,                    // guidance_rescale
    121,                     // num_frames
    24.0f,                   // frame_rate
    std::nullopt             // taylorseer_config
};

// LTX-2.3 defaults, mirroring diffusers 0.40's 'LTX2Pipeline.__call__'
const VideoGenerationConfig LTX2_3_DEFAULT_CONFIG = VideoGenerationConfig{
    std::nullopt,            // negative_prompt
    1,                       // num_videos_per_prompt
    nullptr,                 // generator
    3.0f,                    // guidance_scale
    512,                     // height
    768,                     // width
    30,                      // num_inference_steps
    1024,                    // max_sequence_length
    0.7f,                    // guidance_rescale
    121,                     // num_frames
    24.0f,                   // frame_rate
    std::nullopt,            // taylorseer_config
    std::nullopt,            // adapters
    7.0f,                    // audio_guidance_scale
    1.0f,                    // stg_scale
    1.0f,                    // audio_stg_scale
    3.0f,                    // modality_scale
    3.0f,                    // audio_modality_scale
    0.7f,                    // audio_guidance_rescale
    std::vector<int64_t>{28} // spatio_temporal_guidance_blocks
};

// LTX-2.5 ships its distilled transformer, which runs a fixed 8-sigma schedule with every guidance term off
// (diffusers' DISTILLED_SIGMA_VALUES). num_frames 0 lets the duration head pick the length.
const VideoGenerationConfig LTX2_5_DEFAULT_CONFIG = VideoGenerationConfig{
    std::nullopt,            // negative_prompt
    1,                       // num_videos_per_prompt
    nullptr,                 // generator
    1.0f,                    // guidance_scale
    512,                     // height
    768,                     // width
    8,                       // num_inference_steps
    1024,                    // max_sequence_length
    0.0f,                    // guidance_rescale
    0,                       // num_frames
    24.0f,                   // frame_rate
    std::nullopt,            // taylorseer_config
    std::nullopt,            // adapters
    1.0f,                    // audio_guidance_scale
    0.0f,                    // stg_scale
    0.0f,                    // audio_stg_scale
    1.0f,                    // modality_scale
    1.0f,                    // audio_modality_scale
    0.0f,                    // audio_guidance_rescale
    std::vector<int64_t>{28}, // spatio_temporal_guidance_blocks
    std::vector<float>{1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f} // sigmas
};

// LTX-2.5 with its full (non-distilled) transformer: LTX-2.3's guidance defaults, and the duration head
const VideoGenerationConfig LTX2_5_FULL_DEFAULT_CONFIG = [] {
    VideoGenerationConfig config = LTX2_3_DEFAULT_CONFIG;
    config.num_frames = 0;
    return config;
}();

// Repeats each batch entry of a [halves, ...] tensor num_videos times: [neg, pos] -> [neg x n, pos x n],
// matching the [uncond videos, cond videos] latent layout
ov::Tensor repeat_per_video(const ov::Tensor& input, size_t num_videos_per_prompt) {
    if (num_videos_per_prompt == 1) {
        return input;
    }
    ov::Shape repeated_shape = input.get_shape();
    const size_t halves = repeated_shape[0];
    repeated_shape[0] *= num_videos_per_prompt;
    ov::Tensor repeated(input.get_element_type(), repeated_shape);
    for (size_t h = 0; h < halves; ++h) {
        for (size_t v = 0; v < num_videos_per_prompt; ++v) {
            numpy_utils::batch_copy(input, repeated, h, h * num_videos_per_prompt + v);
        }
    }
    return repeated;
}

// Half of a CFG-batched [uncond, cond] tensor: [2N, ...] -> [N, ...]
ov::Tensor cfg_half(const ov::Tensor& tensor, size_t num_videos_per_prompt, size_t half_index) {
    ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape[0] == 2 * num_videos_per_prompt,
                    "Expected a CFG-batched tensor of batch ", 2 * num_videos_per_prompt, ", got ", shape[0]);
    shape[0] = num_videos_per_prompt;
    ov::Tensor half(tensor.get_element_type(), shape);
    numpy_utils::batch_copy(tensor, half, half_index * num_videos_per_prompt, 0, num_videos_per_prompt);
    return half;
}

// [B, C, L, M] -> [B, L, C * M]
ov::Tensor pack_audio_latents(const ov::Tensor& latents) {
    const ov::Shape shape = latents.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, "pack_audio_latents expects [B, C, L, M]");
    const size_t B = shape[0], C = shape[1], L = shape[2], M = shape[3];

    ov::Tensor packed(latents.get_element_type(), {B, L, C * M});
    const float* src = latents.data<const float>();
    float* dst = packed.data<float>();
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t l = 0; l < L; ++l) {
                std::memcpy(dst + ((b * L + l) * C + c) * M, src + ((b * C + c) * L + l) * M, M * sizeof(float));
            }
        }
    }
    return packed;
}

// [B, L, C * M] -> [B, C, L, M]
ov::Tensor unpack_audio_latents(const ov::Tensor& latents, size_t num_channels, size_t mel_bins) {
    const ov::Shape shape = latents.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[2] == num_channels * mel_bins,
                    "unpack_audio_latents expects [B, L, C * M]");
    const size_t B = shape[0], L = shape[1];

    ov::Tensor unpacked(latents.get_element_type(), {B, num_channels, L, mel_bins});
    const float* src = latents.data<const float>();
    float* dst = unpacked.data<float>();
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t c = 0; c < num_channels; ++c) {
                std::memcpy(dst + ((b * num_channels + c) * L + l) * mel_bins,
                            src + ((b * L + l) * num_channels + c) * mel_bins,
                            mel_bins * sizeof(float));
            }
        }
    }
    return unpacked;
}

}  // anonymous namespace

namespace ov::genai {

class LTX2Pipeline : public VideoPipeline {
    VideoPipelineType m_pipeline_type = VideoPipelineType::TEXT_2_VIDEO;
    std::shared_ptr<IScheduler> m_video_scheduler;
    std::shared_ptr<IScheduler> m_audio_scheduler;
    FlowMatchEulerDiscreteScheduler::Config m_scheduler_config;
    std::shared_ptr<Gemma3TextEncoder> m_text_encoder;
    std::shared_ptr<LTX2TextConnectors> m_connectors;
    std::shared_ptr<LTX2VideoTransformer3DModel> m_transformer;
    std::shared_ptr<AutoencoderKLLTX2Video> m_vae;
    std::shared_ptr<AutoencoderKLLTX2Audio> m_audio_vae;
    std::shared_ptr<LTX2Vocoder> m_vocoder;
    // LTX-2.5 onwards: picks num_frames when the caller leaves it 0
    std::shared_ptr<LTX2DurationHead> m_duration_head;
    // LTX-2.5 onwards: rewrites prompts on request. Created on first use, on the text encoding device; its
    // tokenizer is loaded with the pipeline.
    std::shared_ptr<Tokenizer> m_prompt_enhancer_tokenizer;
    std::shared_ptr<LTX2PromptEnhancer> m_prompt_enhancer;
    std::string m_text_encode_device = "CPU";
    std::string m_deferred_text_encode_device;
    std::string m_deferred_denoise_device;
    std::string m_deferred_vae_device;
    ov::AnyMap m_deferred_properties;

    size_t m_latent_num_frames = 0;
    size_t m_latent_height = 0;
    size_t m_latent_width = 0;
    std::filesystem::path m_models_dir;

    // Points at the defaults of the loaded model version; the single source of truth for 'replace_defaults'
    const VideoGenerationConfig* m_default_config = &LTX2_DEFAULT_CONFIG;

    void check_inputs(const VideoGenerationConfig& generation_config) const {
        video_generation_utils::check_video_size(generation_config.height,
                                                 generation_config.width,
                                                 m_vae->get_config().spatial_compression_ratio);
        OPENVINO_ASSERT(generation_config.max_sequence_length <= 1024,
                        "Gemma3's 'max_sequence_length' must be less or equal to 1024");
        OPENVINO_ASSERT(!generation_config.taylorseer_config,
                        "TaylorSeer is not supported for LTX2 pipelines");
        OPENVINO_ASSERT(!generation_config.adapters, "LoRA adapters are not supported for LTX2 pipelines");
    }

    size_t audio_num_frames_for(const VideoGenerationConfig& generation_config) const {
        const float frame_rate =
            generation_config.frame_rate.value_or(m_default_config->frame_rate.value());
        const double audio_latents_per_second = static_cast<double>(m_audio_vae->get_config().sample_rate) /
                                                m_audio_vae->get_config().mel_hop_length /
                                                m_audio_vae->get_config().temporal_compression_ratio;
        // Match the frame count the video VAE actually produces so audio and video stay in sync
        const auto temporal_ratio = m_vae->get_config().temporal_compression_ratio;
        const int64_t num_frames = (generation_config.num_frames - 1) / temporal_ratio * temporal_ratio + 1;
        const double duration_s = static_cast<double>(num_frames) / frame_rate;
        return static_cast<size_t>(std::lround(duration_s * audio_latents_per_second));
    }

    LTX2TextConnectors::Output compute_hidden_states(const std::string& positive_prompt,
                                                     const std::string& negative_prompt,
                                                     const VideoGenerationConfig& generation_config,
                                                     bool do_classifier_free_guidance) {
        auto text_encoder = m_text_encoder;
        if (!m_deferred_text_encode_device.empty()) {
            text_encoder = m_text_encoder->clone();
            text_encoder->compile(m_deferred_text_encode_device, m_deferred_properties);
        }
        auto infer_start = std::chrono::steady_clock::now();
        ov::Tensor prompt_embeds = text_encoder->infer(positive_prompt,
                                                         negative_prompt,
                                                         do_classifier_free_guidance,
                                                         generation_config.max_sequence_length);
        auto infer_end = std::chrono::steady_clock::now();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = Ms{infer_end - infer_start}.count();

        ov::Tensor prompt_attention_mask = text_encoder->get_prompt_attention_mask();
        prompt_embeds = repeat_per_video(prompt_embeds, generation_config.num_videos_per_prompt);
        prompt_attention_mask = repeat_per_video(prompt_attention_mask, generation_config.num_videos_per_prompt);

        // The output tensors own their storage independently of the compiled encoder.
        text_encoder.reset();
        auto connectors = m_connectors;
        if (!m_deferred_text_encode_device.empty()) {
            connectors = std::make_shared<LTX2TextConnectors>(m_connectors->clone());
            connectors->compile(m_deferred_text_encode_device, m_deferred_properties);
        }
        infer_start = std::chrono::steady_clock::now();
        LTX2TextConnectors::Output connected = connectors->infer(prompt_embeds, prompt_attention_mask);
        infer_end = std::chrono::steady_clock::now();
        m_perf_metrics.encoder_inference_duration["connectors"] = Ms{infer_end - infer_start}.count();

        return connected;
    }

    // Lightricks/LTX-2.5-Diffusers ships the distilled transformer in 'transformer/' and the full one in
    // 'transformer_full/'. Their configs are identical, but the export keeps the folder it was loaded from.
    static bool is_full_transformer_export(const std::filesystem::path& transformer_dir) {
        std::ifstream file(transformer_dir / "config.json");
        if (!file.is_open()) {
            return false;
        }
        const nlohmann::json config = nlohmann::json::parse(file);
        const auto name_or_path = config.find("_name_or_path");
        if (name_or_path == config.end() || !name_or_path->is_string()) {
            return false;
        }
        std::string source = name_or_path->get<std::string>();
        std::replace(source.begin(), source.end(), '\\', '/');
        while (!source.empty() && source.back() == '/') {
            source.pop_back();
        }
        const std::string suffix = "transformer_full";
        return source.size() >= suffix.size() && source.compare(source.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    struct ConditioningRequest {
        const ov::Tensor* image = nullptr;  // a single image, diffusers' LTX2ImageToVideoPipeline
        const ov::Tensor* video = nullptr;  // a source video to edit
        std::vector<VideoCondition> conditions;
    };

    // diffusers' enhance_prompt: the enhancer describes the single image, or the first frame of the first condition
    std::string enhance_prompt(const std::string& prompt,
                               const VideoGenerationConfig& generation_config,
                               const ConditioningRequest& request) {
        OPENVINO_ASSERT(m_prompt_enhancer_tokenizer,
                        "Prompt enhancement needs an LTX-2.5 export with its 'prompt_enhancer' and 'processor'");
        ov::Tensor first_frame;
        if (request.image) {
            first_frame = *request.image;
        } else if (!request.conditions.empty()) {
            const ov::Tensor frames = ltx2_conditioning::as_video(request.conditions.front().frames);
            const ov::Shape shape = frames.get_shape();
            first_frame = ov::Tensor(frames.get_element_type(), {1, shape[1], shape[2], shape[3]}, frames.data());
        }

        // A GPU text encoding device holds its models only for the stage, so does the enhancer
        std::shared_ptr<LTX2PromptEnhancer> enhancer = m_prompt_enhancer;
        if (!enhancer) {
            enhancer = std::make_shared<LTX2PromptEnhancer>(m_models_dir,
                                                            *m_prompt_enhancer_tokenizer,
                                                            m_text_encode_device,
                                                            m_deferred_properties);
            if (m_deferred_text_encode_device.empty()) {
                m_prompt_enhancer = enhancer;
            }
        }
        const auto enhance_start = std::chrono::steady_clock::now();
        const std::string enhanced =
            enhancer->enhance(prompt, generation_config.system_prompt, first_frame ? &first_frame : nullptr);
        m_perf_metrics.encoder_inference_duration["prompt_enhancer"] =
            Ms{std::chrono::steady_clock::now() - enhance_start}.count();
        GENAI_INFO("Enhanced prompt: %s", enhanced.c_str());
        return enhanced;
    }

    // The duration head reads the positive prompt's connector outputs, first video only, like diffusers
    size_t predict_num_frames(const LTX2TextConnectors::Output& connected,
                              const VideoGenerationConfig& generation_config,
                              bool do_classifier_free_guidance) {
        auto duration_head = m_duration_head;
        if (!m_deferred_text_encode_device.empty()) {
            duration_head = std::make_shared<LTX2DurationHead>(m_duration_head->clone());
            duration_head->compile(m_deferred_text_encode_device, m_deferred_properties);
        }
        const size_t positive_row = do_classifier_free_guidance ? generation_config.num_videos_per_prompt : 0;
        auto first_positive = [&](const ov::Tensor& tokens) {
            ov::Shape shape = tokens.get_shape();
            shape[0] = 1;
            ov::Tensor row(tokens.get_element_type(), shape);
            numpy_utils::batch_copy(tokens, row, positive_row, 0, 1);
            return row;
        };
        const auto infer_start = std::chrono::steady_clock::now();
        const float seconds = duration_head->infer(first_positive(connected.video_text_embedding),
                                                   first_positive(connected.audio_text_embedding));
        m_perf_metrics.encoder_inference_duration["duration_head"] =
            Ms{std::chrono::steady_clock::now() - infer_start}.count();

        const float frame_rate = generation_config.frame_rate.value_or(m_default_config->frame_rate.value());
        return LTX2DurationHead::duration_to_num_frames(seconds,
                                                        frame_rate,
                                                        m_vae->get_config().temporal_compression_ratio);
    }

    ov::Tensor decode_video(const ov::Tensor& latent) {
        auto vae = m_vae;
        if (!m_deferred_vae_device.empty()) {
            vae = std::make_shared<AutoencoderKLLTX2Video>(m_vae->clone());
            vae->compile(m_deferred_vae_device, m_deferred_properties);
        }
        return vae->decode(latent);
    }

    // Text conditioning for one transformer pass, all at batch N
    struct PassConditioning {
        ov::Tensor video_text_embedding;
        ov::Tensor audio_text_embedding;
        ov::Tensor connector_attention_mask;
    };

    void bind_conditioning(LTX2VideoTransformer3DModel& transformer, const PassConditioning& conditioning) {
        transformer.set_hidden_states("encoder_hidden_states", conditioning.video_text_embedding);
        transformer.set_hidden_states("audio_encoder_hidden_states", conditioning.audio_text_embedding);
        transformer.set_hidden_states("encoder_attention_mask", conditioning.connector_attention_mask);
        transformer.set_hidden_states("audio_encoder_attention_mask", conditioning.connector_attention_mask);
    }

    // Per-dimension [start, end) patch boundaries in pixel space; temporal axis in seconds (see
    // LTX2AudioVideoRotaryPosEmbed.prepare_video_coords)
    ov::Tensor prepare_video_coords(size_t batch_size, float frame_rate) const {
        const auto& config = m_transformer->get_config();
        const auto& scale = config.vae_scale_factors;
        const int64_t causal_offset = config.causal_offset;
        const size_t sequence_length = m_latent_num_frames * m_latent_height * m_latent_width;

        ov::Tensor coords(ov::element::f32, {batch_size, 3, sequence_length, 2});
        float* data = coords.data<float>();

        auto temporal_bound = [&](int64_t latent_index) {
            const float pixel = static_cast<float>(
                std::max<int64_t>(latent_index * scale[0] + causal_offset - scale[0], 0));
            return pixel / frame_rate;
        };

        size_t token = 0;
        for (size_t f = 0; f < m_latent_num_frames; ++f) {
            for (size_t h = 0; h < m_latent_height; ++h) {
                for (size_t w = 0; w < m_latent_width; ++w, ++token) {
                    data[(0 * sequence_length + token) * 2] = temporal_bound(f);
                    data[(0 * sequence_length + token) * 2 + 1] = temporal_bound(f + 1);
                    data[(1 * sequence_length + token) * 2] = h * scale[1];
                    data[(1 * sequence_length + token) * 2 + 1] = (h + 1) * scale[1];
                    data[(2 * sequence_length + token) * 2] = w * scale[2];
                    data[(2 * sequence_length + token) * 2 + 1] = (w + 1) * scale[2];
                }
            }
        }

        const size_t batch_stride = 3 * sequence_length * 2;
        for (size_t b = 1; b < batch_size; ++b) {
            std::memcpy(data + b * batch_stride, data, batch_stride * sizeof(float));
        }
        return coords;
    }

    // [start, end) timestamps in seconds per latent frame (see prepare_audio_coords)
    ov::Tensor prepare_audio_coords(size_t batch_size, size_t audio_num_frames) const {
        const auto& config = m_transformer->get_config();
        const int64_t scale = config.audio_scale_factor;
        const int64_t causal_offset = config.causal_offset;
        const float seconds_per_mel = static_cast<float>(config.audio_hop_length) / config.audio_sampling_rate;

        ov::Tensor coords(ov::element::f32, {batch_size, 1, audio_num_frames, 2});
        float* data = coords.data<float>();

        auto bound = [&](int64_t latent_index) {
            return std::max<int64_t>(latent_index * scale + causal_offset - scale, 0) * seconds_per_mel;
        };

        for (size_t i = 0; i < audio_num_frames; ++i) {
            data[i * 2] = bound(i);
            data[i * 2 + 1] = bound(i + 1);
        }

        const size_t batch_stride = audio_num_frames * 2;
        for (size_t b = 1; b < batch_size; ++b) {
            std::memcpy(data + b * batch_stride, data, batch_stride * sizeof(float));
        }
        return coords;
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

        return video_generation_utils::denormalize_latents(
            decoded,
            video_generation_utils::tensor_from_vector(m_vae->get_config().latents_mean_data),
            video_generation_utils::tensor_from_vector(m_vae->get_config().latents_std_data),
            m_vae->get_config().scaling_factor);
    }

    // audio_latents = audio_latents * std + mean, applied on packed [B, L, C * M] latents
    void denormalize_audio_latents(ov::Tensor& latents) const {
        const ov::Shape shape = latents.get_shape();
        const size_t packed_dim = shape[2];
        const std::vector<float>& mean = m_audio_vae->get_config().latents_mean_data;
        const std::vector<float>& std_data = m_audio_vae->get_config().latents_std_data;
        if (mean.empty() && std_data.empty()) {
            GENAI_WARN("Audio VAE config carries no latents_mean/std - skipping audio latents denormalization");
            return;
        }
        OPENVINO_ASSERT(mean.size() == packed_dim && std_data.size() == packed_dim,
                        "Audio latents_mean/std size (", mean.size(),
                        ") does not match packed audio channels (", packed_dim, ")");

        float* data = latents.data<float>();
        const size_t rows = shape[0] * shape[1];
        for (size_t row = 0; row < rows; ++row) {
            float* ptr = data + row * packed_dim;
            for (size_t d = 0; d < packed_dim; ++d) {
                ptr[d] = ptr[d] * std_data[d] + mean[d];
            }
        }
    }

public:
    LTX2Pipeline(const std::filesystem::path& root_dir,
                 std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now())
        : LTX2Pipeline(VideoPipelineType::TEXT_2_VIDEO, root_dir, start_time) {}

    LTX2Pipeline(VideoPipelineType pipeline_type,
                 const std::filesystem::path& root_dir,
                 std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now())
        : m_pipeline_type(pipeline_type),
          m_scheduler_config(root_dir / "scheduler/scheduler_config.json") {
        m_models_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";

        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);

        m_video_scheduler = std::make_shared<FlowMatchEulerDiscreteScheduler>(m_scheduler_config);
        m_audio_scheduler = std::make_shared<FlowMatchEulerDiscreteScheduler>(m_scheduler_config);

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        // LTX-2.5's Gemma 4 text encoder is exported with the same interface: packed hidden states of all layers
        if (text_encoder == "Gemma3ForConditionalGeneration" || text_encoder == "Gemma4UnifiedForConditionalGeneration") {
            m_text_encoder = std::make_shared<Gemma3TextEncoder>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string connectors = data["connectors"][1].get<std::string>();
        if (connectors == "LTX2TextConnectors") {
            m_connectors = std::make_shared<LTX2TextConnectors>(root_dir / "connectors");
        } else {
            OPENVINO_THROW("Unsupported '", connectors, "' connectors type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "LTX2VideoTransformer3DModel") {
            m_transformer = std::make_shared<LTX2VideoTransformer3DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKLLTX2Video" && m_pipeline_type == VideoPipelineType::TEXT_2_VIDEO) {
            m_vae = std::make_shared<AutoencoderKLLTX2Video>(root_dir / "vae_decoder");
        } else if (vae == "AutoencoderKLLTX2Video") {
            OPENVINO_ASSERT(std::filesystem::exists(root_dir / "vae_encoder"),
                            "Image and video conditioning need a 'vae_encoder' directory in ", root_dir,
                            ". Export the model with optimum-intel, which includes it for LTX-2.");
            m_vae = std::make_shared<AutoencoderKLLTX2Video>(root_dir / "vae_encoder", root_dir / "vae_decoder");
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string audio_vae = data["audio_vae"][1].get<std::string>();
        if (audio_vae == "AutoencoderKLLTX2Audio") {
            m_audio_vae = std::make_shared<AutoencoderKLLTX2Audio>(root_dir / "audio_vae_decoder");
        } else {
            OPENVINO_THROW("Unsupported '", audio_vae, "' audio VAE decoder type");
        }

        const std::string vocoder = data["vocoder"][1].get<std::string>();
        // LTX-2.3's 'LTX2VocoderWithBWE' has the same IR interface; its 48 kHz output rate comes from the config
        if (vocoder == "LTX2Vocoder" || vocoder == "LTX2VocoderWithBWE") {
            m_vocoder = std::make_shared<LTX2Vocoder>(root_dir / "vocoder");
        } else {
            OPENVINO_THROW("Unsupported '", vocoder, "' vocoder type");
        }

        // LTX-2.5 onwards; LTX-2.3 exports made with newer diffusers list it as [null, null]
        const auto duration_head_entry = data.find("duration_head");
        if (duration_head_entry != data.end() && duration_head_entry->is_array() && duration_head_entry->size() == 2 &&
            (*duration_head_entry)[1].is_string() && std::filesystem::exists(root_dir / "duration_head")) {
            m_duration_head = std::make_shared<LTX2DurationHead>(root_dir / "duration_head");
        }

        if (LTX2PromptEnhancer::is_available(root_dir)) {
            m_prompt_enhancer_tokenizer = std::make_shared<Tokenizer>(root_dir / "processor");
        }

        if (m_duration_head) {
            m_default_config = is_full_transformer_export(root_dir / "transformer") ? &LTX2_5_FULL_DEFAULT_CONFIG
                                                                                     : &LTX2_5_DEFAULT_CONFIG;
        } else {
            // 'perturbed_attn' is set on LTX-2.3 and absent on LTX-2.0
            m_default_config =
                m_transformer->get_config().perturbed_attn ? &LTX2_3_DEFAULT_CONFIG : &LTX2_DEFAULT_CONFIG;
        }
        m_generation_config = *m_default_config;
        m_load_time = Ms{std::chrono::steady_clock::now() - start_time};
    }

    LTX2Pipeline(const std::filesystem::path& models_dir,
                 const std::string& device,
                 const ov::AnyMap& properties,
                 std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now())
        : LTX2Pipeline(VideoPipelineType::TEXT_2_VIDEO, models_dir, device, properties, start_time) {}

    LTX2Pipeline(VideoPipelineType pipeline_type,
                 const std::filesystem::path& models_dir,
                 const std::string& device,
                 const ov::AnyMap& properties,
                 std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now())
        : LTX2Pipeline(pipeline_type, models_dir, start_time) {
        compile(device, properties);
        m_load_time = Ms{std::chrono::steady_clock::now() - start_time};
    }

    std::shared_ptr<VideoPipeline> clone() override {
        OPENVINO_ASSERT(m_is_compiled, "Cannot clone an uncompiled LTX2Pipeline");
        auto cloned = std::make_shared<LTX2Pipeline>(*this);
        cloned->m_generation_config.generator.reset();
        cloned->m_video_scheduler = std::make_shared<FlowMatchEulerDiscreteScheduler>(m_scheduler_config);
        cloned->m_audio_scheduler = std::make_shared<FlowMatchEulerDiscreteScheduler>(m_scheduler_config);
        cloned->m_text_encoder = m_text_encoder->clone();
        cloned->m_connectors = std::make_shared<LTX2TextConnectors>(m_connectors->clone());
        cloned->m_transformer = std::make_shared<LTX2VideoTransformer3DModel>(m_transformer->clone());
        cloned->m_vae = std::make_shared<AutoencoderKLLTX2Video>(m_vae->clone());
        cloned->m_audio_vae = std::make_shared<AutoencoderKLLTX2Audio>(m_audio_vae->clone());
        cloned->m_vocoder = std::make_shared<LTX2Vocoder>(m_vocoder->clone());
        if (m_duration_head) {
            cloned->m_duration_head = std::make_shared<LTX2DurationHead>(m_duration_head->clone());
        }
        cloned->m_prompt_enhancer.reset();
        return cloned;
    }

    // LTX-2.3 enable predicates, matching diffusers' LTX2Pipeline
    static bool do_spatio_temporal_guidance(const VideoGenerationConfig& config) {
        return config.stg_scale.value_or(0.0f) > 0.0f || config.audio_stg_scale.value_or(0.0f) > 0.0f;
    }

    // 1.0 is the neutral modality scale, so it is the fallback for an unset one
    static bool do_modality_isolation_guidance(const VideoGenerationConfig& config) {
        return config.modality_scale.value_or(1.0f) > 1.0f || config.audio_modality_scale.value_or(1.0f) > 1.0f;
    }

    // The transformer always runs at batch N, so CFG is tracked by 'm_compiled_batch_size_multiplier' instead
    size_t get_transformer_expected_batch_size() const override {
        return 0;
    }

    void reshape_models(const VideoGenerationConfig& generation_config, size_t batch_size_multiplier) override {
        OPENVINO_ASSERT(generation_config.num_frames > 0,
                        "Static shapes need 'num_frames': the duration head cannot choose it after a reshape");
        m_reshape_batch_size_multiplier = batch_size_multiplier;
        const size_t audio_num_frames = audio_num_frames_for(generation_config);
        m_text_encoder->reshape(batch_size_multiplier, generation_config.max_sequence_length);
        m_connectors->reshape(generation_config.num_videos_per_prompt * batch_size_multiplier);
        // Image and video conditioning pass one timestep per latent frame. Conditions placed after the first frame
        // add tokens, which a static sequence cannot take.
        m_transformer->reshape(generation_config.num_videos_per_prompt,
                               generation_config.num_frames,
                               generation_config.height,
                               generation_config.width,
                               audio_num_frames,
                               m_pipeline_type != VideoPipelineType::TEXT_2_VIDEO);
        m_vae->reshape(generation_config.num_videos_per_prompt,
                       generation_config.num_frames,
                       generation_config.height,
                       generation_config.width);
        m_audio_vae->reshape(generation_config.num_videos_per_prompt, audio_num_frames);
        m_vocoder->reshape(generation_config.num_videos_per_prompt);
        if (m_duration_head) {
            // It predicts for the first video only
            m_duration_head->reshape(1);
        }
    }

    VideoGenerationResult generate(const std::string& positive_prompt, const ov::AnyMap& properties) override {
        return generate_impl(positive_prompt, properties, ConditioningRequest{});
    }

    // diffusers' LTX2ImageToVideoPipeline: the image becomes the first frame
    VideoGenerationResult generate(const ov::Tensor& image,
                                   const std::string& positive_prompt,
                                   const ov::AnyMap& properties) override {
        OPENVINO_ASSERT(m_pipeline_type == VideoPipelineType::IMAGE_2_VIDEO,
                        "Image-to-video generation needs a pipeline created for it");
        ConditioningRequest request;
        request.image = &image;
        return generate_impl(positive_prompt, properties, request);
    }

    // diffusers' LTX2ConditionPipeline: conditions, and a source video to edit
    VideoGenerationResult generate(const ov::Tensor* source_video,
                                   const std::vector<VideoCondition>& conditions,
                                   const std::string& positive_prompt,
                                   const ov::AnyMap& properties) override {
        OPENVINO_ASSERT(m_pipeline_type != VideoPipelineType::TEXT_2_VIDEO,
                        "Image and video conditions need a pipeline created for image-to-video or video-to-video");
        ConditioningRequest request;
        request.video = source_video;
        request.conditions = conditions;
        return generate_impl(positive_prompt, properties, request);
    }

private:
    // The conditioned latent sequence: the video's tokens, then the keyframe tokens of conditions placed after the
    // first frame
    struct PreparedConditioning {
        bool active = false;
        // diffusers' LTX2ConditionPipeline draws the noise on the packed sequence, the other pipelines on 5D latents
        bool packed_noise = false;
        ov::Tensor init;                 // [N, S, C] latents before noising: the edited video, conditions, zeros
        ov::Tensor clean;                // [N, S, C] the conditions' latents, zeros elsewhere
        std::vector<float> frame_mask;   // conditioning mask per latent frame of the sequence
        ov::Tensor keyframe_coords;      // [N, 3, K, 2] positions of the keyframe tokens, empty without them
        size_t base_tokens = 0;          // tokens of the generated video
    };

    // Writes 'packed' [1, T, C] into every video of 'destination' [N, S, C] from token 'offset'
    static void write_tokens(ov::Tensor& destination, const ov::Tensor& packed, size_t offset) {
        const ov::Shape dst_shape = destination.get_shape(), src_shape = packed.get_shape();
        const size_t row = src_shape[1] * src_shape[2];
        for (size_t b = 0; b < dst_shape[0]; ++b) {
            std::memcpy(destination.data<float>() + (b * dst_shape[1] + offset) * dst_shape[2],
                        packed.data<const float>(),
                        row * sizeof(float));
        }
    }

    // Concatenates [N, S1, C] and [N, S2, C] along the tokens
    static ov::Tensor concat_tokens(const ov::Tensor& first, const ov::Tensor& second) {
        const ov::Shape a = first.get_shape(), b = second.get_shape();
        ov::Tensor out(ov::element::f32, {a[0], a[1] + b[1], a[2]});
        float* dst = out.data<float>();
        for (size_t n = 0; n < a[0]; ++n) {
            std::memcpy(dst, first.data<const float>() + n * a[1] * a[2], a[1] * a[2] * sizeof(float));
            dst += a[1] * a[2];
            std::memcpy(dst, second.data<const float>() + n * b[1] * b[2], b[1] * b[2] * sizeof(float));
            dst += b[1] * b[2];
        }
        return out;
    }

    PreparedConditioning prepare_conditioning(const ConditioningRequest& request,
                                              const VideoGenerationConfig& config,
                                              bool edit,
                                              float frame_rate) {
        PreparedConditioning prepared;
        const size_t num_videos = config.num_videos_per_prompt;
        const auto& transformer_config = m_transformer->get_config();
        const size_t channels = transformer_config.in_channels;
        const size_t tokens_per_frame = m_latent_height * m_latent_width;
        const auto temporal_ratio = static_cast<size_t>(m_vae->get_config().temporal_compression_ratio);
        prepared.base_tokens = m_latent_num_frames * tokens_per_frame;
        prepared.frame_mask.assign(m_latent_num_frames, 0.0f);
        if (!request.image && !edit && request.conditions.empty()) {
            return prepared;
        }
        prepared.active = true;
        prepared.packed_noise = request.image == nullptr;

        auto vae = m_vae;
        if (!m_deferred_vae_device.empty()) {
            vae = std::make_shared<AutoencoderKLLTX2Video>(m_vae->clone());
            vae->compile(m_deferred_vae_device, m_deferred_properties);
        }
        const auto encode_start = std::chrono::steady_clock::now();
        auto encode_packed = [&](const ov::Tensor& pixels) {
            ov::Tensor latent = vae->encode(pixels);
            return video_generation_utils::pack_latents(latent,
                                                        transformer_config.patch_size,
                                                        transformer_config.patch_size_t);
        };

        prepared.init = ov::Tensor(ov::element::f32, {num_videos, prepared.base_tokens, channels});
        std::fill_n(prepared.init.data<float>(), prepared.init.get_size(), 0.0f);
        prepared.clean = ov::Tensor(ov::element::f32, prepared.init.get_shape());
        std::fill_n(prepared.clean.data<float>(), prepared.clean.get_size(), 0.0f);

        if (request.image) {
            ov::Tensor image = ltx2_conditioning::as_video(*request.image);
            OPENVINO_ASSERT(image.get_shape()[0] == 1, "Image-to-video takes a single image");
            if (image.get_shape()[1] != static_cast<size_t>(config.height) ||
                image.get_shape()[2] != static_cast<size_t>(config.width)) {
                // diffusers' VideoProcessor resizes with PIL, bilinear
                image = ltx2_conditioning::pillow_resize(image, config.height, config.width, /* bicubic */ false);
            }
            const ov::Tensor packed = encode_packed(ltx2_conditioning::image_to_pixels(image));
            write_tokens(prepared.init, packed, 0);
            write_tokens(prepared.clean, packed, 0);
            prepared.frame_mask[0] = 1.0f;
        }

        if (edit) {
            const ov::Tensor source = ltx2_conditioning::as_video(*request.video);
            const ov::Shape shape = source.get_shape();
            const size_t keep = ltx2_conditioning::trim_num_frames(0, shape[0], config.num_frames, temporal_ratio);
            const ov::Tensor frames(ov::element::u8, {keep, shape[1], shape[2], shape[3]}, source.data());
            write_tokens(prepared.init,
                         encode_packed(ltx2_conditioning::preprocess_frames(frames, config.height, config.width)),
                         0);
        }

        std::vector<float> keyframe_mask;
        ov::Tensor keyframe_tokens;
        for (size_t i = 0; i < request.conditions.size(); ++i) {
            const VideoCondition& condition = request.conditions[i];
            size_t latent_index = 0;
            if (!ltx2_conditioning::resolve_latent_index(condition.index, m_latent_num_frames, latent_index)) {
                GENAI_WARN("The latent index %lld of condition %zu is too big for %zu latent frames; the condition "
                           "is skipped",
                           static_cast<long long>(condition.index), i, m_latent_num_frames);
                continue;
            }
            const ov::Tensor frames = ltx2_conditioning::as_video(condition.frames);
            const ov::Shape shape = frames.get_shape();
            const size_t start_frame = latent_index == 0 ? 0 : (latent_index - 1) * temporal_ratio + 1;
            const size_t keep =
                ltx2_conditioning::trim_num_frames(start_frame, shape[0], config.num_frames, temporal_ratio);
            const ov::Tensor kept(ov::element::u8, {keep, shape[1], shape[2], shape[3]}, frames.data());
            const ov::Tensor packed =
                encode_packed(ltx2_conditioning::preprocess_frames(kept, config.height, config.width));
            const size_t latent_frames = packed.get_shape()[1] / tokens_per_frame;

            if (latent_index == 0) {
                // Replaces the first frames of the video
                write_tokens(prepared.init, packed, 0);
                write_tokens(prepared.clean, packed, 0);
                std::fill_n(prepared.frame_mask.begin(), latent_frames, condition.strength);
                continue;
            }
            // Appended as keyframe tokens positioned at their frame
            ov::Tensor tokens(ov::element::f32, {num_videos, packed.get_shape()[1], channels});
            write_tokens(tokens, packed, 0);
            keyframe_tokens = keyframe_tokens ? concat_tokens(keyframe_tokens, tokens) : tokens;
            keyframe_mask.insert(keyframe_mask.end(), latent_frames, condition.strength);
            const ov::Tensor coords = ltx2_conditioning::keyframe_coords(num_videos,
                                                                         latent_frames,
                                                                         m_latent_height,
                                                                         m_latent_width,
                                                                         start_frame,
                                                                         keep,
                                                                         frame_rate,
                                                                         transformer_config.vae_scale_factors);
            prepared.keyframe_coords = prepared.keyframe_coords
                                           ? ltx2_conditioning::concat_coords(prepared.keyframe_coords, coords)
                                           : coords;
        }
        if (keyframe_tokens) {
            prepared.init = concat_tokens(prepared.init, keyframe_tokens);
            prepared.clean = concat_tokens(prepared.clean, keyframe_tokens);
            prepared.frame_mask.insert(prepared.frame_mask.end(), keyframe_mask.begin(), keyframe_mask.end());
        }
        m_perf_metrics.encoder_inference_duration["vae_encoder"] =
            Ms{std::chrono::steady_clock::now() - encode_start}.count();
        return prepared;
    }

    VideoGenerationResult generate_impl(const std::string& positive_prompt,
                                        const ov::AnyMap& properties,
                                        const ConditioningRequest& request) {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();

        VideoGenerationConfig merged_generation_config = merge_generation_config(properties);
        // A step count passed to this call asks for the scheduler's own schedule, not the default sigmas
        if (properties.count(ov::genai::num_inference_steps.name()) && !properties.count(ov::genai::sigmas.name())) {
            merged_generation_config.sigmas.reset();
        }
        // Editing noises the whole source video, so the output takes its length and size unless they are given
        const bool edit = request.video != nullptr &&
                          (merged_generation_config.noise_scale || merged_generation_config.edit_num_steps);
        OPENVINO_ASSERT(edit || !(merged_generation_config.noise_scale || merged_generation_config.edit_num_steps),
                        "'noise_scale' and 'edit_num_steps' edit a source video: they apply to video-to-video only");
        OPENVINO_ASSERT(request.video == nullptr || edit || !request.conditions.empty(),
                        "Video-to-video needs an edit ('noise_scale' and / or 'edit_num_steps') or 'conditions'");
        OPENVINO_ASSERT(m_pipeline_type != VideoPipelineType::IMAGE_2_VIDEO || request.image || !request.conditions.empty(),
                        "Image-to-video needs at least one image");
        if (edit) {
            const ov::Shape source_shape = ltx2_conditioning::as_video(*request.video).get_shape();
            const auto temporal_ratio = static_cast<size_t>(m_vae->get_config().temporal_compression_ratio);
            const auto spatial_ratio = static_cast<int64_t>(m_vae->get_config().spatial_compression_ratio);
            if (!properties.count(ov::genai::num_frames.name())) {
                merged_generation_config.num_frames = (source_shape[0] - 1) / temporal_ratio * temporal_ratio + 1;
            }
            if (!properties.count(ov::genai::height.name())) {
                merged_generation_config.height = static_cast<int64_t>(source_shape[1]) / spatial_ratio * spatial_ratio;
            }
            if (!properties.count(ov::genai::width.name())) {
                merged_generation_config.width = static_cast<int64_t>(source_shape[2]) / spatial_ratio * spatial_ratio;
            }
        }
        const float audio_guidance_scale =
            merged_generation_config.audio_guidance_scale.value_or(merged_generation_config.guidance_scale);

        // Matches diffusers: CFG is enabled when either modality requests guidance
        const bool cfg_requested = utils::requests_classifier_free_guidance(merged_generation_config);
        const size_t batch_size_multiplier = resolve_batch_size_multiplier(merged_generation_config, cfg_requested);
        const bool use_classifier_free_guidance = batch_size_multiplier > 1;

        check_inputs(merged_generation_config);
        OPENVINO_ASSERT(merged_generation_config.generator, "Generator must not be null");

        const auto& transformer_config = m_transformer->get_config();
        const size_t num_videos_per_prompt = merged_generation_config.num_videos_per_prompt;
        const float frame_rate =
            merged_generation_config.frame_rate.value_or(m_default_config->frame_rate.value());

        const std::string prompt = merged_generation_config.enable_prompt_enhancement
                                       ? enhance_prompt(positive_prompt, merged_generation_config, request)
                                       : positive_prompt;
        LTX2TextConnectors::Output connected = compute_hidden_states(prompt,
                                                                     merged_generation_config.negative_prompt.value_or(""),
                                                                     merged_generation_config,
                                                                     use_classifier_free_guidance);

        if (merged_generation_config.num_frames == 0) {
            OPENVINO_ASSERT(m_duration_head, "'num_frames' must be set: this model has no duration head to choose it");
            merged_generation_config.num_frames =
                predict_num_frames(connected, merged_generation_config, use_classifier_free_guidance);
        }

        m_latent_num_frames =
            (merged_generation_config.num_frames - 1) / m_vae->get_config().temporal_compression_ratio + 1;
        m_latent_height = merged_generation_config.height / m_vae->get_config().spatial_compression_ratio;
        m_latent_width = merged_generation_config.width / m_vae->get_config().spatial_compression_ratio;

        const size_t audio_num_frames = audio_num_frames_for(merged_generation_config);
        const size_t latent_mel_bins =
            m_audio_vae->get_config().mel_bins / m_audio_vae->get_config().mel_compression_ratio;

        const PreparedConditioning conditioning =
            prepare_conditioning(request, merged_generation_config, edit, frame_rate);
        const size_t tokens_per_frame = m_latent_height * m_latent_width;
        const size_t sequence_length =
            conditioning.active ? conditioning.init.get_shape()[1] : conditioning.base_tokens;
        // The timestep of each latent frame is t * (1 - mask): conditioned frames are less noisy, or clean
        std::vector<float> frame_timestep_scales;
        if (conditioning.active) {
            for (const float mask : conditioning.frame_mask) {
                frame_timestep_scales.push_back(1.0f - mask);
            }
        }

        auto transformer = m_transformer;
        if (!m_deferred_denoise_device.empty()) {
            transformer = std::make_shared<LTX2VideoTransformer3DModel>(m_transformer->clone());
            transformer->compile(m_deferred_denoise_device, m_deferred_properties);
        }

        const float guidance_rescale = *merged_generation_config.guidance_rescale;
        const float audio_guidance_rescale =
            merged_generation_config.audio_guidance_rescale.value_or(guidance_rescale);

        // LTX-2.3's extra guidance passes, gated on the model exposing their inputs
        const float stg_scale = merged_generation_config.stg_scale.value_or(0.0f);
        const float audio_stg_scale = merged_generation_config.audio_stg_scale.value_or(stg_scale);
        const float modality_scale = merged_generation_config.modality_scale.value_or(1.0f);
        const float audio_modality_scale = merged_generation_config.audio_modality_scale.value_or(modality_scale);
        const bool stg_requested = do_spatio_temporal_guidance(merged_generation_config);
        const bool modality_isolation_requested = do_modality_isolation_guidance(merged_generation_config);
        const bool use_spatio_temporal_guidance = transformer->has_stg_perturbation_mask() && stg_requested;
        const bool use_modality_isolation_guidance = transformer->has_cross_modality_gate() && modality_isolation_requested;
        // Warn rather than assert, so exports predating these inputs keep working
        if (stg_requested && !transformer->has_stg_perturbation_mask()) {
            GENAI_WARN("'stg_scale' / 'audio_stg_scale' request Spatio-Temporal Guidance, but this "
                       "transformer has no 'stg_perturbation_mask' input and the pass is skipped. Re-export "
                       "the model with LTX-2.3 support, or set both scales to 0 to silence this.");
        }
        if (modality_isolation_requested && !transformer->has_cross_modality_gate()) {
            GENAI_WARN("'modality_scale' / 'audio_modality_scale' request modality isolation guidance, but "
                       "this transformer has no 'cross_modality_gate' input and the pass is skipped. "
                       "Re-export the model with LTX-2.3 support, or set both scales to 1.0 to silence this.");
        }
        const std::vector<int64_t> stg_blocks =
            merged_generation_config.spatio_temporal_guidance_blocks.value_or(std::vector<int64_t>{});
        OPENVINO_ASSERT(!use_spatio_temporal_guidance || !stg_blocks.empty(),
                        "Spatio-Temporal Guidance is enabled but 'spatio_temporal_guidance_blocks' is empty, "
                        "so no block would be perturbed. Set the blocks, or set 'stg_scale' and "
                        "'audio_stg_scale' to 0.");
        const bool use_extra_guidance_passes = use_spatio_temporal_guidance || use_modality_isolation_guidance;

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        // Every pass runs at batch N, CFG included, so the transformer keeps a static batch. The positions
        // are the same for every pass, so RoPE is bound once for the whole generation.
        ov::Tensor video_coords = prepare_video_coords(num_videos_per_prompt, frame_rate);
        if (conditioning.keyframe_coords) {
            video_coords = ltx2_conditioning::concat_coords(video_coords, conditioning.keyframe_coords);
        }
        transformer->set_rope_coords(video_coords, prepare_audio_coords(num_videos_per_prompt, audio_num_frames));
        for (const auto& condition : std::map<std::string, size_t>{
                 {"num_frames", m_latent_num_frames},
                 {"height", m_latent_height},
                 {"width", m_latent_width},
                 {"audio_num_frames", audio_num_frames}}) {
            if (transformer->has_input(condition.first)) {
                transformer->set_hidden_states(condition.first,
                                                video_generation_utils::make_i64_scalar(condition.second));
            }
        }
        if (transformer->has_input("fps")) {
            ov::Tensor fps(ov::element::f32, {});
            fps.data<float>()[0] = frame_rate;
            transformer->set_hidden_states("fps", fps);
        }
        auto conditioning_half = [&](size_t half_index) {
            return PassConditioning{cfg_half(connected.video_text_embedding, num_videos_per_prompt, half_index),
                                    cfg_half(connected.audio_text_embedding, num_videos_per_prompt, half_index),
                                    cfg_half(connected.connector_attention_mask, num_videos_per_prompt, half_index)};
        };
        PassConditioning unconditional_conditioning;
        PassConditioning conditional_conditioning;
        if (use_classifier_free_guidance) {
            unconditional_conditioning = conditioning_half(0);
            conditional_conditioning = conditioning_half(1);
        } else {
            conditional_conditioning = {connected.video_text_embedding,
                                        connected.audio_text_embedding,
                                        connected.connector_attention_mask};
        }
        bind_conditioning(*transformer, conditional_conditioning);

        // mu from the packed video sequence length, as in the reference; audio reuses it
        const double mu = m_video_scheduler->calculate_shift(sequence_length);
        // Editing runs the last steps of the schedule, sliced before the scheduler shifts it. Shifting maps each
        // sigma on its own and the terminal stretch keeps the last one, so this is the tail of the shifted schedule.
        std::optional<std::vector<float>> schedule = merged_generation_config.sigmas;
        if (edit) {
            std::vector<float> full = schedule.value_or(std::vector<float>{});
            if (full.empty()) {
                const auto steps = static_cast<size_t>(merged_generation_config.num_inference_steps);
                for (const double sigma : numpy_utils::linspace<double>(1.0, 1.0 / steps, steps, true)) {
                    full.push_back(static_cast<float>(sigma));
                }
            }
            size_t steps = full.size();
            if (merged_generation_config.edit_num_steps) {
                steps = std::min(*merged_generation_config.edit_num_steps, full.size());
            } else {
                steps = std::count_if(full.begin(), full.end(), [&](float sigma) {
                    return sigma <= *merged_generation_config.noise_scale + 1e-6f;
                });
            }
            OPENVINO_ASSERT(steps > 0, "No step of the schedule is at or below 'noise_scale' ",
                            *merged_generation_config.noise_scale);
            schedule = std::vector<float>(full.end() - steps, full.end());
        }
        // Separate scheduler instance for audio: step() tracks per-modality state
        if (schedule) {
            m_video_scheduler->set_timesteps_from_sigmas(*schedule, mu);
            m_audio_scheduler->set_timesteps_from_sigmas(*schedule, mu);
        } else {
            m_video_scheduler->set_timesteps_with_mu(mu, merged_generation_config.num_inference_steps, 1.0f);
            m_audio_scheduler->set_timesteps_with_mu(mu, merged_generation_config.num_inference_steps, 1.0f);
        }
        std::vector<float> timesteps = m_video_scheduler->get_float_timesteps();

        // The noise layouts of the diffusers pipelines this follows, so a seed gives the same video
        ov::Tensor latent;
        if (conditioning.packed_noise) {
            latent = merged_generation_config.generator->randn_tensor(
                ov::Shape{num_videos_per_prompt, sequence_length, transformer_config.in_channels});
        } else {
            ov::Shape video_noise_shape{num_videos_per_prompt,
                                        transformer_config.in_channels,
                                        m_latent_num_frames,
                                        m_latent_height,
                                        m_latent_width};
            ov::Tensor video_noise = merged_generation_config.generator->randn_tensor(video_noise_shape);
            latent = video_generation_utils::pack_latents(video_noise,
                                                          transformer_config.patch_size,
                                                          transformer_config.patch_size_t);
        }
        if (conditioning.active) {
            // Noised to 'noise_scale' where nothing is kept: latents = noise * s + init * (1 - s), s = (1 - mask) * noise_scale.
            // diffusers' LTX2ConditionPipeline defaults it to the first sigma; image-to-video starts from pure noise.
            const float noise_scale = conditioning.packed_noise
                                          ? merged_generation_config.noise_scale.value_or(schedule ? schedule->front() : 1.0f)
                                          : 1.0f;
            const size_t channels = transformer_config.in_channels;
            float* data = latent.data<float>();
            const float* init = conditioning.init.data<const float>();
            for (size_t n = 0; n < num_videos_per_prompt; ++n) {
                for (size_t token = 0; token < sequence_length; ++token) {
                    const float scaled = (1.0f - conditioning.frame_mask[token / tokens_per_frame]) * noise_scale;
                    const size_t offset = (n * sequence_length + token) * channels;
                    for (size_t c = 0; c < channels; ++c) {
                        data[offset + c] = data[offset + c] * scaled + init[offset + c] * (1.0f - scaled);
                    }
                }
            }
        }

        ov::Tensor audio_latent;
        if (conditioning.packed_noise) {
            audio_latent = merged_generation_config.generator->randn_tensor(ov::Shape{
                num_videos_per_prompt, audio_num_frames, m_audio_vae->get_config().latent_channels * latent_mel_bins});
        } else {
            ov::Shape audio_noise_shape{num_videos_per_prompt,
                                        m_audio_vae->get_config().latent_channels,
                                        audio_num_frames,
                                        latent_mel_bins};
            audio_latent = pack_audio_latents(merged_generation_config.generator->randn_tensor(audio_noise_shape));
        }

        // x0-space guidance, as in diffusers' LTX2Pipeline. Each term is 'scale * sigma * (v_other - v_cond)':
        //   guided = x0_cond + (gs - 1) * (x0_cond - x0_uncond_text)   classifier-free guidance
        //                    +  stg     * (x0_cond - x0_uncond_stg)    Spatio-Temporal Guidance
        //                    + (ms - 1) * (x0_cond - x0_uncond_modal)  modality isolation
        // then rescale_noise_cfg(guided, x0_cond), and back to velocity: v = (sample - guided) / sigma.
        struct GuidanceTerm {
            float scale;
            const float* velocity;  // the other pass's prediction, same element count as 'v_cond'
        };
        ov::Tensor x0_cond(ov::element::f32, {}), guided(ov::element::f32, {});
        auto guided_velocity = [&](const float* v_cond,
                                   const ov::Tensor& sample,
                                   const std::vector<GuidanceTerm>& terms,
                                   float rescale,
                                   float sigma,
                                   ov::Tensor& velocity) {
            const ov::Shape& guided_shape = sample.get_shape();
            x0_cond.set_shape(guided_shape);
            guided.set_shape(guided_shape);

            const size_t elems = x0_cond.get_size();
            const float* sample_data = sample.data<const float>();
            float* x0_cond_data = x0_cond.data<float>();
            float* guided_data = guided.data<float>();

            for (size_t i = 0; i < elems; ++i) {
                x0_cond_data[i] = sample_data[i] - v_cond[i] * sigma;
                guided_data[i] = x0_cond_data[i];
            }
            for (const GuidanceTerm& term : terms) {
                for (size_t i = 0; i < elems; ++i) {
                    guided_data[i] += term.scale * sigma * (term.velocity[i] - v_cond[i]);
                }
            }

            if (rescale > 0.0f) {
                video_generation_utils::rescale_noise_cfg(guided_data,
                                                          x0_cond_data,
                                                          guided_shape[0],
                                                          elems / guided_shape[0],
                                                          rescale);
            }

            velocity.set_shape(guided_shape);
            float* velocity_data = velocity.data<float>();
            for (size_t i = 0; i < elems; ++i) {
                velocity_data[i] = (sample_data[i] - guided_data[i]) / sigma;
            }
            return velocity;
        };
        ov::Tensor video_velocity_buffer(ov::element::f32, {}), audio_velocity_buffer(ov::element::f32, {});
        ov::Tensor blended_velocity(ov::element::f32, {});

        // infer() returns the request's output tensors, which the next pass overwrites
        ov::Tensor video_pred(ov::element::f32, {}), audio_pred(ov::element::f32, {});
        ov::Tensor video_pred_uncond(ov::element::f32, {}), audio_pred_uncond(ov::element::f32, {});
        ov::Tensor video_pred_stg(ov::element::f32, {}), audio_pred_stg(ov::element::f32, {});
        ov::Tensor video_pred_modality(ov::element::f32, {}), audio_pred_modality(ov::element::f32, {});
        auto keep = [](const ov::Tensor& source, ov::Tensor& destination) {
            destination.set_shape(source.get_shape());
            source.copy_to(destination);
        };

        auto run_pass = [&](float t, bool isolate_modalities, const std::vector<int64_t>& blocks) {
            const auto pass_start = std::chrono::steady_clock::now();
            auto predictions =
                transformer->infer(latent, audio_latent, t, isolate_modalities, blocks, frame_timestep_scales);
            const auto pass_duration =
                ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - pass_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(pass_duration));
            return predictions;
        };
        auto run_and_keep = [&](float t, bool isolate_modalities, const std::vector<int64_t>& blocks,
                                ov::Tensor& video_out, ov::Tensor& audio_out) {
            auto [video_prediction, audio_prediction] = run_pass(t, isolate_modalities, blocks);
            keep(video_prediction, video_out);
            keep(audio_prediction, audio_out);
        };

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();
            const float t = timesteps[inference_step];
            const float sigma = t / m_scheduler_config.num_train_timesteps;

            if (use_classifier_free_guidance) {
                bind_conditioning(*transformer, unconditional_conditioning);
                run_and_keep(t, /* isolate_modalities */ false, {}, video_pred_uncond, audio_pred_uncond);
                bind_conditioning(*transformer, conditional_conditioning);
            }
            auto [noise_pred_video, noise_pred_audio] = run_pass(t, /* isolate_modalities */ false, {});

            ov::Tensor video_velocity, audio_velocity;
            if (!use_classifier_free_guidance && !use_extra_guidance_passes) {
                // Unguided: the single prediction is the velocity, no x0 round trip
                video_velocity = noise_pred_video;
                audio_velocity = noise_pred_audio;
            } else {
                // The conditional prediction is the guidance baseline; copy it only if an extra pass follows
                if (use_extra_guidance_passes) {
                    keep(noise_pred_video, video_pred);
                    keep(noise_pred_audio, audio_pred);
                }
                const float* v_cond = (use_extra_guidance_passes ? video_pred : noise_pred_video).data<const float>();
                const float* audio_v_cond =
                    (use_extra_guidance_passes ? audio_pred : noise_pred_audio).data<const float>();

                std::vector<GuidanceTerm> video_terms, audio_terms;
                if (use_classifier_free_guidance) {
                    video_terms.push_back({merged_generation_config.guidance_scale - 1.0f,
                                           video_pred_uncond.data<const float>()});
                    audio_terms.push_back({audio_guidance_scale - 1.0f, audio_pred_uncond.data<const float>()});
                }

                // The extra passes are conditional-only; the conditional text is still bound from above
                if (use_spatio_temporal_guidance) {
                    run_and_keep(t, /* isolate_modalities */ false, stg_blocks, video_pred_stg, audio_pred_stg);
                    video_terms.push_back({stg_scale, video_pred_stg.data<const float>()});
                    audio_terms.push_back({audio_stg_scale, audio_pred_stg.data<const float>()});
                }

                if (use_modality_isolation_guidance) {
                    run_and_keep(t, /* isolate_modalities */ true, {}, video_pred_modality, audio_pred_modality);
                    // Each modality contributes only above its own threshold
                    if (modality_scale > 1.0f) {
                        video_terms.push_back({modality_scale - 1.0f, video_pred_modality.data<const float>()});
                    }
                    if (audio_modality_scale > 1.0f) {
                        audio_terms.push_back({audio_modality_scale - 1.0f, audio_pred_modality.data<const float>()});
                    }
                }

                video_velocity = guided_velocity(v_cond, latent, video_terms, guidance_rescale, sigma,
                                                 video_velocity_buffer);
                audio_velocity = guided_velocity(audio_v_cond, audio_latent, audio_terms,
                                                 audio_guidance_rescale, sigma, audio_velocity_buffer);
            }

            if (conditioning.active) {
                // diffusers' post-process blend: x0 = x0 * (1 - mask) + clean * mask, then back to a velocity.
                // Fully kept tokens (mask 1) stay exactly at their clean latents.
                blended_velocity.set_shape(latent.get_shape());
                const size_t channels = transformer_config.in_channels;
                const float* sample = latent.data<const float>();
                const float* velocity = video_velocity.data<const float>();
                const float* clean = conditioning.clean.data<const float>();
                float* out = blended_velocity.data<float>();
                for (size_t n = 0; n < num_videos_per_prompt; ++n) {
                    for (size_t token = 0; token < sequence_length; ++token) {
                        const float mask = conditioning.frame_mask[token / tokens_per_frame];
                        const size_t offset = (n * sequence_length + token) * channels;
                        for (size_t c = 0; c < channels; ++c) {
                            const size_t i = offset + c;
                            const float x0 = sample[i] - velocity[i] * sigma;
                            const float blended = x0 * (1.0f - mask) + clean[i] * mask;
                            out[i] = (sample[i] - blended) / sigma;
                        }
                    }
                }
                video_velocity = blended_velocity;
            }

            auto video_step_result = m_video_scheduler->step(video_velocity,
                                                             latent,
                                                             inference_step,
                                                             merged_generation_config.generator);
            latent = video_step_result["latent"];
            auto audio_step_result = m_audio_scheduler->step(audio_velocity,
                                                             audio_latent,
                                                             inference_step,
                                                             merged_generation_config.generator);
            audio_latent = audio_step_result["latent"];

            if (callback_ptr && callback_ptr->has_callback() && callback_ptr->write(inference_step, timesteps.size(), latent) == CallbackStatus::STOP) {
                callback_ptr->end();
                auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
                m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));

                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start)
                        .count();
                return {ov::Tensor(ov::element::u8, {}), m_perf_metrics, ov::Tensor(ov::element::f32, ov::Shape{0})};
            }

            auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
        }

        if (callback_ptr != nullptr) {
            callback_ptr->end();
        }

        transformer.reset();
        if (conditioning.active && sequence_length != conditioning.base_tokens) {
            // The keyframe tokens only steer the video; decode the video's own tokens
            const size_t channels = transformer_config.in_channels;
            ov::Tensor video_tokens(ov::element::f32, {num_videos_per_prompt, conditioning.base_tokens, channels});
            for (size_t n = 0; n < num_videos_per_prompt; ++n) {
                std::memcpy(video_tokens.data<float>() + n * conditioning.base_tokens * channels,
                            latent.data<const float>() + n * sequence_length * channels,
                            conditioning.base_tokens * channels * sizeof(float));
            }
            latent = video_tokens;
        }
        latent = postprocess_latents(latent);

        OPENVINO_ASSERT(!m_vae->get_config().timestep_conditioning,
                        "Parameter 'timestep_conditioning' is not currently supported by AutoencoderKLLTX2Video. Please, contact OpenVINO GenAI developers.");

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = decode_video(latent);

        denormalize_audio_latents(audio_latent);
        ov::Tensor audio_latent_unpacked =
            unpack_audio_latents(audio_latent, m_audio_vae->get_config().latent_channels, latent_mel_bins);
        auto audio_vae = m_audio_vae;
        if (!m_deferred_vae_device.empty()) {
            audio_vae = std::make_shared<AutoencoderKLLTX2Audio>(m_audio_vae->clone());
            audio_vae->compile(m_deferred_vae_device, m_deferred_properties);
        }
        ov::Tensor mel_spectrogram = audio_vae->decode(audio_latent_unpacked);
        audio_vae.reset();

        auto vocoder = m_vocoder;
        if (!m_deferred_vae_device.empty()) {
            vocoder = std::make_shared<LTX2Vocoder>(m_vocoder->clone());
            vocoder->compile(m_deferred_vae_device, m_deferred_properties);
        }
        ov::Tensor audio = vocoder->infer(mel_spectrogram);
        vocoder.reset();

        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();

        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();

        return VideoGenerationResult{video,
                                     m_perf_metrics,
                                     audio,
                                     static_cast<uint32_t>(m_vocoder->get_config().output_sampling_rate)};
    }

public:
    VideoGenerationResult decode(const ov::Tensor& latent) override {
        ov::Tensor postprocessed = postprocess_latents(latent);

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = decode_video(postprocessed);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();

        return VideoGenerationResult{video, m_perf_metrics, ov::Tensor(ov::element::f32, ov::Shape{0})};
    }

    void reshape(int64_t num_videos_per_prompt,
                 int64_t num_frames,
                 int64_t height,
                 int64_t width,
                 float guidance_scale) override {
        video_generation_utils::check_video_size(height, width, m_vae->get_config().spatial_compression_ratio);

        VideoGenerationConfig reshaped_config = m_generation_config;
        reshaped_config.num_videos_per_prompt = num_videos_per_prompt;
        reshaped_config.num_frames = num_frames;
        reshaped_config.height = height;
        reshaped_config.width = width;
        reshaped_config.guidance_scale = guidance_scale;
        // The audio scale can request CFG on its own, so size the batch on both
        const size_t batch_size_multiplier = utils::requests_classifier_free_guidance(reshaped_config) ? 2 : 1;
        reshape_models(reshaped_config, batch_size_multiplier);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        OPENVINO_ASSERT(!m_is_compiled, "LTX2Pipeline has already been compiled");
        std::optional<AdapterConfig> adapters;
        auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
        OPENVINO_ASSERT(!adapters || adapters->get_adapters().empty(),
                        "LoRA adapters are not supported for LTX2 pipelines");
        const auto is_gpu = [](const std::string& device) {
            return device == "GPU" || device.rfind("GPU.", 0) == 0;
        };
        // Keep reshaped host models as reusable templates; GPU requests live only for their stage.
        if (is_gpu(text_encode_device)) {
            m_deferred_text_encode_device = text_encode_device;
        } else {
            m_text_encoder->compile(text_encode_device, *filtered_properties);
            m_connectors->compile(text_encode_device, *filtered_properties);
            if (m_duration_head) {
                m_duration_head->compile(text_encode_device, *filtered_properties);
            }
        }
        if (is_gpu(denoise_device)) {
            m_deferred_denoise_device = denoise_device;
        } else {
            m_transformer->compile(denoise_device, *filtered_properties);
        }
        if (is_gpu(vae_device)) {
            m_deferred_vae_device = vae_device;
        } else {
            m_vae->compile(vae_device, *filtered_properties);
            m_audio_vae->compile(vae_device, *filtered_properties);
            m_vocoder->compile(vae_device, *filtered_properties);
        }
        m_deferred_properties = *filtered_properties;
        m_text_encode_device = text_encode_device;
        m_is_compiled = true;
        m_compiled_batch_size_multiplier = m_reshape_batch_size_multiplier;
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        compile(device, device, device, properties);
    }

protected:
    void replace_defaults(VideoGenerationConfig& config) const override {
        const VideoGenerationConfig& defaults = *m_default_config;
        // The default schedule only stands in when the caller chose neither a schedule nor a step count
        if (!config.sigmas.has_value() && -1 == config.num_inference_steps) {
            config.sigmas = defaults.sigmas;
        }
        if (-1 == config.height) {
            config.height = defaults.height;
        }
        if (-1 == config.width) {
            config.width = defaults.width;
        }
        if (-1 == config.num_inference_steps) {
            config.num_inference_steps = defaults.num_inference_steps;
        }
        if (-1 == config.max_sequence_length) {
            config.max_sequence_length = defaults.max_sequence_length;
        }
        if (!config.guidance_rescale.has_value()) {
            config.guidance_rescale = defaults.guidance_rescale;
        }
        if (0 == config.num_frames) {
            config.num_frames = defaults.num_frames;
        }
        if (!config.frame_rate.has_value()) {
            config.frame_rate = defaults.frame_rate;
        }
        // The LTX-2.3 guidance terms. LTX-2.0's defaults leave every one of them nullopt.
        if (!config.audio_guidance_scale.has_value()) {
            config.audio_guidance_scale = defaults.audio_guidance_scale;
        }
        if (!config.stg_scale.has_value()) {
            config.stg_scale = defaults.stg_scale;
        }
        if (!config.audio_stg_scale.has_value()) {
            config.audio_stg_scale = defaults.audio_stg_scale;
        }
        if (!config.modality_scale.has_value()) {
            config.modality_scale = defaults.modality_scale;
        }
        if (!config.audio_modality_scale.has_value()) {
            config.audio_modality_scale = defaults.audio_modality_scale;
        }
        if (!config.audio_guidance_rescale.has_value()) {
            config.audio_guidance_rescale = defaults.audio_guidance_rescale;
        }
        if (!config.spatio_temporal_guidance_blocks.has_value()) {
            config.spatio_temporal_guidance_blocks = defaults.spatio_temporal_guidance_blocks;
        }
    }

};

}  // namespace ov::genai
