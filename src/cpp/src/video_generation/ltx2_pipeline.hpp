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
#include "openvino/genai/video_generation/ltx2_text_connectors.hpp"
#include "openvino/genai/video_generation/ltx2_video_transformer_3d_model.hpp"
#include "openvino/genai/video_generation/ltx2_vocoder.hpp"
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

// LTX-2.3 defaults, mirroring diffusers 0.40's 'LTX2Pipeline.__call__'. Its guidance is three transformer
// passes per step: classifier-free guidance, Spatio-Temporal Guidance and modality isolation.
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

// Conditional half of a CFG-batched tensor: [2N, ...] -> [N, ...], the layout being [uncond, cond]
ov::Tensor conditional_half(const ov::Tensor& tensor, size_t num_videos_per_prompt) {
    ov::Shape shape = tensor.get_shape();
    OPENVINO_ASSERT(shape[0] == 2 * num_videos_per_prompt,
                    "Expected a CFG-batched tensor of batch ", 2 * num_videos_per_prompt, ", got ", shape[0]);
    shape[0] = num_videos_per_prompt;
    ov::Tensor half(tensor.get_element_type(), shape);
    numpy_utils::batch_copy(tensor, half, num_videos_per_prompt, 0, num_videos_per_prompt);
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
    std::shared_ptr<IScheduler> m_video_scheduler;
    std::shared_ptr<IScheduler> m_audio_scheduler;
    FlowMatchEulerDiscreteScheduler::Config m_scheduler_config;
    std::shared_ptr<Gemma3TextEncoder> m_text_encoder;
    std::shared_ptr<LTX2TextConnectors> m_connectors;
    std::shared_ptr<LTX2VideoTransformer3DModel> m_transformer;
    std::shared_ptr<AutoencoderKLLTX2Video> m_vae;
    std::shared_ptr<AutoencoderKLLTX2Audio> m_audio_vae;
    std::shared_ptr<LTX2Vocoder> m_vocoder;

    size_t m_latent_num_frames = 0;
    size_t m_latent_height = 0;
    size_t m_latent_width = 0;
    std::filesystem::path m_models_dir;

    // Points at the defaults of the loaded model version; the single source of truth for 'replace_defaults'
    const VideoGenerationConfig* m_default_config = &LTX2_DEFAULT_CONFIG;

    void check_inputs(const VideoGenerationConfig& generation_config) const {
        utils::validate_generation_config(generation_config);
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
        auto infer_start = std::chrono::steady_clock::now();
        ov::Tensor prompt_embeds = m_text_encoder->infer(positive_prompt,
                                                         negative_prompt,
                                                         do_classifier_free_guidance,
                                                         generation_config.max_sequence_length);
        auto infer_end = std::chrono::steady_clock::now();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = Ms{infer_end - infer_start}.count();

        ov::Tensor prompt_attention_mask = m_text_encoder->get_prompt_attention_mask();
        prompt_embeds = repeat_per_video(prompt_embeds, generation_config.num_videos_per_prompt);
        prompt_attention_mask = repeat_per_video(prompt_attention_mask, generation_config.num_videos_per_prompt);

        infer_start = std::chrono::steady_clock::now();
        LTX2TextConnectors::Output connected = m_connectors->infer(prompt_embeds, prompt_attention_mask);
        infer_end = std::chrono::steady_clock::now();
        m_perf_metrics.encoder_inference_duration["connectors"] = Ms{infer_end - infer_start}.count();

        return connected;
    }

    // Text conditioning and positional coords for one guidance pass. The classifier-free guidance pass runs
    // at batch 2N and binds the full tensors; LTX-2.3's extra passes run at batch N and bind the
    // conditional half, as diffusers does at 'i == 0'.
    struct PassConditioning {
        ov::Tensor video_text_embedding;
        ov::Tensor audio_text_embedding;
        ov::Tensor connector_attention_mask;
        ov::Tensor video_coords;
        ov::Tensor audio_coords;
    };

    void bind_conditioning(const PassConditioning& conditioning) {
        m_transformer->set_hidden_states("encoder_hidden_states", conditioning.video_text_embedding);
        m_transformer->set_hidden_states("audio_encoder_hidden_states", conditioning.audio_text_embedding);
        m_transformer->set_hidden_states("encoder_attention_mask", conditioning.connector_attention_mask);
        m_transformer->set_hidden_states("audio_encoder_attention_mask", conditioning.connector_attention_mask);
        m_transformer->set_hidden_states("video_coords", conditioning.video_coords);
        m_transformer->set_hidden_states("audio_coords", conditioning.audio_coords);
    }

    void set_micro_conditions(size_t audio_num_frames, float frame_rate) {
        using video_generation_utils::make_i64_scalar;
        m_transformer->set_hidden_states("num_frames", make_i64_scalar(m_latent_num_frames));
        m_transformer->set_hidden_states("height", make_i64_scalar(m_latent_height));
        m_transformer->set_hidden_states("width", make_i64_scalar(m_latent_width));
        m_transformer->set_hidden_states("audio_num_frames", make_i64_scalar(audio_num_frames));

        ov::Tensor fps(ov::element::f32, {});
        fps.data<float>()[0] = frame_rate;
        m_transformer->set_hidden_states("fps", fps);
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
        : m_scheduler_config(root_dir / "scheduler/scheduler_config.json") {
        m_models_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";

        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);

        m_video_scheduler = std::make_shared<FlowMatchEulerDiscreteScheduler>(m_scheduler_config);
        m_audio_scheduler = std::make_shared<FlowMatchEulerDiscreteScheduler>(m_scheduler_config);

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "Gemma3ForConditionalGeneration") {
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
        if (vae == "AutoencoderKLLTX2Video") {
            m_vae = std::make_shared<AutoencoderKLLTX2Video>(root_dir / "vae_decoder");
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
        // LTX-2.3's 'LTX2VocoderWithBWE' adds band-width extension (16 kHz in, 48 kHz out) inside the graph.
        // Its IR interface is identical and the output rate is read from the config, so the same class serves both.
        if (vocoder == "LTX2Vocoder" || vocoder == "LTX2VocoderWithBWE") {
            m_vocoder = std::make_shared<LTX2Vocoder>(root_dir / "vocoder");
        } else {
            OPENVINO_THROW("Unsupported '", vocoder, "' vocoder type");
        }

        // 'model_index.json' reports 'LTX2Pipeline' for both versions, so the transformer config decides:
        // 'perturbed_attn' is set on LTX-2.3 and absent on LTX-2.0
        m_default_config = m_transformer->get_config().perturbed_attn ? &LTX2_3_DEFAULT_CONFIG : &LTX2_DEFAULT_CONFIG;
        m_generation_config = *m_default_config;
        m_load_time = Ms{std::chrono::steady_clock::now() - start_time};
    }

    LTX2Pipeline(const std::filesystem::path& models_dir,
                 const std::string& device,
                 const ov::AnyMap& properties,
                 std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now())
        : LTX2Pipeline(models_dir, start_time) {
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
        return cloned;
    }

    // LTX-2.3 enable predicates, matching diffusers' LTX2Pipeline. Both terms stay nullopt on LTX-2.0.
    // Classifier-free guidance itself is decided by 'utils::requests_classifier_free_guidance', which also
    // accounts for the audio scale.
    static bool do_spatio_temporal_guidance(const VideoGenerationConfig& config) {
        return config.stg_scale.value_or(0.0f) > 0.0f || config.audio_stg_scale.value_or(0.0f) > 0.0f;
    }

    // 1.0 is the neutral modality scale, so it is also the fallback for an unset one: the term's weight is
    // 'scale - 1', which a 0.0 fallback would turn into -1 instead of switching the term off.
    static bool do_modality_isolation_guidance(const VideoGenerationConfig& config) {
        return config.modality_scale.value_or(1.0f) > 1.0f || config.audio_modality_scale.value_or(1.0f) > 1.0f;
    }

    // The extra passes run at batch N while classifier-free guidance runs at batch 2N, so the transformer's
    // batch dimension has to stay dynamic to serve both from one compiled model. Without CFG every pass is
    // already batch N and the static shape is kept.
    bool needs_dynamic_transformer_batch(const VideoGenerationConfig& config, size_t batch_size_multiplier) const {
        return batch_size_multiplier > 1 && m_transformer->get_config().perturbed_attn &&
               (do_spatio_temporal_guidance(config) || do_modality_isolation_guidance(config));
    }

    size_t get_transformer_expected_batch_size() const override {
        return m_transformer->get_expected_batch_size();
    }

    void reshape_models(const VideoGenerationConfig& generation_config, size_t batch_size_multiplier) override {
        m_reshape_batch_size_multiplier = batch_size_multiplier;
        const size_t audio_num_frames = audio_num_frames_for(generation_config);
        m_text_encoder->reshape(batch_size_multiplier, generation_config.max_sequence_length);
        m_connectors->reshape(generation_config.num_videos_per_prompt * batch_size_multiplier);
        m_transformer->reshape(generation_config.num_videos_per_prompt * batch_size_multiplier,
                               generation_config.num_frames,
                               generation_config.height,
                               generation_config.width,
                               audio_num_frames,
                               needs_dynamic_transformer_batch(generation_config, batch_size_multiplier));
        m_vae->reshape(generation_config.num_videos_per_prompt,
                       generation_config.num_frames,
                       generation_config.height,
                       generation_config.width);
        m_audio_vae->reshape(generation_config.num_videos_per_prompt, audio_num_frames);
        m_vocoder->reshape(generation_config.num_videos_per_prompt);
    }

    VideoGenerationResult generate(const std::string& positive_prompt, const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();

        VideoGenerationConfig merged_generation_config = merge_generation_config(properties);
        const float audio_guidance_scale =
            merged_generation_config.audio_guidance_scale.value_or(merged_generation_config.guidance_scale);

        // Matches diffusers: CFG is enabled when either modality requests guidance. Shared with
        // 'resolve_negative_prompt' and 'reshape()' so the three cannot disagree about whether CFG runs.
        const bool cfg_requested = utils::requests_classifier_free_guidance(merged_generation_config);
        const size_t batch_size_multiplier = resolve_batch_size_multiplier(merged_generation_config, cfg_requested);
        const bool use_classifier_free_guidance = batch_size_multiplier > 1;

        check_inputs(merged_generation_config);
        OPENVINO_ASSERT(merged_generation_config.generator, "Generator must not be null");

        const float guidance_rescale = *merged_generation_config.guidance_rescale;
        const float audio_guidance_rescale =
            merged_generation_config.audio_guidance_rescale.value_or(guidance_rescale);

        // LTX-2.3's extra guidance passes. Gated on the compiled model actually exposing the inputs, so a
        // 2.3-shaped export missing one degrades to the passes it can run instead of failing.
        const float stg_scale = merged_generation_config.stg_scale.value_or(0.0f);
        const float audio_stg_scale = merged_generation_config.audio_stg_scale.value_or(stg_scale);
        const float modality_scale = merged_generation_config.modality_scale.value_or(1.0f);
        const float audio_modality_scale = merged_generation_config.audio_modality_scale.value_or(modality_scale);
        const bool stg_requested = do_spatio_temporal_guidance(merged_generation_config);
        const bool modality_isolation_requested = do_modality_isolation_guidance(merged_generation_config);
        const bool use_spatio_temporal_guidance = m_transformer->has_stg_perturbation_mask() && stg_requested;
        const bool use_modality_isolation_guidance = m_transformer->has_cross_modality_gate() && modality_isolation_requested;
        // Degrading silently would change the algorithm without telling anyone, so say so once per call.
        // Not an assert: LTX-2.0 and 2.3 exports predating these inputs must keep working.
        if (stg_requested && !m_transformer->has_stg_perturbation_mask()) {
            GENAI_WARN("'stg_scale' / 'audio_stg_scale' request Spatio-Temporal Guidance, but this "
                       "transformer has no 'stg_perturbation_mask' input and the pass is skipped. Re-export "
                       "the model with LTX-2.3 support, or set both scales to 0 to silence this.");
        }
        if (modality_isolation_requested && !m_transformer->has_cross_modality_gate()) {
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
        bool use_extra_guidance_passes = use_spatio_temporal_guidance || use_modality_isolation_guidance;
        if (use_extra_guidance_passes && use_classifier_free_guidance &&
            m_transformer->get_expected_batch_size() > 0) {
            // The extra passes need batch N while this model was compiled for the batch 2N of CFG
            GENAI_WARN("Spatio-Temporal Guidance / modality isolation guidance requested, but this "
                       "transformer was compiled for a static CFG batch and cannot run the extra passes, "
                       "so they are skipped. To enable them, apply these scales with "
                       "'set_generation_config()' before 'reshape()', or construct the pipeline without an "
                       "explicit 'reshape()' so the batch dimension stays dynamic.");
            use_extra_guidance_passes = false;
        }

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        const auto& transformer_config = m_transformer->get_config();
        const size_t num_videos_per_prompt = merged_generation_config.num_videos_per_prompt;
        const float frame_rate =
            merged_generation_config.frame_rate.value_or(m_default_config->frame_rate.value());

        m_latent_num_frames =
            (merged_generation_config.num_frames - 1) / m_vae->get_config().temporal_compression_ratio + 1;
        m_latent_height = merged_generation_config.height / m_vae->get_config().spatial_compression_ratio;
        m_latent_width = merged_generation_config.width / m_vae->get_config().spatial_compression_ratio;

        const size_t audio_num_frames = audio_num_frames_for(merged_generation_config);
        const size_t latent_mel_bins =
            m_audio_vae->get_config().mel_bins / m_audio_vae->get_config().mel_compression_ratio;

        LTX2TextConnectors::Output connected = compute_hidden_states(positive_prompt,
                                                                     merged_generation_config.negative_prompt.value_or(""),
                                                                     merged_generation_config,
                                                                     use_classifier_free_guidance);
        set_micro_conditions(audio_num_frames, frame_rate);

        const size_t total_batch_size = num_videos_per_prompt * batch_size_multiplier;
        const PassConditioning full_conditioning{connected.video_text_embedding,
                                                 connected.audio_text_embedding,
                                                 connected.connector_attention_mask,
                                                 prepare_video_coords(total_batch_size, frame_rate),
                                                 prepare_audio_coords(total_batch_size, audio_num_frames)};
        // The extra passes are conditional-only, so they take the second half of the CFG-batched text
        // conditioning. The coords are identical across batch entries, so they are just rebuilt smaller.
        PassConditioning conditional_conditioning;
        if (use_extra_guidance_passes && use_classifier_free_guidance) {
            conditional_conditioning = {conditional_half(connected.video_text_embedding, num_videos_per_prompt),
                                        conditional_half(connected.audio_text_embedding, num_videos_per_prompt),
                                        conditional_half(connected.connector_attention_mask, num_videos_per_prompt),
                                        prepare_video_coords(num_videos_per_prompt, frame_rate),
                                        prepare_audio_coords(num_videos_per_prompt, audio_num_frames)};
        } else {
            conditional_conditioning = full_conditioning;
        }
        bind_conditioning(full_conditioning);

        ov::Shape video_noise_shape{num_videos_per_prompt,
                                    transformer_config.in_channels,
                                    m_latent_num_frames,
                                    m_latent_height,
                                    m_latent_width};
        ov::Tensor video_noise = merged_generation_config.generator->randn_tensor(video_noise_shape);
        ov::Tensor latent = video_generation_utils::pack_latents(video_noise,
                                                                 transformer_config.patch_size,
                                                                 transformer_config.patch_size_t);

        ov::Shape audio_noise_shape{num_videos_per_prompt,
                                    m_audio_vae->get_config().latent_channels,
                                    audio_num_frames,
                                    latent_mel_bins};
        ov::Tensor audio_noise = merged_generation_config.generator->randn_tensor(audio_noise_shape);
        ov::Tensor audio_latent = pack_audio_latents(audio_noise);

        // The reference evaluates calculate_shift() at the packed video sequence length, i.e. dim 1 of the
        // packed latents, so mu is resolution dependent. Audio reuses the video mu, as the reference does.
        const double mu = m_video_scheduler->calculate_shift(latent.get_shape().at(1));
        m_video_scheduler->set_timesteps_with_mu(mu, merged_generation_config.num_inference_steps, 1.0f);
        // Separate scheduler instance for audio: step() tracks per-modality state
        m_audio_scheduler->set_timesteps_with_mu(mu, merged_generation_config.num_inference_steps, 1.0f);
        std::vector<float> timesteps = m_video_scheduler->get_float_timesteps();

        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);
        ov::Shape audio_shape_cfg = audio_latent.get_shape();
        audio_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor audio_cfg(ov::element::f32, audio_shape_cfg);

        // x0-space guidance (see the LTX2Pipeline denoising loop in diffusers). With x0 = sample - v * sigma,
        // every guidance term is a difference of x0 predictions that collapses to a difference of
        // velocities: x0_cond - x0_other == sigma * (v_other - v_cond). So one accumulator serves all three
        // terms, each contributing 'scale * sigma * (v_other - v_cond)':
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

        // infer() hands back the infer request's own output tensors, which the next pass overwrites, so each
        // pass's prediction has to be copied out before the next call
        ov::Tensor video_pred(ov::element::f32, {}), audio_pred(ov::element::f32, {});
        ov::Tensor video_pred_stg(ov::element::f32, {}), audio_pred_stg(ov::element::f32, {});
        ov::Tensor video_pred_modality(ov::element::f32, {}), audio_pred_modality(ov::element::f32, {});
        auto keep = [](const ov::Tensor& source, ov::Tensor& destination) {
            destination.set_shape(source.get_shape());
            source.copy_to(destination);
        };

        auto run_extra_pass = [&](const ov::Tensor& video_sample,
                                  const ov::Tensor& audio_sample,
                                  float t,
                                  bool isolate_modalities,
                                  const std::vector<int64_t>& blocks,
                                  ov::Tensor& video_out,
                                  ov::Tensor& audio_out) {
            const auto pass_start = std::chrono::steady_clock::now();
            auto [video_prediction, audio_prediction] =
                m_transformer->infer(video_sample, audio_sample, t, isolate_modalities, blocks);
            const auto pass_duration =
                ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - pass_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(pass_duration));
            keep(video_prediction, video_out);
            keep(audio_prediction, audio_out);
        };

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();
            if (batch_size_multiplier > 1) {
                numpy_utils::batch_copy(latent, latent_cfg, 0, 0, num_videos_per_prompt);
                numpy_utils::batch_copy(latent, latent_cfg, 0, num_videos_per_prompt, num_videos_per_prompt);
                numpy_utils::batch_copy(audio_latent, audio_cfg, 0, 0, num_videos_per_prompt);
                numpy_utils::batch_copy(audio_latent, audio_cfg, 0, num_videos_per_prompt, num_videos_per_prompt);
            } else {
                latent_cfg = latent;
                audio_cfg = audio_latent;
            }

            const float t = timesteps[inference_step];
            const float sigma = t / m_scheduler_config.num_train_timesteps;

            auto infer_start = std::chrono::steady_clock::now();
            auto [noise_pred_video, noise_pred_audio] = m_transformer->infer(latent_cfg, audio_cfg, t);
            auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

            ov::Tensor video_velocity, audio_velocity;
            if (batch_size_multiplier == 1 && !use_extra_guidance_passes) {
                // Unguided: the single prediction is the velocity, no x0 round trip
                video_velocity = noise_pred_video;
                audio_velocity = noise_pred_audio;
            } else {
                // The conditional half of the first pass is the baseline every guidance term subtracts from.
                // It only has to be copied out when a later pass would overwrite the output tensors.
                if (use_extra_guidance_passes) {
                    keep(noise_pred_video, video_pred);
                    keep(noise_pred_audio, audio_pred);
                }
                const ov::Tensor& video_base = use_extra_guidance_passes ? video_pred : noise_pred_video;
                const ov::Tensor& audio_base = use_extra_guidance_passes ? audio_pred : noise_pred_audio;
                const size_t cond_offset = batch_size_multiplier > 1 ? latent.get_size() : 0;
                const size_t audio_cond_offset = batch_size_multiplier > 1 ? audio_latent.get_size() : 0;
                const float* v_cond = video_base.data<const float>() + cond_offset;
                const float* audio_v_cond = audio_base.data<const float>() + audio_cond_offset;

                std::vector<GuidanceTerm> video_terms, audio_terms;
                if (batch_size_multiplier > 1) {
                    // The unconditional half sits first in the [uncond, cond] batch layout
                    video_terms.push_back({merged_generation_config.guidance_scale - 1.0f,
                                           video_base.data<const float>()});
                    audio_terms.push_back({audio_guidance_scale - 1.0f, audio_base.data<const float>()});
                }

                if (use_extra_guidance_passes) {
                    // Both extra passes are conditional-only and run at batch N on the unduplicated latents
                    bind_conditioning(conditional_conditioning);

                    if (use_spatio_temporal_guidance) {
                        run_extra_pass(latent, audio_latent, t, /* isolate_modalities */ false, stg_blocks,
                                       video_pred_stg, audio_pred_stg);
                        video_terms.push_back({stg_scale, video_pred_stg.data<const float>()});
                        audio_terms.push_back({audio_stg_scale, audio_pred_stg.data<const float>()});
                    }

                    if (use_modality_isolation_guidance) {
                        run_extra_pass(latent, audio_latent, t, /* isolate_modalities */ true, {},
                                       video_pred_modality, audio_pred_modality);
                        // One pass feeds both modalities, but each contributes only above its own
                        // threshold. diffusers adds both terms unconditionally; it can afford to because
                        // 'audio_modality_scale or modality_scale' cannot express a scale of 0, so its
                        // weight is never negative. 'value_or' keeps an explicit 0.0, which would weigh the
                        // isolated prediction by -1 instead of disabling the term the docs say it disables.
                        if (modality_scale > 1.0f) {
                            video_terms.push_back({modality_scale - 1.0f, video_pred_modality.data<const float>()});
                        }
                        if (audio_modality_scale > 1.0f) {
                            audio_terms.push_back({audio_modality_scale - 1.0f,
                                                   audio_pred_modality.data<const float>()});
                        }
                    }

                    bind_conditioning(full_conditioning);
                }

                video_velocity = guided_velocity(v_cond, latent, video_terms, guidance_rescale, sigma,
                                                 video_velocity_buffer);
                audio_velocity = guided_velocity(audio_v_cond, audio_latent, audio_terms,
                                                 audio_guidance_rescale, sigma, audio_velocity_buffer);
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

        latent = postprocess_latents(latent);

        OPENVINO_ASSERT(!m_vae->get_config().timestep_conditioning,
                        "Parameter 'timestep_conditioning' is not currently supported by AutoencoderKLLTX2Video. Please, contact OpenVINO GenAI developers.");

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = m_vae->decode(latent);

        denormalize_audio_latents(audio_latent);
        ov::Tensor audio_latent_unpacked =
            unpack_audio_latents(audio_latent, m_audio_vae->get_config().latent_channels, latent_mel_bins);
        ov::Tensor mel_spectrogram = m_audio_vae->decode(audio_latent_unpacked);
        ov::Tensor audio = m_vocoder->infer(mel_spectrogram);

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

    VideoGenerationResult decode(const ov::Tensor& latent) override {
        ov::Tensor postprocessed = postprocess_latents(latent);

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = m_vae->decode(postprocessed);
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
        // 'reshape()' only takes the video scale, but audio can request CFG on its own - LTX-2.3 defaults
        // 'audio_guidance_scale' to 7.0 - and 'reshaped_config' carries it. Sizing on the video scale alone
        // compiled a batch-1 transformer that 'generate()' then could not use for guidance.
        const size_t batch_size_multiplier = utils::requests_classifier_free_guidance(reshaped_config) ? 2 : 1;
        reshape_models(reshaped_config, batch_size_multiplier);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        std::optional<AdapterConfig> adapters;
        auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
        OPENVINO_ASSERT(!adapters || adapters->get_adapters().empty(),
                        "LoRA adapters are not supported for LTX2 pipelines");
        m_text_encoder->compile(text_encode_device, *filtered_properties);
        m_connectors->compile(text_encode_device, *filtered_properties);
        m_transformer->compile(denoise_device, *filtered_properties);
        m_vae->compile(vae_device, *filtered_properties);
        m_audio_vae->compile(vae_device, *filtered_properties);
        m_vocoder->compile(vae_device, *filtered_properties);
        m_is_compiled = true;
        m_compiled_batch_size_multiplier = m_reshape_batch_size_multiplier;
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        compile(device, device, device, properties);
    }

protected:
    void replace_defaults(VideoGenerationConfig& config) const override {
        const VideoGenerationConfig& defaults = *m_default_config;
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
        // The LTX-2.3 guidance terms. LTX-2.0's defaults leave every one of them nullopt, which disables
        // the extra passes and makes each audio term fall back to its video counterpart in 'generate()'.
        // On LTX-2.3 they are all set, so the audio terms keep their own diffusers defaults rather than
        // inheriting a user-supplied video value — which is what diffusers' 'audio_x = audio_x or x' does.
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
