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
#include "generation_config_utils.hpp"
#include "video_generation/models/autoencoder_kl_ltx2_audio.hpp"
#include "video_generation/models/autoencoder_kl_ltx2_video.hpp"
#include "video_generation/models/gemma3_text_encoder.hpp"
#include "video_generation/models/ltx2_text_connectors.hpp"
#include "video_generation/models/ltx2_video_transformer_3d_model.hpp"
#include "video_generation/models/ltx2_vocoder.hpp"
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

    void check_inputs(const VideoGenerationConfig& generation_config) const {
        utils::validate_generation_config(generation_config);
        OPENVINO_ASSERT(generation_config.height > 0, "Height must be positive");
        OPENVINO_ASSERT(generation_config.height % 32 == 0,
                        "Height have to be divisible by 32 but got ",
                        generation_config.height);
        OPENVINO_ASSERT(generation_config.width > 0, "Width must be positive");
        OPENVINO_ASSERT(generation_config.width % 32 == 0,
                        "Width have to be divisible by 32 but got ",
                        generation_config.width);
        OPENVINO_ASSERT(generation_config.max_sequence_length <= 1024,
                        "Gemma3's 'max_sequence_length' must be less or equal to 1024");
        OPENVINO_ASSERT(!generation_config.taylorseer_config,
                        "TaylorSeer is not supported for LTX2 pipelines");
        OPENVINO_ASSERT(!generation_config.adapters, "LoRA adapters are not supported for LTX2 pipelines");
    }

    size_t audio_num_frames_for(const VideoGenerationConfig& generation_config) const {
        const float frame_rate =
            generation_config.frame_rate.value_or(LTX2_DEFAULT_CONFIG.frame_rate.value());
        const double audio_latents_per_second = static_cast<double>(m_audio_vae->get_config().sample_rate) /
                                                m_audio_vae->get_config().mel_hop_length /
                                                m_audio_vae->get_config().temporal_compression_ratio;
        const double duration_s = static_cast<double>(generation_config.num_frames) / frame_rate;
        return static_cast<size_t>(std::lround(duration_s * audio_latents_per_second));
    }

    void compute_hidden_states(const std::string& positive_prompt,
                               const std::string& negative_prompt,
                               const VideoGenerationConfig& generation_config,
                               bool do_classifier_free_guidance) {
        auto infer_start = std::chrono::steady_clock::now();
        Gemma3TextEncoder::EncodeResult encoded = m_text_encoder->infer(positive_prompt,
                                                                       negative_prompt,
                                                                       do_classifier_free_guidance,
                                                                       generation_config.max_sequence_length);
        auto infer_end = std::chrono::steady_clock::now();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = Ms{infer_end - infer_start}.count();

        ov::Tensor prompt_embeds = repeat_per_video(encoded.prompt_embeds, generation_config.num_videos_per_prompt);
        ov::Tensor prompt_attention_mask =
            repeat_per_video(encoded.attention_mask, generation_config.num_videos_per_prompt);

        infer_start = std::chrono::steady_clock::now();
        LTX2TextConnectors::Output connected = m_connectors->infer(prompt_embeds, prompt_attention_mask);
        infer_end = std::chrono::steady_clock::now();
        m_perf_metrics.encoder_inference_duration["connectors"] = Ms{infer_end - infer_start}.count();

        m_transformer->set_hidden_states("encoder_hidden_states", connected.video_text_embedding);
        m_transformer->set_hidden_states("audio_encoder_hidden_states", connected.audio_text_embedding);
        m_transformer->set_hidden_states("encoder_attention_mask", connected.connector_attention_mask);
        m_transformer->set_hidden_states("audio_encoder_attention_mask", connected.connector_attention_mask);
    }

    void set_micro_conditions(size_t audio_num_frames, float frame_rate) {
        auto make_i64_scalar = [](int64_t value) {
            ov::Tensor scalar(ov::element::i64, {});
            scalar.data<int64_t>()[0] = value;
            return scalar;
        };
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

    const std::vector<float>& audio_latents_mean() const {
        return m_audio_vae->get_config().latents_mean_data.empty() ? m_vae->get_config().audio_latents_mean_data
                                                                   : m_audio_vae->get_config().latents_mean_data;
    }

    const std::vector<float>& audio_latents_std() const {
        return m_audio_vae->get_config().latents_std_data.empty() ? m_vae->get_config().audio_latents_std_data
                                                                  : m_audio_vae->get_config().latents_std_data;
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
        const std::vector<float>& mean = audio_latents_mean();
        const std::vector<float>& std_data = audio_latents_std();
        if (mean.empty() && std_data.empty()) {
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

        m_video_scheduler = video_generation_utils::cast_scheduler(
            Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));
        m_audio_scheduler = video_generation_utils::cast_scheduler(
            Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

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
        if (vocoder == "LTX2Vocoder") {
            m_vocoder = std::make_shared<LTX2Vocoder>(root_dir / "vocoder");
        } else {
            OPENVINO_THROW("Unsupported '", vocoder, "' vocoder type");
        }

        m_generation_config = LTX2_DEFAULT_CONFIG;
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
        cloned->m_video_scheduler = video_generation_utils::cast_scheduler(
            Scheduler::from_config(m_models_dir / "scheduler/scheduler_config.json"));
        cloned->m_audio_scheduler = video_generation_utils::cast_scheduler(
            Scheduler::from_config(m_models_dir / "scheduler/scheduler_config.json"));
        cloned->m_text_encoder = m_text_encoder->clone();
        cloned->m_connectors = m_connectors->clone();
        cloned->m_transformer = m_transformer->clone();
        cloned->m_vae = m_vae->clone();
        cloned->m_audio_vae = m_audio_vae->clone();
        cloned->m_vocoder = m_vocoder->clone();
        return cloned;
    }

    size_t get_audio_sample_rate() const override {
        return m_vocoder->get_config().output_sampling_rate;
    }

    bool do_classifier_free_guidance(float guidance_scale) const {
        return guidance_scale > 1.0;
    }

    void rebuild_models() {
        m_text_encoder = std::make_shared<Gemma3TextEncoder>(m_models_dir / "text_encoder");
        m_connectors = std::make_shared<LTX2TextConnectors>(m_models_dir / "connectors");
        m_transformer = std::make_shared<LTX2VideoTransformer3DModel>(m_models_dir / "transformer");
        m_vae = std::make_shared<AutoencoderKLLTX2Video>(m_models_dir / "vae_decoder");
        m_audio_vae = std::make_shared<AutoencoderKLLTX2Audio>(m_models_dir / "audio_vae_decoder");
        m_vocoder = std::make_shared<LTX2Vocoder>(m_models_dir / "vocoder");
    }

    void reshape_models(const VideoGenerationConfig& generation_config, size_t batch_size_multiplier) {
        m_reshape_batch_size_multiplier = batch_size_multiplier;
        const size_t audio_num_frames = audio_num_frames_for(generation_config);
        m_text_encoder->reshape(batch_size_multiplier, generation_config.max_sequence_length);
        m_connectors->reshape(generation_config.num_videos_per_prompt * batch_size_multiplier);
        m_transformer->reshape(generation_config.num_videos_per_prompt * batch_size_multiplier,
                               generation_config.num_frames,
                               generation_config.height,
                               generation_config.width,
                               audio_num_frames);
        m_vae->reshape(generation_config.num_videos_per_prompt,
                       generation_config.num_frames,
                       generation_config.height,
                       generation_config.width);
        m_audio_vae->reshape(generation_config.num_videos_per_prompt, audio_num_frames);
        m_vocoder->reshape(generation_config.num_videos_per_prompt);
    }

    void reconfigure_for_guidance_scale(const VideoGenerationConfig& generation_config, size_t batch_size_multiplier) {
        rebuild_models();
        reshape_models(generation_config, batch_size_multiplier);
        if (m_is_compiled) {
            compile(m_text_encode_device, m_denoise_device, m_vae_device, m_compile_properties);
        }
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

        check_inputs(merged_generation_config);
        OPENVINO_ASSERT(merged_generation_config.generator, "Generator must not be null");

        const float audio_guidance_scale =
            merged_generation_config.audio_guidance_scale.value_or(merged_generation_config.guidance_scale);
        const float guidance_rescale = *merged_generation_config.guidance_rescale;

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        const auto& transformer_config = m_transformer->get_config();
        const size_t num_videos_per_prompt = merged_generation_config.num_videos_per_prompt;
        const float frame_rate =
            merged_generation_config.frame_rate.value_or(LTX2_DEFAULT_CONFIG.frame_rate.value());

        m_latent_num_frames =
            (merged_generation_config.num_frames - 1) / m_vae->get_config().temporal_compression_ratio + 1;
        m_latent_height = merged_generation_config.height / m_vae->get_config().spatial_compression_ratio;
        m_latent_width = merged_generation_config.width / m_vae->get_config().spatial_compression_ratio;
        const size_t video_sequence_length = m_latent_num_frames * m_latent_height * m_latent_width;

        const size_t audio_num_frames = audio_num_frames_for(merged_generation_config);
        const size_t latent_mel_bins =
            m_audio_vae->get_config().mel_bins / m_audio_vae->get_config().mel_compression_ratio;

        compute_hidden_states(positive_prompt,
                              merged_generation_config.negative_prompt.value_or(""),
                              merged_generation_config,
                              use_classifier_free_guidance);
        set_micro_conditions(audio_num_frames, frame_rate);

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

        // mu is constant: the reference evaluates calculate_shift() at max_image_seq_len, which
        // resolves to max_shift regardless of resolution
        const double mu = m_video_scheduler->calculate_shift(m_scheduler_config.max_image_seq_len);
        m_video_scheduler->set_timesteps_with_mu(mu, merged_generation_config.num_inference_steps, 1.0f);
        // Separate scheduler instance for audio: step() tracks per-modality state
        m_audio_scheduler->set_timesteps_with_mu(mu, merged_generation_config.num_inference_steps, 1.0f);
        std::vector<float> timesteps = m_video_scheduler->get_float_timesteps();

        const size_t total_batch_size = num_videos_per_prompt * batch_size_multiplier;
        m_transformer->set_hidden_states("video_coords", prepare_video_coords(total_batch_size, frame_rate));
        m_transformer->set_hidden_states("audio_coords", prepare_audio_coords(total_batch_size, audio_num_frames));

        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);
        ov::Shape audio_shape_cfg = audio_latent.get_shape();
        audio_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor audio_cfg(ov::element::f32, audio_shape_cfg);

        // x0-space classifier-free guidance (see LTX2Pipeline denoising loop in diffusers):
        // x0 = sample - v * sigma; guided = x0_pos + (gs - 1) * (x0_pos - x0_neg); v = (sample - guided) / sigma
        ov::Tensor x0_pos(ov::element::f32, {}), guided(ov::element::f32, {});
        auto guided_velocity = [&](const ov::Tensor& pred, const ov::Tensor& sample, float gs, float sigma) {
            ov::Shape guided_shape = pred.get_shape();
            guided_shape[0] /= batch_size_multiplier;
            x0_pos.set_shape(guided_shape);
            guided.set_shape(guided_shape);

            const size_t elems = x0_pos.get_size();
            const float* v_uncond = pred.data<const float>();
            const float* v_cond = v_uncond + elems;
            const float* sample_data = sample.data<const float>();
            float* x0_pos_data = x0_pos.data<float>();
            float* guided_data = guided.data<float>();

            for (size_t i = 0; i < elems; ++i) {
                x0_pos_data[i] = sample_data[i] - v_cond[i] * sigma;
                guided_data[i] = x0_pos_data[i] + (gs - 1.0f) * sigma * (v_uncond[i] - v_cond[i]);
            }

            if (guidance_rescale > 0.0f) {
                video_generation_utils::rescale_noise_cfg(guided_data,
                                                          x0_pos_data,
                                                          guided_shape[0],
                                                          elems / guided_shape[0],
                                                          guidance_rescale);
            }

            ov::Tensor velocity(ov::element::f32, guided_shape);
            float* velocity_data = velocity.data<float>();
            for (size_t i = 0; i < elems; ++i) {
                velocity_data[i] = (sample_data[i] - guided_data[i]) / sigma;
            }
            return velocity;
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

            ov::Tensor video_velocity = batch_size_multiplier > 1
                ? guided_velocity(noise_pred_video, latent, merged_generation_config.guidance_scale, sigma)
                : noise_pred_video;
            ov::Tensor audio_velocity = batch_size_multiplier > 1
                ? guided_velocity(noise_pred_audio, audio_latent, audio_guidance_scale, sigma)
                : noise_pred_audio;

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
                return {ov::Tensor(ov::element::u8, {}), m_perf_metrics, ov::Tensor()};
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

        return VideoGenerationResult{video, m_perf_metrics, audio};
    }

    VideoGenerationResult decode(const ov::Tensor& latent) override {
        ov::Tensor postprocessed = postprocess_latents(latent);

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = m_vae->decode(postprocessed);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();

        return VideoGenerationResult{video, m_perf_metrics, ov::Tensor()};
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
        const size_t batch_size_multiplier = do_classifier_free_guidance(guidance_scale) ? 2 : 1;
        reshape_models(reshaped_config, batch_size_multiplier);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        m_text_encoder->compile(text_encode_device, properties);
        m_connectors->compile(text_encode_device, properties);
        m_transformer->compile(denoise_device, properties);
        m_vae->compile(vae_device, properties);
        m_audio_vae->compile(vae_device, properties);
        m_vocoder->compile(vae_device, properties);
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
        if (-1 == config.height) {
            config.height = LTX2_DEFAULT_CONFIG.height;
        }
        if (-1 == config.width) {
            config.width = LTX2_DEFAULT_CONFIG.width;
        }
        if (-1 == config.num_inference_steps) {
            config.num_inference_steps = LTX2_DEFAULT_CONFIG.num_inference_steps;
        }
        if (-1 == config.max_sequence_length) {
            config.max_sequence_length = LTX2_DEFAULT_CONFIG.max_sequence_length;
        }
        if (!config.guidance_rescale.has_value()) {
            config.guidance_rescale = LTX2_DEFAULT_CONFIG.guidance_rescale;
        }
        if (0 == config.num_frames) {
            config.num_frames = LTX2_DEFAULT_CONFIG.num_frames;
        }
        if (!config.frame_rate.has_value()) {
            config.frame_rate = LTX2_DEFAULT_CONFIG.frame_rate;
        }
    }

private:
    void check_video_size(const int height, const int width) const {
        OPENVINO_ASSERT(height > 0, "Height must be positive");
        OPENVINO_ASSERT(height % 32 == 0, "Height have to be divisible by 32 but got ", height);
        OPENVINO_ASSERT(width > 0, "Width must be positive");
        OPENVINO_ASSERT(width % 32 == 0, "Width have to be divisible by 32 but got ", width);
    }
};

}  // namespace ov::genai
