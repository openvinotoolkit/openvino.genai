// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "image_generation/schedulers/ischeduler.hpp"
#include "image_generation/numpy_utils.hpp"
#include <openvino/op/transpose.hpp>
#include <nlohmann/json.hpp>
#include <fstream>

using namespace ov::genai;

namespace {
std::shared_ptr<IScheduler> cast_scheduler(std::shared_ptr<Scheduler> scheduler) {
    auto casted = std::dynamic_pointer_cast<IScheduler>(scheduler);
    OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
    return casted;
}

void check_inputs(const VideoGenerationConfig& generation_config, size_t vae_scale_factor) {
    OPENVINO_ASSERT(!generation_config.prompt_2.has_value(), "Prompt 2 is not used by LTXPipeline.");
    OPENVINO_ASSERT(!generation_config.prompt_3.has_value(), "Prompt 3 is not used by LTXPipeline.");
    OPENVINO_ASSERT(!generation_config.negative_prompt_2.has_value(), "Negative prompt 2 is not used by LTXPipeline.");
    OPENVINO_ASSERT(!generation_config.negative_prompt_3.has_value(), "Negative prompt 3 is not used by LTXPipeline.");
    OPENVINO_ASSERT(generation_config.max_sequence_length <= 512, "T5's 'max_sequence_length' must be less or equal to 512");
    OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
    // TODO:
    // if height % 32 != 0 or width % 32 != 0:
    //     raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")
    OPENVINO_ASSERT(
        (generation_config.height % vae_scale_factor == 0 || generation_config.height < 0)
            && (generation_config.width % vae_scale_factor == 0 || generation_config.width < 0),
        "Both 'width' and 'height' must be divisible by ",
        vae_scale_factor
    );
}

// Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
// The patch dimensions are then permuted and collapsed into the channel dimension of shape:
// [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
// dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
ov::Tensor pack_latents(ov::Tensor& latents, size_t patch_size, size_t patch_size_t) {
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape.at(0), num_channels = latents_shape.at(1), num_frames = latents_shape.at(2), height = latents_shape.at(3), width = latents_shape.at(4);
    size_t post_patch_num_frames = num_frames / patch_size_t;
    size_t post_patch_height = height / patch_size;
    size_t post_patch_width = width / patch_size;
    latents.set_shape({batch_size, num_channels, post_patch_num_frames, patch_size_t, post_patch_height, patch_size, post_patch_width, patch_size});
    std::array<int64_t, 8> order = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<ov::Tensor> outputs{ov::Tensor(ov::element::f32, {})};
    ov::op::v1::Transpose{}.evaluate(outputs, {latents, ov::Tensor(ov::element::i64, ov::Shape{order.size()}, order.data())});
    ov::Shape permuted_shape = outputs.at(0).get_shape();
    outputs.at(0).set_shape({permuted_shape.at(0), permuted_shape.at(1) * permuted_shape.at(2) * permuted_shape.at(3), permuted_shape.at(4) * permuted_shape.at(5) * permuted_shape.at(6)});
    return outputs.at(0);
}

ov::Tensor prepare_latents(const ov::genai::VideoGenerationConfig& generation_config, size_t num_channels_latents, size_t spatial_compression_ratio, size_t temporal_compression_ratio, size_t transformer_spatial_patch_size, size_t transformer_temporal_patch_size) {
    size_t height = generation_config.height / spatial_compression_ratio;
    size_t width = generation_config.width / spatial_compression_ratio;
    size_t num_frames = (generation_config.num_frames - 1) / temporal_compression_ratio + 1;
    ov::Shape shape{generation_config.num_images_per_prompt, num_channels_latents, num_frames, height, width};
    ov::Tensor noise = generation_config.generator->randn_tensor(shape);
    return pack_latents(noise, transformer_spatial_patch_size, transformer_temporal_patch_size);
}
}  // anonymous namespace

class Text2VideoPipeline::LTXPipeline {
    using Ms = std::chrono::duration<float, std::ratio<1, 1000>>;

    std::shared_ptr<IScheduler> m_scheduler;
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder;
    std::shared_ptr<LTXVideoTransformer3DModel> m_transformer;
    std::shared_ptr<AutoencoderKLLTXVideo> m_vae;
    VideoGenerationPerfMetrics m_perf_metrics;
    double m_latent_timestep = -1.0;  // TODO: float?
    float m_load_time_ms;

    void compute_hidden_states(const std::string& positive_prompt, const std::string& negative_prompt, const VideoGenerationConfig& generation_config) {
        auto infer_start = std::chrono::steady_clock::now();
        bool do_classifier_free_guidance = generation_config.guidance_scale > 1.0;
        ov::Tensor prompt_embeds = m_t5_text_encoder->infer(
            positive_prompt,
            negative_prompt,
            do_classifier_free_guidance,
            generation_config.max_sequence_length, {
                ov::genai::pad_to_max_length(true),
                ov::genai::max_length(generation_config.max_sequence_length),
                ov::genai::add_special_tokens(true)
            }
        );
        auto infer_end = std::chrono::steady_clock::now();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = Ms{infer_end - infer_start}.count();

        prompt_embeds = numpy_utils::repeat(prompt_embeds, generation_config.num_images_per_prompt);

        // text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        ov::Shape text_ids_shape = {prompt_embeds.get_shape()[1], 3};
        ov::Tensor text_ids(ov::element::f32, text_ids_shape);
        std::fill_n(text_ids.data<float>(), text_ids.get_size(), 0.0f);

        // const size_t num_channels_latents = m_transformer->get_config().in_channels / 4;
        // const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        // size_t height = generation_config.height / vae_scale_factor;
        // size_t width = generation_config.width / vae_scale_factor;

        // ov::Tensor latent_image_ids = prepare_latent_image_ids(generation_config.num_images_per_prompt, height / 2, width / 2);

        // if (m_transformer->get_config().guidance_embeds) {
        //     ov::Tensor guidance = ov::Tensor(ov::element::f32, {generation_config.num_images_per_prompt});
        //     std::fill_n(guidance.data<float>(), guidance.get_size(), static_cast<float>(generation_config.guidance_scale));
        //     m_transformer->set_hidden_states("guidance", guidance);
        // }

        // m_transformer->set_hidden_states("pooled_projections", pooled_prompt_embeds);
        // m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds);
        // m_transformer->set_hidden_states("txt_ids", text_ids);
        // m_transformer->set_hidden_states("img_ids", latent_image_ids);
    }

public:
    VideoGenerationConfig m_generation_config;

    LTXPipeline(
        const std::filesystem::path& models_dir,
        const std::string& device,
        const ov::AnyMap& properties,
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()
    ) :
            m_scheduler{cast_scheduler(Scheduler::from_config(models_dir / "scheduler/scheduler_config.json"))},
            m_t5_text_encoder{std::make_shared<T5EncoderModel>(models_dir / "text_encoder", device, properties)},
            m_transformer{std::make_shared<LTXVideoTransformer3DModel>(models_dir / "transformer", device, properties)},
            m_vae{std::make_shared<AutoencoderKLLTXVideo>(models_dir / "vae_decoder", device, properties)},
            m_generation_config{VideoGenerationConfig{ImageGenerationConfig{
                .guidance_scale = 3.0f,
                .height = 512,
                .width = 704,
                .num_inference_steps = 50,
                .max_sequence_length = 128,
                .strength = 1.0f,
            }}} {
        const std::filesystem::path model_index_path = models_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);
        OPENVINO_ASSERT("LTXPipeline" == nlohmann::json::parse(file)["_class_name"].get<std::string>());
        m_load_time_ms = Ms{std::chrono::steady_clock::now() - start_time}.count();
    }

    ov::Tensor generate(const std::string& positive_prompt, const std::string& negative_prompt, const ov::AnyMap& properties = {}) {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();
        VideoGenerationConfig merged_generation_config = m_generation_config;
        merged_generation_config.update_generation_config(properties);
        OPENVINO_ASSERT(merged_generation_config.height > 0, "Height must be positive");
        OPENVINO_ASSERT(merged_generation_config.width > 0, "Width must be positive");
        OPENVINO_ASSERT(1.0f == merged_generation_config.strength, "Strength isn't applicable. Must be set to the default 1.0");

        // Use callback if defined
        std::function<bool(size_t, size_t, ov::Tensor&)> callback;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback = callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>();
        }

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();

        check_inputs(merged_generation_config, vae_scale_factor);

        compute_hidden_states(positive_prompt, negative_prompt, merged_generation_config);

        size_t num_channels_latents = m_transformer->get_config().in_channels;
        size_t spatial_compression_ratio = m_vae->get_config().patch_size * std::pow(2, std::reduce(m_vae->get_config().spatio_temporal_scaling.begin(), m_vae->get_config().spatio_temporal_scaling.end(), 0));
        size_t temporal_compression_ratio = m_vae->get_config().patch_size_t * std::pow(2, std::reduce(m_vae->get_config().spatio_temporal_scaling.begin(), m_vae->get_config().spatio_temporal_scaling.end(), 0));
        size_t transformer_spatial_patch_size = m_transformer->get_config().patch_size;
        size_t transformer_temporal_patch_size = m_transformer->get_config().patch_size_t;

        ov::Tensor latents = prepare_latents(
            merged_generation_config,
            num_channels_latents,
            spatial_compression_ratio,
            temporal_compression_ratio,
            transformer_spatial_patch_size,
            transformer_temporal_patch_size
        );

        // Prepare timesteps
        size_t latent_num_frames = (merged_generation_config.num_frames - 1) / temporal_compression_ratio + 1;
        size_t latent_height = merged_generation_config.height / spatial_compression_ratio;  // TODO: prepare_latents() does the same
        size_t latent_width = merged_generation_config.width / spatial_compression_ratio;
        size_t video_sequence_length = latent_num_frames * latent_height * latent_width;
        m_scheduler->set_timesteps(video_sequence_length, merged_generation_config.num_inference_steps, merged_generation_config.strength);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        // Denoising loop
        // ov::Tensor timestep(ov::element::f32, {1});
        // float* timestep_data = timestep.data<float>();

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();
        //     timestep_data[0] = timesteps[inference_step] / 1000.0f;

            auto infer_start = std::chrono::steady_clock::now();
        //     ov::Tensor noise_pred_tensor = m_transformer->infer(latents, timestep);
        //     auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
        //     m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

        //     auto scheduler_step_result = m_scheduler->step(noise_pred_tensor, latents, inference_step, custom_generation_config.generator);
        //     latents = scheduler_step_result["latent"];

        //     if (m_pipeline_type == PipelineType::INPAINTING) {
        //         blend_latents(latents, image_latent, mask, noise, inference_step);
        //     }

        //     if (callback && callback(inference_step, timesteps.size(), latents)) {
        //         auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
        //         m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));

        //         auto image = ov::Tensor(ov::element::u8, {});
        //         m_perf_metrics.generate_duration =
        //             std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start)
        //                 .count();
        //         return image;
        //     }

        //     auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
        //     m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
        }

        // latents = unpack_latents(latents, custom_generation_config.height, custom_generation_config.width, vae_scale_factor);
        // const auto decode_start = std::chrono::steady_clock::now();
        // auto image = m_vae->decode(latents);
        // m_perf_metrics.vae_decoder_inference_duration =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
        //         .count();
        // m_perf_metrics.generate_duration =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        // return image;
        return {};
    }
};

Text2VideoPipeline::Text2VideoPipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const AnyMap& properties
) : m_impl{std::make_unique<ov::genai::Text2VideoPipeline::LTXPipeline>(
    models_dir, device, properties
)} {}

ov::Tensor Text2VideoPipeline::generate(
    const std::string& positive_prompt,
    const std::string& negative_prompt,
    const ov::AnyMap& properties
) {
    return m_impl->generate(positive_prompt, negative_prompt, properties);
}

const VideoGenerationConfig& Text2VideoPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

void Text2VideoPipeline::set_generation_config(const VideoGenerationConfig& generation_config) {
    m_impl->m_generation_config = generation_config;
}

Text2VideoPipeline::~Text2VideoPipeline() = default;
