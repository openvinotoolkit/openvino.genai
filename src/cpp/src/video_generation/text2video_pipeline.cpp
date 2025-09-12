// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/text2video_pipeline.hpp"

using namespace ov::genai;
class Text2VideoPipeline::LTXPipeline {
public:
    std::chrono::steady_clock::duration m_load_time_ms{0};
    std::shared_ptr<IScheduler> m_scheduler;
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder;
    std::shared_ptr<LTXVideoTransformer3DModel> m_transformer;
    std::shared_ptr<AutoencoderKLLTXVideo> m_vae;
    VideoGenerationConfig m_generation_config;
    ImageGenerationPerfMetrics m_perf_metrics;  // TODO: can it be resused for video?
    double m_latent_timestep = -1.0;  // TODO: float?
    LTXPipeline(const std::filesystem::path& models_dir, const std::string& device, const ov::AnyMap& properties) {
        // TODO: move to common
        const std::filesystem::path model_index_path = models_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        const std::string class_name = data["_class_name"].get<std::string>();
        OPENVINO_ASSERT(class_name == "LTXPipeline");

        // TODO: initializer list
        set_scheduler(Scheduler::from_config(models_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        OPENVINO_ASSERT("T5EncoderModel" == text_encoder);
        m_t5_text_encoder = std::make_shared<T5EncoderModel>(models_dir / "text_encoder", device, properties);

        const std::string transformer = data["transformer"][1].get<std::string>();
        OPENVINO_ASSERT("LTXVideoTransformer3DModel" == transformer);
        m_transformer = std::make_shared<LTXVideoTransformer3DModel>(models_dir / "transformer", device, properties);

        const std::string vae = data["vae"][1].get<std::string>();
        OPENVINO_ASSERT("AutoencoderKLLTXVideo" == vae);
        m_vae = std::make_shared<AutoencoderKLLTXVideo>(models_dir / "vae_decoder", device, properties);

        initialize_generation_config(class_name);
    }
    void check_image_size(const int height, const int width) const {
        // TODO:
        // if height % 32 != 0 or width % 32 != 0:
        //     raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")
        OPENVINO_ASSERT(m_transformer != nullptr);
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) && (width % vae_scale_factor == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ",
                        vae_scale_factor);
    }
    void check_inputs(const ImageGenerationConfig& generation_config) const {
        check_image_size(generation_config.width, generation_config.height);

        OPENVINO_ASSERT(generation_config.max_sequence_length <= 512, "T5's 'max_sequence_length' must be less or equal to 512");

        OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used by FluxPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by FluxPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by FluxPipeline");
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by FluxPipeline");

        OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
    }
    void compute_hidden_states(const std::string& positive_prompt, const std::string& negative_prompt, const ImageGenerationConfig& generation_config) {
        // std::string prompt_2_str = generation_config.prompt_2 != std::nullopt ? *generation_config.prompt_2 : positive_prompt;
        auto infer_start = std::chrono::steady_clock::now();
        bool do_classifier_free_guidance = generation_config.guidance_scale > 1.0;
        ov::Tensor prompt_embeds = m_t5_text_encoder->infer(
            positive_prompt,
            negative_prompt,
            do_classifier_free_guidance,
            generation_config.max_sequence_length, {
                ov::genai::pad_to_max_length(true),
                ov::genai::max_length(generation_config.max_sequence_length),  // TODO: should infer() do that for everyone?
                ov::genai::add_special_tokens(true)
            }
        );
        auto infer_end = std::chrono::steady_clock::now();
        using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;
        m_perf_metrics.encoder_inference_duration["text_encoder"] = Ms{infer_end - infer_start}.count();  // TODO: explain in docstrings available metrics

        prompt_embeds = ::numpy_utils::repeat(prompt_embeds, generation_config.num_images_per_prompt);
        ov::Tensor ref_prompt_embeds = from_npy("prompt_embeds.npy");
        OPENVINO_ASSERT(max_diff(prompt_embeds, ref_prompt_embeds) == 0.0f);

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
    /**
     * Generates image(s) based on prompt and other image generation parameters
     * @param positive_prompt Prompt to generate image(s) from
     * @param negative_prompt
     * @param properties Image generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns A tensor which has dimensions [num_images_per_prompt, height, width, 3]
     */
    ov::Tensor generate(const std::string& positive_prompt, const std::string& negative_prompt, const ov::AnyMap& properties = {}) {
        // TODO: OVLTXPipeline allows prompt_embeds and prompt_attention_mask instead of prompt; Same for negative_prompt_embeds and negative_prompt_attention_mask
        // TODO: OVLTXPipeline allows batched generation with multiple prompts
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

        check_inputs(merged_generation_config);

        // set_lora_adapters(merged_generation_config.adapters);

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
        OPENVINO_ASSERT(max_diff(latents, from_npy("latents.npy")) == 0.0f);

        // Prepare timesteps
        size_t latent_num_frames = (merged_generation_config.num_frames - 1) / temporal_compression_ratio + 1;
        size_t latent_height = merged_generation_config.height / spatial_compression_ratio;  // TODO: prepare_latents() does the same
        size_t latent_width = merged_generation_config.width / spatial_compression_ratio;
        size_t video_sequence_length = latent_num_frames * latent_height * latent_width;
        m_scheduler->set_timesteps(video_sequence_length, merged_generation_config.num_inference_steps, merged_generation_config.strength);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();
        ov::Tensor ref_timesteps = from_npy("timesteps.npy");
        OPENVINO_ASSERT(max_diff(ov::Tensor{ov::element::f32, {timesteps.size()}, timesteps.data()}, ref_timesteps) < 0.0002f);

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

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
            const std::string& positive_prompt,
            const std::string& negative_prompt,
            Properties&&... properties) {
        return generate(positive_prompt, negative_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }
    void initialize_generation_config(const std::string& class_name) {
        // TODO: move to common
        OPENVINO_ASSERT(m_transformer != nullptr);
        OPENVINO_ASSERT(m_vae != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = VideoGenerationConfig();  // TODO: Would video generation be config different?

        m_generation_config.height = 512;
        m_generation_config.width = 704;

        m_generation_config.guidance_scale = 3.0f;
        m_generation_config.num_inference_steps = 50;
        m_generation_config.strength = 1.0f;
        m_generation_config.max_sequence_length = 128;
    }
    void save_load_time(std::chrono::steady_clock::time_point start_time) {
        // TODO: move to common
        auto stop_time = std::chrono::steady_clock::now();
        m_load_time_ms += stop_time - start_time;
    }
    void set_scheduler(std::shared_ptr<Scheduler> scheduler) {
        // TODO: move to common
        auto casted = std::dynamic_pointer_cast<IScheduler>(scheduler);
        OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
        m_scheduler = casted;
    }
};
namespace {
std::unique_ptr<ov::genai::Text2VideoPipeline::LTXPipeline> create_LTXPipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const ov::AnyMap& properties
) {
    // TODO: move to common
    const std::string class_name = get_class_name(models_dir);
    auto start_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT("LTXPipeline" == class_name);
    std::unique_ptr<ov::genai::Text2VideoPipeline::LTXPipeline> impl = std::make_unique<ov::genai::Text2VideoPipeline::LTXPipeline>(models_dir, device, properties);
    impl->save_load_time(start_time);
    return impl;
}
}  // anonymous namespace

namespace ov::genai {
Text2VideoPipeline::Text2VideoPipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const AnyMap& properties
) : m_impl{create_LTXPipeline(models_dir, device, properties)} {}

ov::Tensor Text2VideoPipeline::generate(
    const std::string& positive_prompt,
    const std::string& negative_prompt,
    const ov::AnyMap& properties
) {
    // TODO: explicit negative_prompt arg instead of Property? What other args can be exposed that way?
    return m_impl->generate(positive_prompt, negative_prompt, properties);
}

const VideoGenerationConfig& Text2VideoPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

void Text2VideoPipeline::set_generation_config(const VideoGenerationConfig& generation_config) {
    m_impl->m_generation_config = generation_config;
}

Text2VideoPipeline::~Text2VideoPipeline() = default;
}  // namespace ov::genai
