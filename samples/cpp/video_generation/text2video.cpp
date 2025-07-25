// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <random>
#include <filesystem>

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/op/transpose.hpp"

#include "openvino/genai/image_generation/scheduler.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"

#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/unet2d_condition_model.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"
#include "openvino/genai/image_generation/flux_transformer_2d_model.hpp"

#include "openvino/genai/image_generation/image2image_pipeline.hpp"

#include "image_generation/stable_diffusion_pipeline.hpp"
#include "image_generation/stable_diffusion_xl_pipeline.hpp"
#include "image_generation/stable_diffusion_3_pipeline.hpp"
#include "image_generation/flux_pipeline.hpp"

#include "utils.hpp"
#include "progress_bar.hpp"
#include "debug_utils.hpp"

// TODO: support video2video, inpainting?
// TODO: decode, perf metrics, set_scheduler, set/get_generation_config, reshape, compile, clone()
// TODO: image->video
// TODO: LoRA?
// TODO: test multiple videos per prompt
// TODO: test with different config values
namespace numpy_utils {
void batch_copy(ov::Tensor src, ov::Tensor dst, size_t src_batch, size_t dst_batch, size_t batch_size) {
    const ov::Shape src_shape = src.get_shape(), dst_shape = dst.get_shape();
    ov::Coordinate src_start(src_shape.size(), 0), src_end = src_shape;
    ov::Coordinate dst_start(dst_shape.size(), 0), dst_end = dst_shape;

    src_start[0] = src_batch;
    src_end[0] = src_batch + batch_size;

    dst_start[0] = dst_batch;
    dst_end[0] = dst_batch + batch_size;

    ov::Tensor(src, src_start, src_end).copy_to(ov::Tensor(dst, dst_start, dst_end));
}

ov::Tensor repeat(const ov::Tensor input, const size_t n_times) {
    if (n_times == 1)
        return input;

    ov::Shape input_shape = input.get_shape(), repeated_shape = input_shape;
    repeated_shape[0] *= n_times;

    ov::Tensor tensor_repeated(input.get_element_type(), repeated_shape);
    for (size_t n = 0; n < n_times; ++n) {
        batch_copy(input, tensor_repeated, 0, n, input_shape[0]);
    }
    return tensor_repeated;
}
}  // namespace numpy_utils

namespace ov::genai {
struct VideoGenerationConfig : public ImageGenerationConfig {
    double guidance_rescale = 0.0;
    size_t num_frames = 161;
};
}  // namespace ov::genai
namespace {
ov::Tensor pack_latents(ov::Tensor& latents, size_t patch_size, size_t patch_size_t) {
    // Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
    // The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    // [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    // dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape.at(0), num_channels = latents_shape.at(1), num_frames = latents_shape.at(2), height = latents_shape.at(3), width = latents_shape.at(4);
    size_t post_patch_num_frames = num_frames / patch_size_t;
    size_t post_patch_height = height / patch_size;
    size_t post_patch_width = width / patch_size;
    latents.set_shape({batch_size, num_channels, post_patch_num_frames, patch_size_t, post_patch_height, patch_size, post_patch_width, patch_size});
    // latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    std::vector<int64_t> order = {0, 2, 4, 6, 1, 3, 5, 7};  // Permutation order
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
    ov::Tensor ref_noise = from_npy("noise.npy");
    OPENVINO_ASSERT(noise.get_shape() == ref_noise.get_shape());
    noise = ref_noise;
    return pack_latents(noise, transformer_spatial_patch_size, transformer_temporal_patch_size);
}
}  // anonymous namespace
namespace ov::genai {
struct LTXVideoTransformer3DModel {
    struct OPENVINO_GENAI_EXPORTS Config {  // TODO: video fields instead
        size_t in_channels = 128;  // Comes from transformer/config.json  // TODO: Could I just use model shape?
        bool guidance_embeds = false;
        size_t m_default_sample_size = 128;
        size_t patch_size = 1;  // TODO: read from transformer/config.json
        size_t patch_size_t = 1;  // TODO: read from transformer/config.json
    };
    LTXVideoTransformer3DModel(const std::filesystem::path& dir, const std::string& device, const ov::AnyMap& properties) {}
    Config get_config() const {return Config{};}
};
struct AutoencoderKLLTXVideo {
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 3;
        size_t latent_channels = 4;
        size_t out_channels = 3;
        float scaling_factor = 1.0f;
        float shift_factor = 0.0f;
        std::vector<size_t> block_out_channels = { 64 };
        size_t patch_size = 4;  // TODO: read from vae_decoder/config.json
        std::vector<bool> spatio_temporal_scaling{true, true, true, false};  // TODO: read from vae_decoder/config.json. I use it only to compute sum over it so far
        size_t patch_size_t = 1;  // TODO: read from vae_decoder/config.json
    };
    Config m_config;
    AutoencoderKLLTXVideo(const std::filesystem::path& dir, const std::string& device, const ov::AnyMap& properties) {}
    size_t get_vae_scale_factor() const {  // TODO: verify. Drop?
        return std::pow(2, m_config.block_out_channels.size() - 1);
    }
    const Config& get_config() const {return m_config;}  // TODO: are getters needed?
};
struct LTXPipeline {
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

        // // Denoising loop
        // ov::Tensor timestep(ov::element::f32, {1});
        // float* timestep_data = timestep.data<float>();

        // for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
        //     auto step_start = std::chrono::steady_clock::now();
        //     timestep_data[0] = timesteps[inference_step] / 1000.0f;

        //     auto infer_start = std::chrono::steady_clock::now();
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
        // }

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
}  // namespace ov::genai

namespace {
std::unique_ptr<ov::genai::LTXPipeline> create_LTXPipeline(const std::filesystem::path& models_dir, const std::string& device, const ov::AnyMap& properties) {
    // TODO: move to common
    const std::string class_name = get_class_name(models_dir);
    auto start_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT("LTXPipeline" == class_name);
    std::unique_ptr<ov::genai::LTXPipeline> impl = std::make_unique<ov::genai::LTXPipeline>(models_dir, device, properties);
    impl->save_load_time(start_time);
    return impl;
}
}  // anonymous namespace

namespace ov::genai {
struct Text2VideoPipeline {
    std::unique_ptr<LTXPipeline> m_impl;
    Text2VideoPipeline(const std::filesystem::path& models_dir, const std::string& device, const ov::AnyMap& properties = {}) :
        m_impl{create_LTXPipeline(models_dir, device, properties)} {}
    /**
     * Generates image(s) based on prompt and other image generation parameters
     * @param positive_prompt Prompt to generate image(s) from
     * @param negative_prompt
     * @param properties Image generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns A tensor which has dimensions [num_images_per_prompt, height, width, 3]
     */
    ov::Tensor generate(const std::string& positive_prompt, const std::string& negative_prompt = "", const ov::AnyMap& properties = {}) {
        // TODO: explicit negative_prompt arg instead of Property? What other args can be exposed that way?
        return m_impl->generate(positive_prompt, negative_prompt, properties);
    }

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
        const std::string& positive_prompt,
        const std::string& negative_prompt,
        Properties&&... properties
    ) {
        return generate(positive_prompt, negative_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }
};
}  // namespace ov::genai

int main(int32_t argc, char* argv[]) {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::string models_dir = argv[1], prompt = argv[2];
    // TODO: Test GPU, NPU, HETERO, MULTI, AUTO, different steps on different devices
    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::Text2VideoPipeline pipe(models_dir, device);
    pipe.m_impl->m_generation_config.num_frames = 1;
    ov::Tensor image = pipe.generate(
        prompt,
        "worst quality, inconsistent motion, blurry, jittery, distorted",
        ov::genai::height(512),
        ov::genai::width(704),
        ov::genai::num_inference_steps(50),
        ov::genai::num_images_per_prompt(1),
        ov::genai::callback(progress_bar)
    );
    return EXIT_SUCCESS;
// } catch (const std::exception& error) {
//     try {
//         std::cerr << error.what() << '\n';
//     } catch (const std::ios_base::failure&) {}
//     return EXIT_FAILURE;
// } catch (...) {
//     try {
//         std::cerr << "Non-exception object thrown\n";
//     } catch (const std::ios_base::failure&) {}
//     return EXIT_FAILURE;
}
