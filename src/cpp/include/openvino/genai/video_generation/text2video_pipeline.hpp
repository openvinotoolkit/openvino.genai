// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
#include "debug_utils.hpp"

// TODO: support video2video, inpainting?
// TODO: decode, perf metrics, set_scheduler, set/get_generation_config, reshape, compile, clone()
// TODO: image->video
// TODO: LoRA?
// TODO: test multiple videos per prompt
// TODO: test with different config values
// TODO: throw in num_frames isn't devisable by 8 + 1
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
namespace utils {
ov::Core singleton_core() {
    static ov::Core core;
    return core;
}
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
    ov::InferRequest m_ireq;
    LTXVideoTransformer3DModel(const std::filesystem::path& dir, const std::string& device, const ov::AnyMap& properties)
    : m_ireq{::utils::singleton_core().compile_model(dir / "openvino_model.xml", device, properties).create_infer_request()} {}
// inputs[
// <ConstOutput: names[hidden_states] shape[?,?,128] type: f32>,
// <ConstOutput: names[encoder_hidden_states] shape[?,?,4096] type: f32>,
// <ConstOutput: names[timestep] shape[?] type: f32>,
// <ConstOutput: names[encoder_attention_mask] shape[?,?] type: i64>,
// <ConstOutput: names[num_frames] shape[] type: i64>,
// <ConstOutput: names[height] shape[] type: i64>,
// <ConstOutput: names[width] shape[] type: i64>,
// <ConstOutput: names[rope_interpolation_scale] shape[3] type: f32>
// ]
// outputs[
// <ConstOutput: names[out_sample] shape[?,?,128] type: f32>
// ]>

// Flux:
// inputs[
// <ConstOutput: names[hidden_states] shape[?,?,64] type: f32>,
// <ConstOutput: names[encoder_hidden_states] shape[?,?,4096] type: f32>,
// <ConstOutput: names[pooled_projections] shape[?,768] type: f32>,
// <ConstOutput: names[timestep] shape[?] type: f32>,
// <ConstOutput: names[img_ids] shape[?,3] type: f32>,
// <ConstOutput: names[txt_ids] shape[?,3] type: f32>,
// <ConstOutput: names[guidance] shape[?] type: f32>
// ]
// outputs[
// <ConstOutput: names[out_sample] shape[?,?,64] type: f32>
// ]>
    Config get_config() const {return Config{};}
    void set_hidden_states(const std::string& tensor_name, const ov::Tensor& encoder_hidden_states) {
        OPENVINO_ASSERT(m_ireq, "Transformer model must be compiled first");
        m_ireq.set_tensor(tensor_name, encoder_hidden_states);
    }
    ov::Tensor infer(const ov::Tensor& latent_model_input, const ov::Tensor& timestep) {
        OPENVINO_ASSERT(m_ireq, "Transformer model must be compiled first. Cannot infer non-compiled model");

        m_ireq.set_tensor("hidden_states", latent_model_input);
        m_ireq.set_tensor("timestep", timestep);
        m_ireq.infer();

        return m_ireq.get_output_tensor();
    }
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

}  // namespace ov::genai

namespace ov::genai {
class OPENVINO_GENAI_EXPORTS Text2VideoPipeline {
public:
    // static Text2VideoPipeline ltx_video();
    Text2VideoPipeline(
        const std::filesystem::path& models_dir,
        const std::string& device,
        const AnyMap& properties = {}
    );
    /**
     * Generates image(s) based on prompt and other image generation parameters
     * @param positive_prompt Prompt to generate image(s) from
     * @param negative_prompt
     * @param properties Image generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns A tensor which has dimensions [num_images_per_prompt, height, width, 3]
     */
    ov::Tensor generate(
        const std::string& positive_prompt,
        const std::string& negative_prompt = "",
        const ov::AnyMap& properties = {}
    );

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
        const std::string& positive_prompt,
        const std::string& negative_prompt,
        Properties&&... properties
    ) {
        return generate(positive_prompt, negative_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    const VideoGenerationConfig& get_generation_config() const;
    void set_generation_config(const VideoGenerationConfig& generation_config);

    ~Text2VideoPipeline();

    class LTXPipeline;  // TODO: private
    std::unique_ptr<LTXPipeline> m_impl;  // TODO: better pimpl
private:
    explicit Text2VideoPipeline(std::unique_ptr<LTXPipeline>&& impl);
};
}  // namespace ov::genai
