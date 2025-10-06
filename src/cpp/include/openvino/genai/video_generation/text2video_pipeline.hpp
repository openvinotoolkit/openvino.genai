// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"
#include "openvino/genai/image_generation/scheduler.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "utils.hpp"

namespace ov::genai {
using VideoGenerationPerfMetrics = ImageGenerationPerfMetrics;

struct VideoGenerationConfig : public ImageGenerationConfig {
    double guidance_rescale = 0.0;
    size_t num_frames = 161;
};

// struct LTXVideoTransformer3DModel {
//     struct OPENVINO_GENAI_EXPORTS Config {  // TODO: video fields instead
//         size_t in_channels = 128;  // Comes from transformer/config.json  // TODO: Could I just use model shape?
//         bool guidance_embeds = false;
//         size_t m_default_sample_size = 128;
//         size_t patch_size = 1;  // TODO: read from transformer/config.json
//         size_t patch_size_t = 1;  // TODO: read from transformer/config.json
//     };
//     ov::InferRequest m_ireq;
//     LTXVideoTransformer3DModel(const std::filesystem::path& dir, const std::string& device, const ov::AnyMap& properties)
//     : m_ireq{utils::singleton_core().compile_model(dir / "openvino_model.xml", device, properties).create_infer_request()} {}
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
//     Config get_config() const {return Config{};}
//     void set_hidden_states(const std::string& tensor_name, const ov::Tensor& encoder_hidden_states) {
//         OPENVINO_ASSERT(m_ireq, "Transformer model must be compiled first");
//         m_ireq.set_tensor(tensor_name, encoder_hidden_states);
//     }
//     ov::Tensor infer(const ov::Tensor& latent_model_input, const ov::Tensor& timestep) {
//         OPENVINO_ASSERT(m_ireq, "Transformer model must be compiled first. Cannot infer non-compiled model");

//         m_ireq.set_tensor("hidden_states", latent_model_input);
//         m_ireq.set_tensor("timestep", timestep);
//         m_ireq.infer();

//         return m_ireq.get_output_tensor();
//     }
// };
struct AutoencoderKLLTXVideo {
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 3;
        size_t latent_channels = 4;
        size_t out_channels = 3;
        float scaling_factor = 1.0f;
        float shift_factor = 0.0f;
        std::vector<size_t> block_out_channels = { 64 };
        size_t patch_size = 4;  // TODO: read from vae_decoder/config.json
        std::vector<bool> spatio_temporal_scaling{true, true, true, false};  // TODO: read from vae_decoder/config.json. I use it only to compute sum over it so far, so it may be removed
        size_t patch_size_t = 1;  // TODO: read from vae_decoder/config.json
    };
    Config m_config;
    AutoencoderKLLTXVideo(const std::filesystem::path& dir, const std::string& device, const ov::AnyMap& properties) {}
    size_t get_vae_scale_factor() const {  // TODO: compare with reference. Drop?
        return std::pow(2, m_config.block_out_channels.size() - 1);
    }
    const Config& get_config() const {return m_config;}  // TODO: are getters needed?
};

class OPENVINO_GENAI_EXPORTS Text2VideoPipeline {
public:
    static Text2VideoPipeline ltx_video();
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

private:
    class LTXPipeline;
    std::unique_ptr<LTXPipeline> m_impl;
};
}  // namespace ov::genai
