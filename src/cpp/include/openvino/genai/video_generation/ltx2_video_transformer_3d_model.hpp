// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS LTX2VideoTransformer3DModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 128;
        size_t audio_in_channels = 128;
        size_t patch_size = 1;
        size_t patch_size_t = 1;
        std::vector<int64_t> vae_scale_factors = {8, 32, 32};
        int64_t audio_scale_factor = 4;
        int64_t causal_offset = 1;
        int64_t audio_sampling_rate = 16000;
        int64_t audio_hop_length = 160;
        /// Number of transformer blocks; sizes 'stg_perturbation_mask' when the IR dimension is dynamic
        size_t num_layers = 0;
        /// True on LTX-2.3, absent on LTX-2.0. Selects the LTX-2.3 generation defaults.
        bool perturbed_attn = false;

        // Rotary positional embedding (RoPE) parameters, used when the export takes RoPE as
        // precomputed (cos, sin) inputs. Defaults follow diffusers' LTX2VideoTransformer3DModel.
        size_t num_attention_heads = 32;
        size_t attention_head_dim = 128;
        size_t audio_num_attention_heads = 32;
        size_t audio_attention_head_dim = 64;
        size_t audio_cross_attention_dim = 2048;
        int64_t pos_embed_max_pos = 20;
        int64_t audio_pos_embed_max_pos = 20;
        int64_t base_height = 2048;
        int64_t base_width = 2048;
        double rope_theta = 10000.0;
        bool rope_double_precision = true;
        /// "split" (LTX-2.3) or "interleaved"
        std::string rope_type = "interleaved";

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir);

    LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir,
                                const std::string& device,
                                const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir,
                                const std::string& device,
                                Properties&&... properties)
        : LTX2VideoTransformer3DModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    LTX2VideoTransformer3DModel(const LTX2VideoTransformer3DModel&);

    LTX2VideoTransformer3DModel clone();

    const Config& get_config() const;

    LTX2VideoTransformer3DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<LTX2VideoTransformer3DModel&, Properties...> compile(const std::string& device,
                                                                                        Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_hidden_states(const std::string& tensor_name, const ov::Tensor& tensor);

    /// @brief Whether an exported input exists, before or after compilation.
    bool has_input(const std::string& name);

    /// @brief Computes the RoPE (cos, sin) transformer inputs on the host from the positions and binds them. The
    /// rotary angles (up to ~1.6e4 rad) cannot be resolved in f16. Legacy exports receive coordinates
    /// directly and retain their internal RoPE computation.
    /// @param video_coords [B, 3, video_sequence_length, 2] patch boundaries (see prepare_video_coords)
    /// @param audio_coords [B, 1, audio_num_frames, 2] patch boundaries (see prepare_audio_coords)
    void set_rope_coords(const ov::Tensor& video_coords, const ov::Tensor& audio_coords);

    /// @brief Builds the 'timestep' input matching the compiled model and runs joint video + audio denoising.
    /// Rank-1 exports receive [B]. Rank-2 exports receive [B, S] for legacy internal RoPE, or [B, 1] for
    /// external RoPE, where the model broadcasts the shared timestep over all tokens.
    /// @param isolate_modalities Turns off audio-to-video and video-to-audio cross attention for this pass
    /// (LTX-2.3 modality isolation guidance). Requires a 'cross_modality_gate' input.
    /// @param stg_blocks Transformer block indices to perturb for this pass (LTX-2.3 Spatio-Temporal
    /// Guidance). Indices outside the model's block range are ignored. Requires a 'stg_perturbation_mask'
    /// input. Both arguments are no-ops at their defaults, which reproduce LTX-2.0 behaviour.
    /// @returns A pair of video and audio velocity predictions
    std::pair<ov::Tensor, ov::Tensor> infer(const ov::Tensor& video_latent,
                                            const ov::Tensor& audio_latent,
                                            float timestep,
                                            bool isolate_modalities = false,
                                            const std::vector<int64_t>& stg_blocks = {});

    LTX2VideoTransformer3DModel& reshape(int64_t batch_size,
                                         int64_t num_frames,
                                         int64_t height,
                                         int64_t width,
                                         int64_t audio_num_frames);

    size_t get_expected_batch_size() const;

    /// @brief Rank of the compiled model's 'timestep' input: 1 for legacy [B] exports,
    /// 2 for [B, S] per-token conditioning.
    size_t get_timestep_rank();

    /// @brief Whether the compiled model can isolate the audio and video modalities (LTX-2.3).
    bool has_cross_modality_gate() const;

    /// @brief Whether the compiled model supports Spatio-Temporal Guidance (LTX-2.3).
    bool has_stg_perturbation_mask() const;

private:
    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    size_t m_expected_batch_size = 0;
    size_t m_timestep_rank = 0;
    bool m_has_audio_timestep = false;
    bool m_has_cross_modality_gate = false;
    bool m_has_stg_perturbation_mask = false;
    bool m_external_rope = false;
    size_t m_num_stg_blocks = 0;
};

}  // namespace ov::genai
