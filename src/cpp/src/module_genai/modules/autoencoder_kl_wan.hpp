// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>
#include <string>
#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai::module {

class AutoencoderKLWan {
public:
    struct Config {
        size_t base_dim;
        size_t z_dim;
        size_t num_res_blocks;
        std::vector<size_t> dim_mult;
        float dropout;
        std::vector<float> latents_mean;
        std::vector<float> latents_std;
        std::vector<bool> temperal_downsample;

        explicit Config(const std::filesystem::path &config_path);
    };

    // Tiling configuration
    struct TilingConfig {
        bool enabled = false;
        int tile_sample_min_height = 256;
        int tile_sample_min_width = 256;
        int tile_sample_stride_height = 192;
        int tile_sample_stride_width = 192;
        int spatial_compression_ratio = 8;  // Wan 2.1 default: 8x spatial downsampling
    };

    AutoencoderKLWan(const std::filesystem::path &vae_decoder_path,
                      const std::string &device,
                      const ov::AnyMap &properties = {});

    AutoencoderKLWan(const AutoencoderKLWan &) = delete;
    AutoencoderKLWan& operator=(const AutoencoderKLWan &) = delete;

    // Main decode function - automatically selects tiled or non-tiled based on config
    ov::Tensor decode(ov::Tensor latents);

    // Enable spatial tiling for large resolution videos
    void enable_tiling(int tile_sample_min_height = 256,
                       int tile_sample_min_width = 256,
                       int tile_sample_stride_height = 192,
                       int tile_sample_stride_width = 192);

    void disable_tiling();

    bool is_tiling_enabled() const { return m_tiling_config.enabled; }

    // Model warmup - trigger JIT compilation with dummy input
    void warmup(size_t num_frames = 9);
    bool is_warmed_up() const { return m_warmed_up; }

private:
    Config m_config;
    TilingConfig m_tiling_config;
    ov::InferRequest m_decoder_request;
    std::shared_ptr<ov::Model> m_decoder_model;
    bool m_enable_postprocess = true;
    bool m_warmed_up = false;

    void init_prepostprocess(bool enable_postprocess = true);

    // Internal decode for a single tile (no tiling)
    ov::Tensor decode_single(ov::Tensor latents);

    // Tiled decode for large resolution - spatial tiling with full temporal
    ov::Tensor tiled_decode(ov::Tensor latents);

    // 5D tensor utilities
    ov::Tensor slice_5d(const ov::Tensor& tensor,
                        size_t h_start, size_t h_end,
                        size_t w_start, size_t w_end);
    ov::Tensor concat_tiles_5d(const std::vector<std::vector<ov::Tensor>>& tiles,
                               size_t tile_stride_h, size_t tile_stride_w);

    // Pad a 5D latent tile to fixed size to avoid dynamic shape recompilation
    ov::Tensor pad_tile_5d(const ov::Tensor& tile, size_t target_h, size_t target_w);

    // Crop decoded tile output to remove padding (handles [B, T, H, W, C] format)
    ov::Tensor crop_decoded_tile_5d(const ov::Tensor& decoded,
                                    size_t orig_latent_h, size_t orig_latent_w,
                                    size_t full_latent_h, size_t full_latent_w);
};

}
