// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include <vector>

#include "module_genai/module.hpp"
#include "module_genai/pipeline_impl.hpp"
#include "module_genai/transformer_config.hpp"

namespace ov {
namespace genai {

class IScheduler;

namespace module {

/**
 * @brief Unified VAE Decoder Tiling Module for both Image and Video Generation
 *
 * Supports:
 * - 4D image latents: [N, C, H, W] (ZImage, Stable Diffusion, etc.)
 * - 5D video latents: [B, C, T, H, W] (Wan 2.1, etc.)
 *
 * Automatically detects tensor dimensionality and applies appropriate tiling.
 */
class VAEDecoderTilingModule : public IBaseModule {
    DeclareModuleConstructor(VAEDecoderTilingModule);

public:
    // Content type for automatic detection
    enum class ContentType {
        IMAGE,  // 4D tensor [N, C, H, W]
        VIDEO   // 5D tensor [B, C, T, H, W]
    };

private:
    bool initialize();
    bool init_tile_params(const std::filesystem::path& model_path);
    bool init_tile_params_from_config();

    std::shared_ptr<ModulePipelineImpl> m_sub_pipeline_impl = nullptr;

    ov::InferRequest pp_infer_request;
    bool init_post_process();

    DiffusionModelType m_model_type;
    ContentType m_content_type = ContentType::IMAGE;

    // Tiling parameters
    float m_tile_overlap_factor = 0.25f;
    int m_sample_size = 256;
    int m_tile_latent_min_size = 32;
    int m_tile_sample_min_size = 256;
    int m_tile_sample_stride = 192;
    int m_spatial_compression_ratio = 8;
    bool m_enable_tiling = true;
    InferRequest m_slice_infer_request;

    // 4D tiling (images)
    void tile_decode_4d(const ov::Tensor& latent, ov::Tensor& output_latent);

    // 5D tiling (videos)
    void tile_decode_5d(const ov::Tensor& latent, ov::Tensor& output_latent);
    ov::Tensor slice_5d(const ov::Tensor& tensor, size_t h_start, size_t h_end, size_t w_start, size_t w_end);
    ov::Tensor pad_tile_5d(const ov::Tensor& tile, size_t target_h, size_t target_w);
    ov::Tensor crop_decoded_tile_5d(const ov::Tensor& decoded, size_t orig_h, size_t orig_w, size_t full_h, size_t full_w);
    ov::Tensor concat_tiles_5d(const std::vector<std::vector<ov::Tensor>>& tiles, size_t stride_h, size_t stride_w);

    // Common
    ov::Tensor decoder(const ov::Tensor& tile);
};

}  // namespace module
}  // namespace genai
}  // namespace ov
