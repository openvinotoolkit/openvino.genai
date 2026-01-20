// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>

#include "module_genai/module.hpp"
#include "module_genai/pipeline_impl.hpp"
#include "module_genai/transformer_config.hpp"

namespace ov {
namespace genai {

class IScheduler;

namespace module {

class VAEDecoderTilingModule : public IBaseModule {
    DeclareModuleConstructor(VAEDecoderTilingModule);

private:
    bool initialize();
    bool init_tile_params(const std::filesystem::path& model_path);

    std::shared_ptr<ModulePipelineImpl> m_sub_pipeline_impl = nullptr;

    ov::InferRequest pp_infer_request;
    bool init_post_process();

    ImageGenerationModelType m_model_type;
    float m_tile_overlap_factor = 0.25f;
    int m_sample_size;
    int m_tile_latent_min_size;
    int m_tile_sample_min_size;
    bool m_enable_tiling = true;
    InferRequest m_slice_infer_request;

    void tile_decode(const ov::Tensor& latent, ov::Tensor& output_latent);
    ov::Tensor decoder(const ov::Tensor& tile);
    ov::Tensor blend_v(ov::Tensor& tile1, ov::Tensor& tile2, size_t blend_extent);
    ov::Tensor blend_h(ov::Tensor& tile1, ov::Tensor& tile2, size_t blend_extent);
};

}  // namespace module
}  // namespace genai
}  // namespace ov