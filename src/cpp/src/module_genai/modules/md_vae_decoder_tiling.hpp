// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "module_genai/module.hpp"
#include "module_genai/pipeline_impl.hpp"
#include "module_genai/transformer_config.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "circular_buffer_queue.hpp"
#include <memory>

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
    bool init_sub_pipeline(const std::string& sub_pipeline_name);

    ov::InferRequest pp_infer_request;
    bool init_post_process();
    
    ImageGenerationModelType m_model_type;
    float m_tile_overlap_factor = 0.25f;
    int m_sample_size;
    int m_tile_latent_min_size;
    int m_tile_sample_min_size;
    bool m_enable_tiling = true;

    void tile_decode(const ov::Tensor& latent, ov::Tensor& output_latent);
    ov::Tensor decoder(const ov::Tensor& tile);
    ov::Tensor blend_v(ov::Tensor& tile1, ov::Tensor& tile2, size_t blend_extent);
    ov::Tensor blend_h(ov::Tensor& tile1, ov::Tensor& tile2, size_t blend_extent);
};

}
}
}