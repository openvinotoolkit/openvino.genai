// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "module_genai/module.hpp"
#include "module_genai/transformer_config.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "circular_buffer_queue.hpp"
#include <memory>

namespace ov {
namespace genai {

class IScheduler;

namespace module {

class ZImageDenoiserLoopModule : public IBaseModule {
    DeclareModuleConstructor(ZImageDenoiserLoopModule);

private:
    bool initialize();
    int get_vae_scale_factor(const std::filesystem::path &model_path) const;
    ov::Tensor run(
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        const ImageGenerationConfig &generation_config,
        std::optional<ov::Tensor> init_latents = std::nullopt);
    ov::Tensor prepare_latents(
        size_t batch_size,
        int num_channels,
        size_t width,
        size_t height,
        element::Type element_type,
        std::shared_ptr<Generator> generator);
    ImageGenerationModelType m_model_type;
    TransformerConfig m_transformer_config;
    std::shared_ptr<IScheduler> m_scheduler;
    ov::InferRequest m_request;
    bool m_is_multi_prompts {false};
    int m_vae_scale_factor {8};
};

}
}
}