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
protected:
    ZImageDenoiserLoopModule() = delete;
    ZImageDenoiserLoopModule(const IBaseModuleDesc::PTR& desc);
public:
    ~ZImageDenoiserLoopModule();

    void run() override;

    using PTR = std::shared_ptr<ZImageDenoiserLoopModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new ZImageDenoiserLoopModule(desc));
    }
    static void print_static_config();

private:
    bool initialize();
    int get_vae_scale_factor(const std::filesystem::path &model_path) const;
    std::vector<ov::Tensor> run(
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        const ImageGenerationConfig &generation_config);
    ov::Tensor prepare_latents(
        int batch_size,
        int num_channels,
        int height,
        int width,
        element::Type element_type,
        std::shared_ptr<Generator> generator);
    ov::Tensor stack(const std::vector<ov::Tensor>& tensors);
    ImageGenerationModelType m_model_type;
    TransformerConfig m_transformer_config;
    std::unique_ptr<IScheduler> m_scheduler;
    ov::InferRequest m_request;
    bool m_is_multi_prompts {false};
    int m_vae_scale_factor {8};
};

}
}
}