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
    ov::Tensor run(
        ov::Tensor latents,
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        const ImageGenerationConfig &generation_config);
    ImageGenerationModelType m_model_type;
    std::shared_ptr<IScheduler> m_scheduler;
    ov::InferRequest m_request;
    bool m_is_multi_prompts {false};
    float m_cfg_truncation;
    bool m_cfg_normalization;
};

}
}
}