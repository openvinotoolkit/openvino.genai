// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include <numeric>

#include "circular_buffer_queue.hpp"
#include "module_genai/module.hpp"
#include "module_genai/transformer_config.hpp"
#include "splitted_model_infer.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "../unipc_multistep_scheduler.hpp"

namespace ov {
namespace genai {

class IScheduler;

namespace module {

class DenoiserLoopModule : public IBaseModule {
    DeclareModuleConstructor(DenoiserLoopModule);

private:
    bool initialize();
    // Image generation
    ov::Tensor run(
        ov::Tensor latents,
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        const ImageGenerationConfig &generation_config);
    // Video generation
    ov::Tensor run(
        ov::Tensor latents,
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        int num_inference_steps,
        float guidance_scale);
    DiffusionModelType m_model_type;
    std::variant<std::shared_ptr<IScheduler>, std::shared_ptr<UniPCMultistepScheduler>> m_scheduler;
    ov::InferRequest m_request;
    bool m_is_multi_prompts {false};
    float m_cfg_truncation;
    bool m_cfg_normalization;

    // Release weights need m_compiled_model
    ov::CompiledModel m_compiled_model;
    CSplittedModelInfer::PTR m_splitted_model_infer = nullptr;
};

}
}
}