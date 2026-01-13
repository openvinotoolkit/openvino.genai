// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "module_genai/transformer_config.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"

namespace ov::genai::module {

class RandomLatentImageModule : public IBaseModule {
    DeclareModuleConstructor(RandomLatentImageModule);

private:
    ov::Tensor prepare_latents(
        size_t batch_size,
        int num_channels,
        size_t width,
        size_t height,
        element::Type element_type,
        const std::shared_ptr<Generator> &generator) const;
    static int get_vae_scale_factor(const std::filesystem::path &model_path);

    TransformerConfig m_transformer_config;
    int m_vae_scale_factor {8};
};

REGISTER_MODULE_CONFIG(RandomLatentImageModule);

}
