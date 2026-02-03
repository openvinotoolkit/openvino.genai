// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "module_genai/module.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "autoencoder_kl_wan.hpp"
#include "module_genai/diffusion_model_type.hpp"

namespace ov {
namespace genai {
namespace module {

class VAEDecoderModule : public IBaseModule {
    DeclareModuleConstructor(VAEDecoderModule);

private:
    bool initialize();
    void create_vae_decoder(const std::filesystem::path &model_path,
                            const std::string &device,
                            const ov::AnyMap &properties);
    DiffusionModelType m_model_type;
    std::variant<std::shared_ptr<AutoencoderKL>, std::shared_ptr<AutoencoderKLWan>> m_vae;
};

} // namespace module
} // namespace genai
} // namespace ov
