// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "module_genai/module.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"

namespace ov {
namespace genai {
namespace module {

class VAEDecoderModule : public IBaseModule {
    DeclareModuleConstructor(VAEDecoderModule);

private:
    bool initialize();
    std::shared_ptr<AutoencoderKL> m_vae;
};

} // namespace module
} // namespace genai
} // namespace ov
