// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "module_genai/module.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"

namespace ov {
namespace genai {
namespace module {

class VAEDecoderModule : public IBaseModule {
protected:
    VAEDecoderModule() = delete;
    VAEDecoderModule(const IBaseModuleDesc::PTR& desc);
public:
    ~VAEDecoderModule();

    void run() override;

    using PTR = std::shared_ptr<VAEDecoderModule>;
    static PTR create(const IBaseModuleDesc::PTR& desc) {
        return PTR(new VAEDecoderModule(desc));
    }

    static void print_static_config();

private:
    bool initialize();
    std::shared_ptr<AutoencoderKL> m_vae;
};

} // namespace module
} // namespace genai
} // namespace ov
