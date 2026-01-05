// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_vae_decoder.hpp"
#include "utils.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include <iostream>

namespace ov {
namespace genai {
namespace module {

void VAEDecoderModule::print_static_config() {
    std::cout << R"(
  - name: "VAE_decoder"
    type: "VAEDecoderModule"
    inputs:
      - name: "latents"
        type: "OVTensor"
    outputs:
      - name: "image"
        type: "OVTensor"
    params:
      model_path: "model"
      enable_postprocess: "bool value"    # [Optional], default true.
    )" << std::endl;
}

VAEDecoderModule::VAEDecoderModule(const IBaseModuleDesc::PTR &desc) : IBaseModule(desc) {
    if (!initialize()) {
    	 GENAI_ERR("Failed to initialize VAEDecoderModule");
    }
}

VAEDecoderModule::~VAEDecoderModule() {

}

bool VAEDecoderModule::initialize() {
    const auto &params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("VAEDecoderModule[" + module_desc->name + "]: 'model_path' not found in params");
        return false;
    }

    std::filesystem::path model_path = module_desc->get_full_path(it_path->second);
    std::string device = module_desc->device.empty() ? "CPU" : module_desc->device;

    bool enable_postprocess = true;
    if (params.find("enable_postprocess") != params.end()) {
        std::string val = module_desc->params["enable_postprocess"];
        if (val == "false" || val == "False" || val == "FALSE" || val == "0") {
            enable_postprocess = false;
        }
    }

    ov::AnyMap properties;
    properties["enable_postprocess"] = enable_postprocess;

    try {
        if (std::filesystem::exists(model_path / "vae_decoder")) {
             m_vae = std::make_shared<AutoencoderKL>(model_path / "vae_decoder", device, properties);
        } else if (std::filesystem::exists(model_path / "vae")) {
             m_vae = std::make_shared<AutoencoderKL>(model_path / "vae", device, properties);
        } else {
             m_vae = std::make_shared<AutoencoderKL>(model_path, device, properties);
        }
    } catch (const std::exception& e) {
        GENAI_ERR("VAEDecoderModule[" + module_desc->name + "]: Failed to load AutoencoderKL: " + e.what());
        return false;
    }

    return true;
}

void VAEDecoderModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);

    OPENVINO_ASSERT(m_vae, "VAEDecoderModule[VAE_decoder]: VAE model not initialized");

    prepare_inputs();
    
    if (this->inputs.find("latents") == this->inputs.end()) {
        GENAI_ERR("VAEDecoderModule[" + module_desc->name + "]: 'latents' input not found");
        return;
    }

    ov::Tensor latents = this->inputs["latents"].data.as<ov::Tensor>();
    ov::Tensor image = m_vae->decode(latents);

    this->outputs["image"].data = image;
}

} // namespace module
} // namespace genai
} // namespace ov
