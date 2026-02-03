// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_vae_decoder.hpp"
#include "utils.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include "module_genai/module_factory.hpp"
#include <iostream>
#include "module_genai/utils/profiler.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(VAEDecoderModule);

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

VAEDecoderModule::VAEDecoderModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    if (!initialize()) {
    	 OPENVINO_THROW("Failed to initialize VAEDecoderModule");
    }
}

VAEDecoderModule::~VAEDecoderModule() {}

bool VAEDecoderModule::initialize() {
    const auto &params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("VAEDecoderModule[" + module_desc->name + "]: 'model_path' not found in params");
        return false;
    }

    m_model_type = to_diffusion_model_type(module_desc->model_type);

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
             create_vae_decoder(model_path / "vae_decoder", device, properties);
        } else if (std::filesystem::exists(model_path / "vae")) {
             create_vae_decoder(model_path / "vae", device, properties);
        } else {
             create_vae_decoder(model_path, device, properties);
        }
    } catch (const std::exception& e) {
        GENAI_ERR("VAEDecoderModule[" + module_desc->name + "]: Failed to load AutoencoderKL: " + e.what());
        return false;
    }

    return true;
}

void VAEDecoderModule::create_vae_decoder(const std::filesystem::path &model_path,
                            const std::string &device,
                            const ov::AnyMap &properties) {
    if (m_model_type == DiffusionModelType::WAN_2_1) {
        m_vae = std::make_shared<AutoencoderKLWan>(model_path, device, properties);
    } else if (m_model_type == DiffusionModelType::ZIMAGE) {
        m_vae = std::make_shared<AutoencoderKL>(model_path, device, properties);
    } else {
        OPENVINO_THROW("VAEDecoderModule[" + module_desc->name + "]: Unsupported model type for VAE decoder");
    }
}

void VAEDecoderModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);

    OPENVINO_ASSERT(
        std::visit([](const auto& ptr) { return static_cast<bool>(ptr); }, m_vae),
        "VAEDecoderModule[VAE_decoder]: VAE model not initialized");

    prepare_inputs();
    
    if (this->inputs.find("latents") == this->inputs.end()) {
        GENAI_ERR("VAEDecoderModule[" + module_desc->name + "]: 'latents' input not found");
        return;
    }

    ov::Tensor image;
    auto& latent_data = this->inputs["latents"].data.as<ov::Tensor>();
    if (m_model_type == DiffusionModelType::ZIMAGE) {
        if (latent_data.get_shape().size() == 3u) {
            PROFILE(pm, "vae infer");
            ov::Tensor latent = tensor_utils::unsqueeze(this->inputs["latents"].data.as<ov::Tensor>(), 0);
            image = std::get<std::shared_ptr<AutoencoderKL>>(m_vae)->decode(latent);
        } else {
            PROFILE(pm, "vae infer");
            image = std::get<std::shared_ptr<AutoencoderKL>>(m_vae)->decode(latent_data);
        }
    } else if (m_model_type == DiffusionModelType::WAN_2_1) {
        PROFILE(pm, "vae infer");
        image = std::get<std::shared_ptr<AutoencoderKLWan>>(m_vae)->decode(latent_data);
    } else {
        OPENVINO_THROW("VAEDecoderModule[" + module_desc->name + "]: Unsupported model type for VAE decoder");
    }    

    this->outputs["image"].data = image;
}

} // namespace module
} // namespace genai
} // namespace ov
