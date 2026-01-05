// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_zimage_denoiser_loop.hpp"
#include "module_genai/transformer_config.hpp"
#include "utils.hpp"
#include "image_generation/schedulers/flow_match_euler_discrete.hpp"
#include "json_utils.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include <fstream>

namespace ov {
namespace genai {
namespace module {

void ZImageDenoiserLoopModule::print_static_config() {
    std::cout << R"(
  transformer:
    type: "TransformerModule"
    device: "GPU"
    inputs:
      - name: "prompt_embed"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "prompt_embeds"
        type: "VecOVTensor"                                # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "prompt_embed_negative"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "prompt_embeds_negative"
        type: "VecOVTensor"                                # Support DataType: [VecOVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "width"                                      # [optional]
        type: "Int"                                        # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "height"                                     # [optional]
        type: "Int"                                        # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "num_inference_steps"                        # [optional]
        type: "Int"                                        # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "num_images_per_prompt"                      # [optional]
        type: "Int"                                        # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "seed"                                       # [optional]
        type: "Int"                                        # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "guidance_scale"                             # [optional]
        type: "Float"                                      # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "latents"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
    params:
      model_path: "model"
    )" << std::endl;
}

ZImageDenoiserLoopModule::ZImageDenoiserLoopModule(const IBaseModuleDesc::PTR &desc) : IBaseModule(desc) {
    m_model_type = to_image_generation_model_type(desc->model_type);
    if (m_model_type != ImageGenerationModelType::ZIMAGE) {
        GENAI_ERR("TransformerModule[" + desc->name + "]: Unsupported model type: " + desc->model_type);
        return;
    }
    if (!initialize()) {
        GENAI_ERR("Failed to initiate TransformerModule");
    }
}

ZImageDenoiserLoopModule::~ZImageDenoiserLoopModule() {

}

bool ZImageDenoiserLoopModule::initialize() {
    const auto &params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("TransformerModule[" + module_desc->name + "]: 'model_path' not found in params");
        return false;
    }

    std::filesystem::path model_path = module_desc->get_full_path(it_path->second);

    m_transformer_config = utils::from_config_json_if_exists<TransformerConfig>(
        model_path, "transformer/config.json");

    if (m_model_type == ImageGenerationModelType::ZIMAGE) {
        m_scheduler = std::make_unique<FlowMatchEulerDiscreteScheduler>(model_path / "scheduler/scheduler_config.json");
    } else {
        OPENVINO_THROW("Unsupported '", module_desc->model_type, "' Transformer model type");
    }
    auto transformer_model_path = model_path / "transformer/openvino_model.xml";
    if (!std::filesystem::exists(transformer_model_path)) {
        GENAI_ERR("TransformerModule[" + module_desc->name + "]: model file not found at " + transformer_model_path.string());
        return false;
    }
    auto model = utils::singleton_core().read_model(
        transformer_model_path);
    auto compiled_model = utils::singleton_core().compile_model(
        model,
        module_desc->device.empty() ? "CPU" : module_desc->device,
        ov::AnyMap{});
    m_request = compiled_model.create_infer_request();
    m_vae_scale_factor = get_vae_scale_factor(model_path);
    return true;
}

int ZImageDenoiserLoopModule::get_vae_scale_factor(const std::filesystem::path &model_path) const {
    std::filesystem::path vae_config_path = model_path / "vae/config.json";
    if (!std::filesystem::exists(vae_config_path)) {
        return 8;
    }
    std::ifstream vae_config(vae_config_path);
    nlohmann::json parsed = nlohmann::json::parse(vae_config);
    std::vector<int> block_out_channels;
    utils::read_json_param(parsed, "block_out_channels", block_out_channels);
    return std::pow(2, block_out_channels.size() - 1);
}

void ZImageDenoiserLoopModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    std::vector<ov::Tensor> prompt_embeds;
    std::vector<ov::Tensor> negative_prompt_embeds;
    if (this->inputs.find("prompt_embed") != this->inputs.end()) {
        prompt_embeds.push_back(this->inputs["prompt_embed"].data.as<ov::Tensor>());
        m_is_multi_prompts = false;
    } else if (this->inputs.find("prompt_embeds") != this->inputs.end()) {
        prompt_embeds = this->inputs["prompt_embeds"].data.as<std::vector<ov::Tensor>>();
        m_is_multi_prompts = true;
    } else {
        GENAI_ERR("TransformerModule[" + module_desc->name + "]: 'prompt_embed' or 'prompt_embeds' input not found");
        return;
    }
    if (this->inputs.find("prompt_embed_negative") != this->inputs.end()) {
        negative_prompt_embeds.push_back(this->inputs["prompt_embed_negative"].data.as<ov::Tensor>());
    } else if (this->inputs.find("prompt_embeds_negative") != this->inputs.end()) {
        negative_prompt_embeds = this->inputs["prompt_embeds_negative"].data.as<std::vector<ov::Tensor>>();
    } else {
        // empty negative prompt embeds
    }

    ImageGenerationConfig generation_config {};
    if (this->inputs.find("width") != this->inputs.end()) {
        generation_config.width = this->inputs["width"].data.as<int>();
    } else {
        generation_config.width = 512;
    }
    if (this->inputs.find("height") != this->inputs.end()) {
        generation_config.height = this->inputs["height"].data.as<int>();
    } else {
        generation_config.height = 512;
    }
    if (this->inputs.find("num_inference_steps") != this->inputs.end()) {
        generation_config.num_inference_steps = this->inputs["num_inference_steps"].data.as<int>();
    } else {
        generation_config.num_inference_steps = 10;
    }
    if (this->inputs.find("num_images_per_prompt") != this->inputs.end()) {
        generation_config.num_images_per_prompt = this->inputs["num_images_per_prompt"].data.as<int>();
    } else {
        generation_config.num_images_per_prompt = 1;
    }
    if (this->inputs.find("seed") != this->inputs.end()) {
        int seed = this->inputs["seed"].data.as<int>();
        generation_config.generator = std::make_shared<CppStdGenerator>(seed);
    } else {
        generation_config.generator = std::make_shared<CppStdGenerator>(42);
    }
    if (this->inputs.find("guidance_scale") != this->inputs.end()) {
        generation_config.guidance_scale = this->inputs["guidance_scale"].data.as<float>();
    } else {
        generation_config.guidance_scale = 1.0f;
    }
    this->outputs["latents"].data = run(prompt_embeds, negative_prompt_embeds, generation_config);
}

ov::Tensor ZImageDenoiserLoopModule::run(
        const std::vector<ov::Tensor>& prompt_embeds,
        const std::vector<ov::Tensor>& negative_prompt_embeds,
        const ImageGenerationConfig &generation_config) {
    // TODO: Add multi image support and negative prompt support
    size_t batch_size = prompt_embeds.size();
    int num_channels_latents = m_transformer_config.in_channels;
    ov::Tensor latents = prepare_latents(
        batch_size * generation_config.num_images_per_prompt, 
        num_channels_latents, 
        generation_config.width, 
        generation_config.height, 
        element::f32, generation_config.generator);
    std::vector<ov::Tensor> processed_embeds = prompt_embeds;
    std::vector<ov::Tensor> processed_negative_embeds = negative_prompt_embeds;
    if (generation_config.num_images_per_prompt > 1) {
        std::vector<ov::Tensor> expanded_prompt_embeds;
        for (const auto& embed : prompt_embeds) {
            for (int i = 0; i < generation_config.num_images_per_prompt; ++i) {
                expanded_prompt_embeds.push_back(embed);
            }
        }
        processed_embeds.assign(expanded_prompt_embeds.begin(), expanded_prompt_embeds.end());
        if ((generation_config.guidance_scale - 1.0f) > 1e-4) {
            std::vector<ov::Tensor> expanded_negative_embeds;
            for (const auto& embed : negative_prompt_embeds) {
                for (int i = 0; i < generation_config.num_images_per_prompt; ++i) {
                    expanded_negative_embeds.push_back(embed);
                }
            }
            processed_negative_embeds.assign(expanded_negative_embeds.begin(), expanded_negative_embeds.end());
        }
    }

    m_scheduler->set_timesteps(
        generation_config.num_inference_steps,
         generation_config.strength);
    std::vector<float> timesteps = m_scheduler->get_float_timesteps();
    ov::Tensor prompt_tensor = tensor_utils::stack(processed_embeds);
    for (size_t i = 0; i < timesteps.size(); i++) {
        timesteps[i] = (1000.0f - timesteps[i]) / 1000.0f;
    }
    for (size_t inference_step = 0; inference_step < timesteps.size(); inference_step++) {
        ov::Tensor unsqueezed_latents = tensor_utils::unsqueeze(latents, 2);
        ov::Tensor timestep(ov::element::f32, {1}, &timesteps[inference_step]);
        m_request.set_tensor("hidden_states", unsqueezed_latents);
        m_request.set_tensor("timestep", timestep);
        m_request.set_tensor("encoder_hidden_states", prompt_tensor);
        m_request.infer();
        ov::Tensor noise_pred = m_request.get_output_tensor();
        float *noise_pred_data = noise_pred.data<float>();
        for (size_t i = 0; i < noise_pred.get_size(); i++) {
            noise_pred_data[i] = -noise_pred_data[i];
        }
        auto scheduler_step_result = m_scheduler->step(
            noise_pred, latents, inference_step, generation_config.generator);
        latents = scheduler_step_result["latent"];
    }
    return latents;
}

ov::Tensor ZImageDenoiserLoopModule::prepare_latents(
        size_t batch_size,
        int num_channels,
        size_t width,
        size_t height,
        element::Type element_type,
        std::shared_ptr<Generator> generator) {
    height = 2 * (height / (m_vae_scale_factor * 2));
    width = 2 * (width / (m_vae_scale_factor * 2));

    ov::Shape latent_shape = {static_cast<size_t>(batch_size),
                                        static_cast<size_t>(num_channels),
                                        static_cast<size_t>(height),
                                        static_cast<size_t>(width)};            
    ov::Tensor latents = generator->randn_tensor(latent_shape);
    return latents;
}

}
}
}