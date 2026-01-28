// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_random_latent_image.hpp"
#include "utils.hpp"
#include "module_genai/module_factory.hpp"
#include <fstream>
#include "json_utils.hpp"
#include "module_genai/image_generation_model_type.hpp"
#include "module_genai/video_generation_model_type.hpp"

namespace ov::genai::module {

GENAI_REGISTER_MODULE_SAME(RandomLatentImageModule);

void RandomLatentImageModule::print_static_config() {
    std::cout << R"(
  latent_image:                       # Module Name
    type: "RandomLatentImageModule"
    description: "Create initial latent image"
    device: "CPU"
    inputs:
      - name: "width"
        type: "Int"                   # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "height"
        type: "Int"                   # Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "batch_size"
        type: "Int"                   # [Optional] Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "num_images_per_prompt"
        type: "Int"                   # [Optional] Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "seed"
        type: "Int"                   # [Optional] Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
      - name: "num_frames"
        type: "Int"                   # [Optional] Support DataType: [Int]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "latents"
        type: "OVTensor"              # Support DataType: [OVTensor]
    params:
      model_path: "model"
    )" << std::endl;
}

RandomLatentImageModule::RandomLatentImageModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    const auto &params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        OPENVINO_THROW("TransformerModule[" + module_desc->name + "]: 'model_path' not found in params");
    }
    std::filesystem::path model_path = module_desc->get_full_path(it_path->second);
    m_transformer_config = utils::from_config_json_if_exists<TransformerConfig>(
        model_path, "transformer/config.json");
    if (is_image_generation_model(module_desc->model_type)) {
        m_vae_scale_factor = get_vae_scale_factor(model_path);
    } else if (is_video_generation_model(module_desc->model_type)) {
        m_vae_scale_factor_spatial = get_vae_scale_factor_spatial(model_path);
        m_vae_scale_factor_temporal = get_vae_scale_factor_temporal(model_path);
    } else {
        OPENVINO_THROW("Unsupported model type: " + module_desc->model_type);
    }
    
}

RandomLatentImageModule::~RandomLatentImageModule() {}

void RandomLatentImageModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    int width = 0;
    int height = 0;
    int batch_size = 1;
    int num_images_per_prompt = 1;
    int seed = 42;
    int num_frames = 1;
    if (exists_input("width")) {
        width = this->inputs["width"].data.as<int>();
    } else {
        OPENVINO_THROW("The input width should not be empty");
    }
    if (exists_input("height")) {
        height = this->inputs["height"].data.as<int>();
    } else {
        OPENVINO_THROW("The input height should not be empty");
    }
    if (exists_input("batch_size")) {
        batch_size = this->inputs["batch_size"].data.as<int>();
    }
    if (exists_input("num_images_per_prompt")) {
        num_images_per_prompt = this->inputs["num_images_per_prompt"].data.as<int>();
    }
    if (exists_input("seed")) {
        seed = this->inputs["seed"].data.as<int>();
    }
    if (exists_input("num_frames")) {
        num_frames = this->inputs["num_frames"].data.as<int>();
    } else if (is_video_generation_model(module_desc->model_type)) {
        OPENVINO_THROW("The input num_frames should not be empty for video generation model");
    }

    ov::Tensor latents = prepare_latents(
        batch_size * num_images_per_prompt,
        m_transformer_config.in_channels,
        width,
        height,
        num_frames,
        element::f32,
        std::make_shared<CppStdGenerator>(seed));
    outputs["latents"].data = latents;
}

ov::Tensor RandomLatentImageModule::prepare_latents(
        size_t batch_size,
        int num_channels,
        size_t width,
        size_t height,
        size_t num_frames,
        element::Type element_type,
        const std::shared_ptr<Generator> &generator) const {
    ov::Shape latent_shape;
    if (is_image_generation_model(module_desc->model_type)) {
        height = 2 * (height / (m_vae_scale_factor * 2));
        width = 2 * (width / (m_vae_scale_factor * 2));

        latent_shape = {static_cast<size_t>(batch_size),
                        static_cast<size_t>(num_channels),
                        static_cast<size_t>(height),
                        static_cast<size_t>(width)};
    } else if (is_video_generation_model(module_desc->model_type)) {
        size_t num_latent_frames = (num_frames - 1) / m_vae_scale_factor_temporal + 1;
        width = width / m_vae_scale_factor_spatial;
        height = height / m_vae_scale_factor_spatial;
        latent_shape = {static_cast<size_t>(batch_size),
                        static_cast<size_t>(num_channels),
                        num_latent_frames,
                        static_cast<size_t>(height),
                        static_cast<size_t>(width)};
    } else {
        OPENVINO_THROW("Unsupported model type: " + module_desc->model_type);
    }
           
    ov::Tensor latents = generator->randn_tensor(latent_shape);
    return latents;
}

int RandomLatentImageModule::get_vae_scale_factor(const std::filesystem::path &model_path) {

    auto block_out_channels = get_vae_config<std::vector<int>>(model_path, "block_out_channels");
    if (!block_out_channels.has_value() || block_out_channels->empty()) {
        return 8;
    }
    return static_cast<int>(std::pow(2, block_out_channels->size() - 1));
}

int RandomLatentImageModule::get_vae_scale_factor_spatial(const std::filesystem::path &model_path) {
    auto temperal_downsample = get_vae_config<std::vector<bool>>(model_path, "temporal_downsample");
    if (!temperal_downsample.has_value() || temperal_downsample->empty()) {
        return 8;
    }
    return static_cast<int>(std::pow(2, temperal_downsample->size()));
}

int RandomLatentImageModule::get_vae_scale_factor_temporal(const std::filesystem::path &model_path) {
    auto temperal_downsample = get_vae_config<std::vector<bool>>(model_path, "temporal_downsample");
    if (!temperal_downsample.has_value() || temperal_downsample->empty()) {
        return 4;
    }
    int count_true = 0;
    for (auto val : temperal_downsample.value()) {
        if (val) {
            count_true++;
        }
    }
    return static_cast<int>(std::pow(2, count_true));
}

template<typename T>
std::optional<T> RandomLatentImageModule::get_vae_config(const std::filesystem::path &model_path, const std::string &key) {
    std::filesystem::path vae_config_path = model_path / "vae/config.json";
    if (!std::filesystem::exists(vae_config_path)) {
        return std::nullopt;
    }
    std::ifstream vae_config(vae_config_path);
    nlohmann::json parsed = nlohmann::json::parse(vae_config);
    T value;
    utils::read_json_param(parsed, key, value);
    return value;
}

}