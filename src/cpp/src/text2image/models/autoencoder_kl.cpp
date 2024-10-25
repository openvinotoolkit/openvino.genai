// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text2image/autoencoder_kl.hpp"

#include <fstream>
#include <memory>

#include "openvino/runtime/core.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"

#include "utils.hpp"

#include "json_utils.hpp"
#include "lora_helper.hpp"

namespace ov {
namespace genai {

AutoencoderKL::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "latent_channels", latent_channels);
    read_json_param(data, "out_channels", out_channels);
    read_json_param(data, "scaling_factor", scaling_factor);
    read_json_param(data, "block_out_channels", block_out_channels);
}

AutoencoderKL::AutoencoderKL(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    ov::Core core = utils::singleton_core();
    m_model = core.read_model((root_dir / "openvino_model.xml").string());
    // apply VaeImageProcessor postprocessing steps by merging them into the VAE decoder model
    merge_vae_image_processor();
}

AutoencoderKL::AutoencoderKL(const std::filesystem::path& root_dir,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : AutoencoderKL(root_dir) {
    compile(device, properties);
}

AutoencoderKL::AutoencoderKL(const AutoencoderKL&) = default;

AutoencoderKL& AutoencoderKL::reshape(int batch_size, int height, int width) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    const size_t vae_scale_factor = std::pow(2, m_config.block_out_channels.size() - 1);

    height /= vae_scale_factor;
    width /= vae_scale_factor;

    ov::PartialShape input_shape = m_model->input(0).get_partial_shape();
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], height, width}}};
    m_model->reshape(idx_to_shape);

    return *this;
}

AutoencoderKL& AutoencoderKL::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model;
    if (auto filtered_properties = extract_adapters_from_properties(properties)) {
        compiled_model = core.compile_model(m_model, device, *filtered_properties);
    } else {
        compiled_model = core.compile_model(m_model, device, properties);
    }
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor AutoencoderKL::infer(ov::Tensor latent) {
    OPENVINO_ASSERT(m_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    m_request.set_input_tensor(latent);
    m_request.infer();
    return m_request.get_output_tensor();
}

void AutoencoderKL::merge_vae_image_processor() const {
    ov::preprocess::PrePostProcessor ppp(m_model);

    // scale input before VAE encoder
    ppp.input().preprocess().scale(m_config.scaling_factor);

    // apply VaeImageProcessor normalization steps
    // https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/image_processor.py#L159
    ppp.output().postprocess().custom([](const ov::Output<ov::Node>& port) {
        auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
        auto constant_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);
        auto scaled_0_5 = std::make_shared<ov::op::v1::Multiply>(port, constant_0_5);
        auto added_0_5 = std::make_shared<ov::op::v1::Add>(scaled_0_5, constant_0_5);
        auto clamped = std::make_shared<ov::op::v0::Clamp>(added_0_5, 0.0f, 1.0f);
        return std::make_shared<ov::op::v1::Multiply>(clamped, constant_255);
    });
    ppp.output().postprocess().convert_element_type(ov::element::u8);
    // layout conversion
    // https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/image_processor.py#L144
    ppp.output().model().set_layout("NCHW");
    ppp.output().tensor().set_layout("NHWC");

    ppp.build();
}

} // namespace genai
} // namespace ov
