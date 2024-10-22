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

AutoencoderKL::AutoencoderKL(const std::filesystem::path& vae_decoder_path)
    : m_config(vae_decoder_path / "config.json") {
    ov::Core core = utils::singleton_core();
    m_decoder_model = core.read_model((vae_decoder_path / "openvino_model.xml").string());
    // apply VaeImageProcessor postprocessing steps by merging them into the VAE decoder model
    merge_vae_image_post_processing();
}

AutoencoderKL::AutoencoderKL(const std::filesystem::path& vae_encoder_path,
                             const std::filesystem::path& vae_decoder_path)
    : m_config(vae_decoder_path / "config.json") {
    ov::Core core = utils::singleton_core();
    m_encoder_model = core.read_model((vae_encoder_path / "openvino_model.xml").string());
    m_decoder_model = core.read_model((vae_decoder_path / "openvino_model.xml").string());
    // apply VaeImageProcessor postprocessing steps by merging them into the VAE decoder model
    merge_vae_image_post_processing();
}

AutoencoderKL::AutoencoderKL(const std::filesystem::path& vae_decoder_path,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : AutoencoderKL(vae_decoder_path) {
    if (auto filtered_properties = extract_adapters_from_properties(properties)) {
        compile(device, *filtered_properties);
    } else {
        compile(device, properties);
    }
}

AutoencoderKL::AutoencoderKL(const std::filesystem::path& vae_encoder_path,
                             const std::filesystem::path& vae_decoder_path,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : AutoencoderKL(vae_encoder_path, vae_decoder_path) {
    if (auto filtered_properties = extract_adapters_from_properties(properties)) {
        compile(device, *filtered_properties);
    } else {
        compile(device, properties);
    }
}

AutoencoderKL::AutoencoderKL(const AutoencoderKL&) = default;

AutoencoderKL& AutoencoderKL::reshape(int batch_size, int height, int width) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot reshape already compiled model");

    const size_t vae_scale_factor = std::pow(2, m_config.block_out_channels.size() - 1);

    OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) &&
            (width % vae_scale_factor == 0 || width < 0), "Both 'width' and 'height' must be divisible by",
            vae_scale_factor);

    if (m_encoder_model) {
        ov::PartialShape input_shape = m_encoder_model->input(0).get_partial_shape();
        std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], height, width}}};
        m_encoder_model->reshape(idx_to_shape);
    }

    height /= vae_scale_factor;
    width /= vae_scale_factor;

    ov::PartialShape input_shape = m_decoder_model->input(0).get_partial_shape();
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, {batch_size, input_shape[1], height, width}}};
    m_decoder_model->reshape(idx_to_shape);

    return *this;
}

AutoencoderKL& AutoencoderKL::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::Core core = utils::singleton_core();

    if (m_encoder_model) {
        ov::CompiledModel encoder_compiled_model = core.compile_model(m_encoder_model, device, properties);
        m_encoder_request = encoder_compiled_model.create_infer_request();
        // release the original model
        m_encoder_model.reset();
    }

    ov::CompiledModel decoder_compiled_model = core.compile_model(m_decoder_model, device, properties);
    m_decoder_request = decoder_compiled_model.create_infer_request();
    // release the original model
    m_decoder_model.reset();

    return *this;
}

ov::Tensor AutoencoderKL::decode(ov::Tensor latent) {
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    m_decoder_request.set_input_tensor(latent);
    m_decoder_request.infer();
    return m_decoder_request.get_output_tensor();
}

ov::Tensor AutoencoderKL::encode(ov::Tensor image) {
    OPENVINO_ASSERT(m_decoder_request, "VAE encoder model must be compiled first. Cannot infer non-compiled model");

    m_encoder_request.set_input_tensor(image);
    m_encoder_request.infer();
    return m_encoder_request.get_output_tensor();
}

void AutoencoderKL::merge_vae_image_post_processing() const {
    ov::preprocess::PrePostProcessor ppp(m_decoder_model);

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
