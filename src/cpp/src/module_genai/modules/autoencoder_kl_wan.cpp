// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "autoencoder_kl_wan.hpp"
#include "utils.hpp"
#include "json_utils.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include <fstream>

namespace ov::genai::module {

AutoencoderKLWan::Config::Config(const std::filesystem::path &config_path) {
    if (!std::filesystem::exists(config_path)) {
        OPENVINO_THROW("AutoencoderKLWan config file does not exist: " + config_path.string());
    }
    std::ifstream config_file(config_path);
    nlohmann::json parsed = nlohmann::json::parse(config_file);
    utils::read_json_param(parsed, "base_dim", base_dim);
    utils::read_json_param(parsed, "z_dim", z_dim);
    utils::read_json_param(parsed, "num_res_blocks", num_res_blocks);
    utils::read_json_param(parsed, "dim_mult", dim_mult);
    utils::read_json_param(parsed, "dropout", dropout);
    utils::read_json_param(parsed, "latents_mean", latents_mean);
    utils::read_json_param(parsed, "latents_std", latents_std);
    utils::read_json_param(parsed, "temperal_downsample", temperal_downsample);
}

AutoencoderKLWan::AutoencoderKLWan(const std::filesystem::path &vae_decoder_path,
                                   const std::string &device,
                                   const ov::AnyMap &properties)
    : m_config(vae_decoder_path / "config.json") {
    if (std::filesystem::exists(vae_decoder_path / "openvino_model.xml")) {
        m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "openvino_model.xml");
    } else if (std::filesystem::exists(vae_decoder_path / "vae_decoder.xml")) {
        m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "vae_decoder.xml");
    } else {
        OPENVINO_THROW("AutoencoderKLWan decoder model file does not exist in: " + vae_decoder_path.string());
    }

    auto properties_copy = properties;
    bool enable_postprocess = true;
    if (auto it = properties_copy.find("enable_postprocess"); it != properties_copy.end()) {
        enable_postprocess = it->second.as<bool>();
        properties_copy.erase(it);
    }
    
    init_prepostprocess(enable_postprocess);

    m_decoder_request = utils::singleton_core().compile_model(m_decoder_model, device, properties_copy).create_infer_request();
}

void AutoencoderKLWan::init_prepostprocess(bool enable_postprocess) {
    ov::preprocess::PrePostProcessor ppp(m_decoder_model);
    ppp.input().tensor().set_layout("NCDHW");
    ppp.output().model().set_layout("NCDHW");
    std::vector<float> inv_std, neg_mean;
    for (size_t i = 0; i < m_config.latents_mean.size(); i++) {
        inv_std.push_back(1.0f / m_config.latents_std[i]);
        neg_mean.push_back(-m_config.latents_mean[i]);
    }

    ppp.input().preprocess()
        .scale(inv_std)
        .mean(neg_mean);
    
    if (enable_postprocess) {
        ppp.output().postprocess().custom([](const ov::Output<ov::Node> &port) {
            auto permute = ov::op::v0::Constant::create(
                ov::element::i64,
                ov::Shape{5},
                {0, 2, 1, 3, 4});
            auto transposed = std::make_shared<ov::op::v1::Transpose>(port, permute);
            auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
            auto scaled_0_5 = std::make_shared<ov::op::v1::Multiply>(transposed, constant_0_5);
            auto added_0_5 = std::make_shared<ov::op::v1::Add>(scaled_0_5, constant_0_5);
            auto clamped = std::make_shared<ov::op::v0::Clamp>(added_0_5, 0.0f, 1.0f);
            auto constant_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);
            auto multiplied = std::make_shared<ov::op::v1::Multiply>(clamped, constant_255);
            auto permute_1 = ov::op::v0::Constant::create(
                ov::element::i64,
                ov::Shape{5},
                {0, 1, 3, 4, 2});
            return std::make_shared<ov::op::v1::Transpose>(multiplied, permute_1);
        });
        ppp.output().postprocess().convert_element_type(ov::element::u8);
    }
    m_decoder_model = ppp.build();
}

ov::Tensor AutoencoderKLWan::decode(ov::Tensor latents) {
    m_decoder_request.set_input_tensor(latents);
    m_decoder_request.infer();
    return m_decoder_request.get_output_tensor();
}

}