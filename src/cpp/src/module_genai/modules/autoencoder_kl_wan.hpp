// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai::module {

class AutoencoderKLWan {
public:
    struct Config {
        size_t base_dim;
        size_t z_dim;
        size_t num_res_blocks;
        std::vector<size_t> dim_mult;
        float dropout;
        std::vector<float> latents_mean;
        std::vector<float> latents_std;
        std::vector<bool> temperal_downsample;

        explicit Config(const std::filesystem::path &config_path);
    };

    AutoencoderKLWan(const std::filesystem::path &vae_decoder_path,
                      const std::string &device,
                      const ov::AnyMap &properties = {});
    
    AutoencoderKLWan(const AutoencoderKLWan &) = delete;
    AutoencoderKLWan& operator=(const AutoencoderKLWan &) = delete;

    ov::Tensor decode(ov::Tensor latents);

private:
    Config m_config;
    ov::InferRequest m_decoder_request;
    std::shared_ptr<ov::Model> m_decoder_model;

    void init_prepostprocess(bool enable_postprocess = true);
};

}