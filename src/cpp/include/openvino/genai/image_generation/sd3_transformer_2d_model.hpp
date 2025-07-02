// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/lora_adapter.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS SD3Transformer2DModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t sample_size = 128;
        size_t patch_size = 2;
        size_t in_channels = 16;
        size_t joint_attention_dim = 4096;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit SD3Transformer2DModel(const std::filesystem::path& root_dir);

    SD3Transformer2DModel(const std::filesystem::path& root_dir,
                          const std::string& device,
                          const ov::AnyMap& properties = {});

    SD3Transformer2DModel(const std::string& model,
                          const Tensor& weights,
                          const Config& config,
                          const size_t vae_scale_factor);

    SD3Transformer2DModel(const std::string& model,
                          const Tensor& weights,
                          const Config& config,
                          const size_t vae_scale_factor,
                          const std::string& device,
                          const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    SD3Transformer2DModel(const std::filesystem::path& root_dir, const std::string& device, Properties&&... properties)
        : SD3Transformer2DModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    SD3Transformer2DModel(const std::string& model,
                          const Tensor& weights,
                          const Config& config,
                          const size_t vae_scale_factor,
                          const std::string& device,
                          Properties&&... properties)
        : SD3Transformer2DModel(model, weights, config, vae_scale_factor, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    SD3Transformer2DModel(const SD3Transformer2DModel&);

    SD3Transformer2DModel clone();

    const Config& get_config() const;

    SD3Transformer2DModel& reshape(int batch_size, int height, int width, int tokenizer_model_max_length);

    SD3Transformer2DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<SD3Transformer2DModel&, Properties...> compile(const std::string& device,
                                                                                  Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states);

    ov::Tensor infer(const ov::Tensor latent, const ov::Tensor timestep);

private:
    class Inference;
    std::shared_ptr<Inference> m_impl;

    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    size_t m_vae_scale_factor;
    AdapterController m_adapter_controller;

    class InferenceDynamic;
    class InferenceStaticBS1;
};

}  // namespace genai
}  // namespace ov
