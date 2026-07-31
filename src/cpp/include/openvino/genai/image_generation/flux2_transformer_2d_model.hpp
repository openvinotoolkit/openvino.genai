// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/lora_adapter.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS Flux2Transformer2DModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 128;
        bool guidance_embeds = true;
        size_t default_sample_size = 128;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit Flux2Transformer2DModel(const std::filesystem::path& root_dir);

    Flux2Transformer2DModel(const std::filesystem::path& root_dir,
                            const std::string& device,
                            const ov::AnyMap& properties = {});

    Flux2Transformer2DModel(const std::string& model,
                            const Tensor& weights,
                            const Config& config,
                            const size_t vae_scale_factor);

    Flux2Transformer2DModel(const std::string& model,
                            const Tensor& weights,
                            const Config& config,
                            const size_t vae_scale_factor,
                            const std::string& device,
                            const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Flux2Transformer2DModel(const std::filesystem::path& root_dir, const std::string& device, Properties&&... properties)
        : Flux2Transformer2DModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Flux2Transformer2DModel(const std::string& model,
                            const Tensor& weights,
                            const Config& config,
                            const size_t vae_scale_factor,
                            const std::string& device,
                            Properties&&... properties)
        : Flux2Transformer2DModel(model, weights, config, vae_scale_factor, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    Flux2Transformer2DModel(const Flux2Transformer2DModel&);

    Flux2Transformer2DModel clone();

    const Config& get_config() const;

    Flux2Transformer2DModel& reshape(int batch_size, int height, int width, int tokenizer_model_max_length);

    Flux2Transformer2DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<Flux2Transformer2DModel&, Properties...> compile(const std::string& device,
                                                                                    Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states);

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    ov::Tensor infer(const ov::Tensor latent, const ov::Tensor timestep);

private:
    Config m_config;
    AdapterController m_adapter_controller;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    size_t m_vae_scale_factor;
};

}  // namespace genai
}  // namespace ov
