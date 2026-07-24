// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/lora_adapter.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS QwenImageTransformer2DModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 64;
        size_t out_channels = 16;
        bool guidance_embeds = false;
        size_t joint_attention_dim = 3584;
        size_t attention_head_dim = 128;
        size_t patch_size = 2;
        std::vector<size_t> axes_dims_rope = {16, 56, 56};
        size_t default_sample_size = 128;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit QwenImageTransformer2DModel(const std::filesystem::path& root_dir);

    QwenImageTransformer2DModel(const std::filesystem::path& root_dir,
                                const std::string& device,
                                const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    QwenImageTransformer2DModel(const std::filesystem::path& root_dir, const std::string& device, Properties&&... properties)
        : QwenImageTransformer2DModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    QwenImageTransformer2DModel(const QwenImageTransformer2DModel&);

    QwenImageTransformer2DModel clone();

    const Config& get_config() const;

    QwenImageTransformer2DModel& reshape(int batch_size, int height, int width, int tokenizer_model_max_length);

    QwenImageTransformer2DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<QwenImageTransformer2DModel&, Properties...> compile(const std::string& device,
                                                                                        Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_hidden_states(const std::string& tensor_name, ov::Tensor tensor);

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
