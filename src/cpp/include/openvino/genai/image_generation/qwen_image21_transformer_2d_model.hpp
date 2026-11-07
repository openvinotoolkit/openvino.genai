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

/// @brief Single-stream block-causal transformer of Qwen-Image 2.1.
/// The exported graph is a single pass whose data-dependent tensors (rotary 'cos'/'sin', the joint-sequence
/// 'gather_idx', the dense block-causal 'attn_mask' and the 'modulation_mask') are precomputed by the pipeline
/// and passed in as graph inputs.
class OPENVINO_GENAI_EXPORTS QwenImage21Transformer2DModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 64;
        size_t out_channels = 64;
        size_t context_in_dim = 4096;
        size_t attention_head_dim = 128;
        size_t num_layers = 32;
        std::vector<size_t> axes_dims_rope = {16, 56, 56};

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit QwenImage21Transformer2DModel(const std::filesystem::path& root_dir);

    QwenImage21Transformer2DModel(const std::filesystem::path& root_dir,
                                  const std::string& device,
                                  const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    QwenImage21Transformer2DModel(const std::filesystem::path& root_dir,
                                  const std::string& device,
                                  Properties&&... properties)
        : QwenImage21Transformer2DModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    QwenImage21Transformer2DModel(const QwenImage21Transformer2DModel&);

    QwenImage21Transformer2DModel clone();

    const Config& get_config() const;

    QwenImage21Transformer2DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<QwenImage21Transformer2DModel&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_hidden_states(const std::string& tensor_name, ov::Tensor tensor);

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    /// @brief Denoises the joint sequence.
    /// @return The transformer output over the whole joint sequence, of shape (batch, joint_sequence_length, out_channels).
    ov::Tensor infer(const ov::Tensor latent, const ov::Tensor timestep);

private:
    Config m_config;
    AdapterController m_adapter_controller;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

}  // namespace genai
}  // namespace ov
