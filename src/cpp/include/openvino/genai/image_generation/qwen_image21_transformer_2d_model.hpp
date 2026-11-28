// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
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
/// A graph exported with the prefix KV cache keeps the text and condition-image (prefix) keys/values, which do not
/// change across denoising steps, in its state: a call over the whole joint sequence appends them to the state, and a
/// call over the target image's tokens only reads them. Every cache context (e.g. each CFG pass) has its own infer
/// request and thus its own cache.
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

    /// @brief Denoises the tokens selected by the bound joint-sequence inputs.
    /// @return The transformer output over the processed tokens, of shape (batch, processed_length, out_channels).
    ov::Tensor infer(const ov::Tensor latent, const ov::Tensor timestep);

    /// @brief Whether the graph keeps the prefix keys/values in its state. Models exported before the prefix KV
    /// cache have no state and always process the whole joint sequence.
    bool has_prefix_cache() const;

    /// @brief Makes the infer request of the given cache context the one that the inputs, the adapters and the
    /// inference apply to. Context 0 is the request created at compilation, the other ones are created on first use.
    void set_cache_context(size_t context);

    /// @brief Empties the prefix KV cache of the current cache context, keeping the rest of its state (e.g. LoRA).
    void reset_prefix_cache();

private:
    Config m_config;
    AdapterController m_adapter_controller;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    bool m_has_prefix_cache = false;
    // Infer requests of the cache contexts, m_request is the current one. Empty until a context is selected.
    std::vector<ov::InferRequest> m_context_requests;
    std::optional<AdapterConfig> m_adapters;
};

}  // namespace genai
}  // namespace ov
