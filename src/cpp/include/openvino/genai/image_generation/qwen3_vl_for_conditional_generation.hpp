// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/lora_adapter.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace genai {

/// @brief Qwen3-VL model producing the prompt embeddings consumed by Qwen-Image 2.1.
/// The text-only path runs the language model over 'input_ids'. The image-conditioned path additionally runs the
/// vision tower over the condition image and feeds the language model with the vision embeddings, 3D M-RoPE
/// position ids and the DeepStack features.
class OPENVINO_GENAI_EXPORTS Qwen3VLForConditionalGeneration {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t hidden_size = 4096;
        int64_t image_token_id = 151655;

        explicit Config(const std::filesystem::path& config_path);
    };

    struct OPENVINO_GENAI_EXPORTS VisionConfig {
        size_t hidden_size = 1152;
        size_t num_heads = 16;
        size_t in_channels = 3;
        size_t patch_size = 16;
        size_t temporal_patch_size = 2;
        size_t spatial_merge_size = 2;
        size_t num_position_embeddings = 2304;
        size_t num_deepstack_layers = 3;

        VisionConfig() = default;
        explicit VisionConfig(const std::filesystem::path& config_path);
    };

    /// @brief Spatial size the vision tower expects the condition image to be resized to.
    struct ImageSize {
        size_t height = 0;
        size_t width = 0;
    };

    /// @brief Rounds a target area and aspect ratio to the spatial granularity the models require.
    static ImageSize calculate_dimensions(size_t target_area, double aspect_ratio);

    explicit Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir);

    /// @brief Loads the image-conditioned models. The text-only language model is not read: both graphs wrap the
    /// same weights, and infer() always takes the image-conditioned one once a vision tower is present.
    Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir,
                                    const std::filesystem::path& vision_encoder_path,
                                    const std::filesystem::path& text_encoder_i2i_path);

    Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir,
                                    const std::string& device,
                                    const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir,
                                    const std::string& device,
                                    Properties&&... properties)
        : Qwen3VLForConditionalGeneration(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    Qwen3VLForConditionalGeneration(const Qwen3VLForConditionalGeneration&);

    std::shared_ptr<Qwen3VLForConditionalGeneration> clone();

    Qwen3VLForConditionalGeneration& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<Qwen3VLForConditionalGeneration&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    bool has_vision_tower() const;

    /// @brief Embeds a text-only prompt.
    /// @return Prompt embeddings of shape (1, prompt_sequence_length, hidden_size). The chat template prefix
    /// occupied by the system message is dropped, so the sequence length depends on the prompt.
    ov::Tensor infer(const std::string& prompt, int max_sequence_length);

    /// @brief Embeds a prompt together with a condition image.
    /// @param condition_image Normalized image of shape (1, 3, height, width) resized to get_vision_image_size().
    ov::Tensor infer(const std::string& prompt, const ov::Tensor condition_image, int max_sequence_length);

    /// @brief Marks the prompt positions the vision tower reserved for the condition image.
    /// @return Boolean tensor of shape (1, prompt_sequence_length). All false after a text-only infer().
    ov::Tensor get_image_pad_mask() const;

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    const Config& get_config() const;

    const VisionConfig& get_vision_config() const;

private:
    static const std::string SYSTEM_PREFIX;
    static const std::string PROMPT_TEMPLATE;
    static const std::string PROMPT_TEMPLATE_WITH_IMAGE;

    ov::Tensor infer_vision_tower(const ov::Tensor condition_image, std::vector<ov::Tensor>& deepstack_features);

    ov::Tensor drop_system_prefix(const ov::Tensor hidden_states, size_t prompt_length) const;

    Config m_config;
    VisionConfig m_vision_config;
    AdapterController m_adapter_controller;
    ov::InferRequest m_request, m_vision_request, m_i2i_request;
    std::shared_ptr<ov::Model> m_model, m_vision_model, m_i2i_model;
    Tokenizer m_tokenizer;
    size_t m_system_prefix_length;
    ov::Tensor m_image_pad_mask;
};

}  // namespace genai
}  // namespace ov
