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

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS AutoencoderKLQwenImage {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t z_dim = 16;
        size_t input_channels = 3;
        std::vector<bool> temperal_downsample = {false, true, true};
        std::vector<float> latents_mean;
        std::vector<float> latents_std;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit AutoencoderKLQwenImage(const std::filesystem::path& vae_decoder_path);

    AutoencoderKLQwenImage(const std::filesystem::path& vae_encoder_path,
                           const std::filesystem::path& vae_decoder_path);

    AutoencoderKLQwenImage(const std::filesystem::path& vae_decoder_path,
                           const std::string& device,
                           const ov::AnyMap& properties = {});

    AutoencoderKLQwenImage(const std::filesystem::path& vae_encoder_path,
                           const std::filesystem::path& vae_decoder_path,
                           const std::string& device,
                           const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    AutoencoderKLQwenImage(const std::filesystem::path& vae_decoder_path,
                           const std::string& device,
                           Properties&&... properties)
        : AutoencoderKLQwenImage(vae_decoder_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    AutoencoderKLQwenImage(const std::filesystem::path& vae_encoder_path,
                           const std::filesystem::path& vae_decoder_path,
                           const std::string& device,
                           Properties&&... properties)
        : AutoencoderKLQwenImage(vae_encoder_path, vae_decoder_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    AutoencoderKLQwenImage(const AutoencoderKLQwenImage&);

    AutoencoderKLQwenImage clone();

    AutoencoderKLQwenImage& reshape(int batch_size, int height, int width);

    AutoencoderKLQwenImage& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<AutoencoderKLQwenImage&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    // Decode latent to image. Input: 5D (B, z_dim, 1, H_lat, W_lat), Output: NHWC u8 image.
    ov::Tensor decode(ov::Tensor latent);

    // Encode image to latent parameters. Input: 5D (B, 3, 1, H, W), Output: 5D (B, z_dim*2, 1, H_lat, W_lat).
    ov::Tensor encode(ov::Tensor image, std::shared_ptr<Generator> generator);

    ov::Tensor encode(ov::Tensor image);

    const Config& get_config() const;

    size_t get_vae_scale_factor() const;

private:
    void merge_vae_image_post_processing() const;

    Config m_config;
    ov::InferRequest m_encoder_request, m_decoder_request;
    std::shared_ptr<ov::Model> m_encoder_model = nullptr, m_decoder_model = nullptr;
};

}  // namespace genai
}  // namespace ov
