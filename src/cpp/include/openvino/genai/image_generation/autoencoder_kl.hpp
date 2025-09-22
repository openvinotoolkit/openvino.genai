// Copyright (C) 2023-2025 Intel Corporation
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

class OPENVINO_GENAI_EXPORTS AutoencoderKL {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 3;
        size_t latent_channels = 4;
        size_t out_channels = 3;
        float scaling_factor = 1.0f;
        float shift_factor = 0.0f;
        std::vector<size_t> block_out_channels = { 64 };

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit AutoencoderKL(const std::filesystem::path& vae_decoder_path);

    AutoencoderKL(const std::filesystem::path& vae_encoder_path,
                  const std::filesystem::path& vae_decoder_path);

    AutoencoderKL(const std::filesystem::path& vae_decoder_path,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    AutoencoderKL(const std::filesystem::path& vae_encoder_path,
                  const std::filesystem::path& vae_decoder_path,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    AutoencoderKL(const std::string& vae_decoder_model,
                  const Tensor& vae_decoder_weights,
                  const Config& vae_decoder_config);

    AutoencoderKL(const std::string& vae_encoder_model,
                  const Tensor& vae_encoder_weights,
                  const std::string& vae_decoder_model,
                  const Tensor& vae_decoder_weights,
                  const Config& vae_decoder_config);

    AutoencoderKL(const std::string& vae_decoder_model,
                  const Tensor& vae_decoder_weights,
                  const Config& vae_decoder_config,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    AutoencoderKL(const std::string& vae_encoder_model,
                  const Tensor& vae_encoder_weights,
                  const std::string& vae_decoder_model,
                  const Tensor& vae_decoder_weights,
                  const Config& vae_decoder_config,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    AutoencoderKL(const std::filesystem::path& vae_decoder_path,
                  const std::string& device,
                  Properties&&... properties)
        : AutoencoderKL(vae_decoder_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    AutoencoderKL(const std::filesystem::path& vae_encoder_path,
                  const std::filesystem::path& vae_decoder_path,
                  const std::string& device,
                  Properties&&... properties)
        : AutoencoderKL(vae_encoder_path, vae_decoder_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    AutoencoderKL(const std::string& vae_decoder_model,
                  const Tensor& vae_decoder_weights,
                  const Config& vae_decoder_config,
                  const std::string& device,
                  Properties&&... properties)
        : AutoencoderKL(vae_decoder_model,
                        vae_decoder_weights,
                        vae_decoder_config,
                        device,
                        ov::AnyMap{std::forward<Properties>(properties)...}) { }

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    AutoencoderKL(const std::string& vae_encoder_model,
                  const Tensor& vae_encoder_weights,
                  const std::string& vae_decoder_model,
                  const Tensor& vae_decoder_weights,
                  const Config& vae_decoder_config,
                  const std::string& device,
                  Properties&&... properties)
        : AutoencoderKL(vae_encoder_model,
                        vae_encoder_weights,
                        vae_decoder_model,
                        vae_decoder_weights,
                        vae_decoder_config,
                        device,
                        ov::AnyMap{std::forward<Properties>(properties)...}) { }

    AutoencoderKL(const AutoencoderKL&);

    AutoencoderKL clone();

    AutoencoderKL& reshape(int batch_size, int height, int width);

    AutoencoderKL& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<AutoencoderKL&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ov::Tensor decode(ov::Tensor latent);

    ov::Tensor encode(ov::Tensor image, std::shared_ptr<Generator> generator);

    const Config& get_config() const;

    size_t get_vae_scale_factor() const;

    /**
     * @brief Exports compiled models to a specified directory.
     * @param export_path A path to a directory to export compiled models to
     *
     * See @ref ov::genai::blob_path property to load previously exported models and for more details.
     */
    void export_model(const std::filesystem::path& export_path);

private:
    void merge_vae_image_post_processing() const;
    void import_model(const std::filesystem::path& blob_path, const std::string& device, const ov::AnyMap& properties = {});

    Config m_config;
    ov::InferRequest m_encoder_request, m_decoder_request;
    std::shared_ptr<ov::Model> m_encoder_model = nullptr, m_decoder_model = nullptr;
};

} // namespace genai
} // namespace ov
