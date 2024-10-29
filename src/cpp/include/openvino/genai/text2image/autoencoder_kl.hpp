// Copyright (C) 2023-2024 Intel Corporation
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

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS AutoencoderKL {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 3;
        size_t latent_channels = 4;
        size_t out_channels = 3;
        float scaling_factor = 0.18215f;
        float shift_factor = 0.0609f;
        std::vector<size_t> block_out_channels = { 64 };

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit AutoencoderKL(const std::filesystem::path& root_dir);

    AutoencoderKL(const std::filesystem::path& root_dir,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    AutoencoderKL(const std::filesystem::path& root_dir,
                  const std::string& device,
                  Properties&&... properties)
        : AutoencoderKL(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    AutoencoderKL(const AutoencoderKL&);

    AutoencoderKL& reshape(int batch_size, int height, int width);

    AutoencoderKL& compile(const std::string& device, const ov::AnyMap& properties = {});

    const Config& get_config() const;

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<AutoencoderKL&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ov::Tensor infer(ov::Tensor latent);

private:
    void merge_vae_image_processor() const;

    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

} // namespace genai
} // namespace ov
