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

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS LTXVideoTransformer3DModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        // add params if necessary
        size_t in_channels = 128;
        size_t patch_size = 1;
        size_t patch_size_t = 1;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit LTXVideoTransformer3DModel(const std::filesystem::path& root_dir);

    LTXVideoTransformer3DModel(const std::filesystem::path& root_dir,
                          const std::string& device,
                          const ov::AnyMap& properties = {});

    LTXVideoTransformer3DModel(const LTXVideoTransformer3DModel&);

    const Config& get_config() const;

    LTXVideoTransformer3DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<LTXVideoTransformer3DModel&, Properties...> compile(const std::string& device,
                                                                                  Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states);

    ov::Tensor infer(const ov::Tensor latent, const ov::Tensor timestep);

    LTXVideoTransformer3DModel& reshape(int64_t batch_size, int64_t num_frames, int64_t height, int64_t width, int64_t tokenizer_model_max_length);

    void get_patch_size(const std::filesystem::path& config_path, int64_t& patch_size, int64_t& patch_size_t);

private:
    class Inference;
    std::shared_ptr<Inference> m_impl;

    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    int64_t m_spatial_compression_ratio, m_temporal_compression_ratio; // calculated based on vae config, needed for reshape

    class InferenceDynamic;
    class InferenceStaticBS1;
};

}  // namespace genai
}  // namespace ov
