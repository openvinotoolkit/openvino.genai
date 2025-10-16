// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>
#include <string>
#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/lora_adapter.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS UNet2DConditionModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 4;
        size_t sample_size = 0;
        int time_cond_proj_dim = -1;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit UNet2DConditionModel(const std::filesystem::path& root_dir);

    UNet2DConditionModel(const std::filesystem::path& root_dir,
                         const std::string& device,
                         const ov::AnyMap& properties = {});

    UNet2DConditionModel(const std::string& model,
                         const Tensor& weights,
                         const Config& config,
                         const size_t vae_scale_factor);

    UNet2DConditionModel(const std::string& model,
                         const Tensor& weights,
                         const Config& config,
                         const size_t vae_scale_factor,
                         const std::string& device,
                         const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    UNet2DConditionModel(const std::filesystem::path& root_dir,
                         const std::string& device,
                         Properties&&... properties)
        : UNet2DConditionModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    UNet2DConditionModel(const std::string& model,
                         const Tensor& weights,
                         const Config& config,
                         const size_t vae_scale_factor,
                         const std::string& device,
                         Properties&&... properties)
        : UNet2DConditionModel(model,
                               weights,
                               config,
                               vae_scale_factor,
                               device,
                               ov::AnyMap{std::forward<Properties>(properties)...}) { }

    UNet2DConditionModel(const UNet2DConditionModel&);

    UNet2DConditionModel clone();

    const Config& get_config() const;

    UNet2DConditionModel& reshape(int batch_size, int height, int width, int tokenizer_model_max_length);

    UNet2DConditionModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<UNet2DConditionModel&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states);

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    ov::Tensor infer(ov::Tensor sample, ov::Tensor timestep);

    bool do_classifier_free_guidance(float guidance_scale) const {
        return guidance_scale > 1.0f && m_config.time_cond_proj_dim < 0;
    }

    /**
     * @brief Exports compiled model to a specified directory.
     * @param export_path A path to a directory to export compiled model to
     *
     * See @ref ov::genai::blob_path property to load previously exported model and for more details.
     */
    void export_model(const std::filesystem::path& export_path);

private:
    class UNetInference;
    std::shared_ptr<UNetInference> m_impl;

    Config m_config;
    AdapterController m_adapter_controller;
    std::shared_ptr<ov::Model> m_model;
    size_t m_vae_scale_factor;

    void import_model(const std::filesystem::path& blob_path, const std::string& device, const ov::AnyMap& properties = {});

    class UNetInferenceDynamic;
    class UNetInferenceStaticBS1;
};

} // namespace genai
} // namespace ov
