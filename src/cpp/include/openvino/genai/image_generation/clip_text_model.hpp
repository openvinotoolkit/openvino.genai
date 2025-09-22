// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/lora_adapter.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS CLIPTextModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t max_position_embeddings = 77;
        size_t num_hidden_layers = 12;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit CLIPTextModel(const std::filesystem::path& root_dir);

    CLIPTextModel(const std::filesystem::path& root_dir,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    CLIPTextModel(const std::string& model,
                  const Tensor& weights,
                  const Config& config,
                  const Tokenizer& clip_tokenizer);

    CLIPTextModel(const std::string& model,
                  const Tensor& weights,
                  const Config& config,
                  const Tokenizer& clip_tokenizer,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    CLIPTextModel(const std::filesystem::path& root_dir,
                  const std::string& device,
                  Properties&&... properties)
        : CLIPTextModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    CLIPTextModel(const std::string& model,
                  const Tensor& weights,
                  const Config& config,
                  const Tokenizer& clip_tokenizer,
                  const std::string& device,
                  Properties&&... properties)
        : CLIPTextModel(model,
                        weights,
                        config,
                        clip_tokenizer,
                        device,
                        ov::AnyMap{std::forward<Properties>(properties)...}) { }

    CLIPTextModel(const CLIPTextModel&);

    std::shared_ptr<CLIPTextModel> clone();

    const Config& get_config() const;

    CLIPTextModel& reshape(int batch_size);

    CLIPTextModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<CLIPTextModel&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    ov::Tensor infer(const std::string& pos_prompt, const std::string& neg_prompt, bool do_classifier_free_guidance);

    ov::Tensor get_output_tensor(const size_t idx);

    /**
     * @brief Exports compiled model to a specified directory.
     * @param export_path A path to a directory to export compiled model to
     *
     * See @ref ov::genai::blob_path property to load previously exported model and for more details.
     */
    void export_model(const std::filesystem::path& export_path);

private:
    Config m_config;
    AdapterController m_adapter_controller;
    Tokenizer m_clip_tokenizer;
    bool m_slice_batch1_output = false;

protected:
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;

    void import_model(const std::filesystem::path& blob_path, const std::string& device, const ov::AnyMap& properties);
};

} // namespace genai
} // namespace ov
