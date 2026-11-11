// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov::genai {

extern const std::string FRONTEND_GGUF_READER;
extern const std::string LEGACY_GGUF_READER;

namespace gguf_utils {

// Internal property carrying a target converted together with its MTP draft. The holder is
// emptied by the first read, so a repeated read converts the file again.
extern const std::string GGUF_MTP_TARGET_PROPERTY;
using GGUFModelHolder = std::shared_ptr<std::shared_ptr<ov::Model>>;

struct GGUFProperties {
    ov::AnyMap rest;
    bool enable_save_ov_model = false;
    std::optional<bool> legacy_reader;
    GGUFModelHolder mtp_target;

    bool use_legacy_reader() const {
        return legacy_reader.value_or(true);
    }
};

GGUFProperties extract_gguf_properties(const ov::AnyMap& properties);
std::shared_ptr<ov::Model> read_gguf_model(const std::filesystem::path& model_path,
                                         const GGUFProperties& properties);
bool has_gguf_recurrent_states(const ov::Model& model);
struct GGUFMtpModels {
    std::shared_ptr<ov::Model> target;
    std::shared_ptr<ov::Model> draft;
};
/// Converts the target and its MTP draft once, with shared weight buffers.
GGUFMtpModels read_gguf_mtp_models(const std::filesystem::path& model_path, const GGUFProperties& properties);
std::shared_ptr<ov::Model> take_text_embeddings(const std::shared_ptr<ov::Model>& model);
bool has_gguf_mtp_layers(const std::filesystem::path& model_path);

}  // namespace gguf_utils
}  // namespace ov::genai
