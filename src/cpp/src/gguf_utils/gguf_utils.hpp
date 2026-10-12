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

bool is_gguf_model(const std::filesystem::path& file_path);

namespace gguf_utils {

struct GGUFProperties {
    ov::AnyMap rest;
    bool enable_save_ov_model = false;
    std::optional<bool> legacy_reader;

    bool use_legacy_reader() const {
        return legacy_reader.value_or(true);
    }
};

GGUFProperties extract_gguf_properties(const ov::AnyMap& properties);
std::shared_ptr<ov::Model> read_gguf_model(const std::filesystem::path& model_path,
                                         const GGUFProperties& properties);
bool has_gguf_recurrent_states(const ov::Model& model);
void save_openvino_model(const std::shared_ptr<ov::Model>& model,
                          const std::string& save_path,
                          bool compress_to_fp16);
void print_gguf_debug_info(const std::string& debug_info);

}  // namespace gguf_utils
}  // namespace ov::genai
