// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "gguf.hpp"

namespace ov {
namespace genai {
bool is_gguf_model(const std::filesystem::path& file_path);

std::map<std::string, GGUFMetaData> tokenizer_config_from_meta(
    const std::unordered_map<std::string, GGUFMetaData>& metadata);

std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, GGUFMetaData>>
create_tokenizer_from_config(const std::shared_ptr<void>& shared_object_ov_tokenizers,
                             const std::filesystem::path& gguf_model_path);

std::shared_ptr<void> load_shared_object(const std::filesystem::path& path);

void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbolName);
}  // namespace genai
}  // namespace ov
