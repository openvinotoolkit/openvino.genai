// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "gguf.hpp"

using FactoryCreateType = ov::OutputVector (*)(const std::string& op_type,
                                               const ov::OutputVector& inputs,
                                               const ov::AnyMap& attributes);

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

template <typename T>
const T* get_if_exist(const std::map<std::string, GGUFMetaData>& tokenizer_config, const std::string& attribute_name) {
    if (tokenizer_config.count(attribute_name)) {
        auto val = std::get_if<T>(&tokenizer_config.at(attribute_name));
        return val;
    }
    return nullptr;
}

/**
 * @brief This function aim to patch the chat template stored in gguf model to
 * be consistent with chat template stored in the original tokenizer_config.json of huggingface models.
 * If certain mismatched pattern found, then the pattern will be replaced with a specific substring.
 * Otherwise, the original chat template is returned.
 * Current this function is used to patch the chat template for Qwen2.5 models, but the logic can be extended to other
 * models
 *
 *
 * Example: The function finds the substring for Qwen2.5:
 * "{{\"name\": <function-name>, \"arguments\": <args-json-object>}}"
 * in the input string (str1_content) and replaces it with:
 * "{\"name\": <function-name>, \"arguments\": <args-json-object>}"
 *
 * This assumes that "<function-name>" and "<args-json-object>" are literal
 * parts of the substring to be found.
 *
 * @param chat_template A string contains original chat template stored in gguf models
 * @return patched_chat_template A new string contains updated chat template with the specific replacement made if
 * pattern matched.
 */
std::string patch_gguf_chat_template(const std::string& chat_template);

}  // namespace genai
}  // namespace ov
