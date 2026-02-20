// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>
#include <utility>
#include <variant>

#include "openvino/runtime/core.hpp"

namespace ov {
namespace genai {

using PathExtensions = std::vector<std::variant<std::filesystem::path, std::shared_ptr<ov::Extension>>>;

/**
 * @brief Wrap paths and Extensions into an ov::AnyMap-compatible pair that can be passed to pipeline constructors
 * like LLMPipeline or VLMPipeline, ensuring that the extensions are loaded into the OpenVINO Core instance before
 * model loading.
 */
template <typename T,
          typename = std::enable_if_t<
              std::is_same_v<std::decay_t<T>,
                             std::vector<std::variant<std::filesystem::path, std::shared_ptr<ov::Extension>>>>>>
std::pair<std::string, ov::Any> extensions(T&& vector) {
    return {"extensions", ov::Any(std::forward<T>(vector))};
}

/**
 * @brief A helper allowing extensions({"path"}) instead of explicit std::variant.
 *
 * The following code is an example usage showing how to pass extensions to pipeline constructors
 *
 * @code
 * ov::AnyMap properties = {ov::genai::extensions(std::vector<std::filesystem::path>{"path1", "path2"})};
 * ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, device, properties);
 * @endcode
 */
inline std::pair<std::string, ov::Any> extensions(const std::vector<std::filesystem::path>& paths) {
    return extensions(
        std::vector<std::variant<std::filesystem::path, std::shared_ptr<ov::Extension>>>{paths.begin(), paths.end()});
}

/**
 * @brief A helper allowing extensions({extensions}) instead of explicit std::variant.
 */
inline std::pair<std::string, ov::Any> extensions(const std::vector<std::shared_ptr<ov::Extension>>& extension_list) {
    return extensions(
        std::vector<std::variant<std::filesystem::path, std::shared_ptr<ov::Extension>>>{extension_list.begin(),
                                                                                         extension_list.end()});
}

}  // namespace genai
}  // namespace ov
