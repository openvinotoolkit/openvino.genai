// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "openvino/runtime/core.hpp"

namespace ov {
namespace genai {

/**
 * @brief Wrap paths and Extensions into ov::AnyMap compatible pair. ov::Core::add_extension() is called for each item
 * in pipeline constructor.
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
 */
inline std::pair<std::string, ov::Any> extensions(const std::vector<std::filesystem::path>& paths) {
    return extensions(
        std::vector<std::variant<std::filesystem::path, std::shared_ptr<ov::Extension>>>{paths.begin(), paths.end()});
}

/**
 * @brief A helper allowing extensions({extensions}) instead of explicit std::variant.
 */
inline std::pair<std::string, ov::Any> extensions(const std::vector<std::shared_ptr<ov::Extension>>& extens) {
    return extensions(
        std::vector<std::variant<std::filesystem::path, std::shared_ptr<ov::Extension>>>{extens.begin(), extens.end()});
}

/**
 * @brief A helper allowing extensions(initializer_list{"path"}) instead of explicit std::variant.
 */
template <class T, class = std::enable_if_t<std::is_constructible<std::filesystem::path, const T&>::value>>
inline std::pair<std::string, ov::Any> extensions(std::initializer_list<T> paths) {
    std::vector<std::variant<std::filesystem::path, std::shared_ptr<ov::Extension>>> paths_var;
    for (const auto& path : paths) {
        paths_var.emplace_back(std::filesystem::path{path});
    }
    return extensions(std::move(paths_var));
}

}  // namespace genai
}  // namespace ov
