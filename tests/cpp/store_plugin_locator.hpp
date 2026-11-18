// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>

namespace ov::genai::test {

/// A path in the environment variable is taken as is, so a CI job that sets it fails on a wrong path instead of skipping.
/// Otherwise the first existing candidate is used; candidates are relative to the working directory of a build-tree run.
inline std::optional<std::string> find_plugin_library(const char* env_name,
                                                      std::initializer_list<const char*> build_tree_candidates) {
    if (const char* from_env = std::getenv(env_name); from_env != nullptr && *from_env != '\0') {
        return std::string(from_env);
    }
    for (const char* candidate : build_tree_candidates) {
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::absolute(candidate).string();
        }
    }
    return std::nullopt;
}

/// openvino_genai's own shared library, which exports the store ABI entry symbol as well (abi_file_store_provider.cpp).
inline std::optional<std::string> find_self_as_plugin_library() {
    return find_plugin_library("KV_CACHE_STORE_SELF_PLUGIN",
#ifdef _WIN32
                               {
                                   "../../openvino_genai/openvino_genai.dll",
                                   "../openvino_genai/openvino_genai.dll",
                                   "openvino_genai/openvino_genai.dll",
                               }
#elif defined(__APPLE__)
                               {
                                   "../../openvino_genai/libopenvino_genai.dylib",
                                   "../openvino_genai/libopenvino_genai.dylib",
                                   "openvino_genai/libopenvino_genai.dylib",
                               }
#else
                               {
                                   "../../openvino_genai/libopenvino_genai.so",
                                   "../openvino_genai/libopenvino_genai.so",
                                   "openvino_genai/libopenvino_genai.so",
                               }
#endif
    );
}

/// The standalone plugin of samples/c/kv_cache_store_plugin, which is built only with ENABLE_SAMPLES.
inline std::optional<std::string> find_sample_plugin_library() {
    return find_plugin_library("KV_CACHE_STORE_SAMPLE_PLUGIN",
#ifdef _WIN32
                               {
                                   "../../samples/c/kv_cache_store_plugin/Release/sample_kv_cache_store_plugin.dll",
                                   "../../samples/c/kv_cache_store_plugin/RelWithDebInfo/sample_kv_cache_store_plugin.dll",
                                   "../../samples/c/kv_cache_store_plugin/sample_kv_cache_store_plugin.dll",
                               }
#elif defined(__APPLE__)
                               {
                                   "../../samples/c/kv_cache_store_plugin/libsample_kv_cache_store_plugin.dylib",
                               }
#else
                               {
                                   "../../samples/c/kv_cache_store_plugin/libsample_kv_cache_store_plugin.so",
                               }
#endif
    );
}

}  // namespace ov::genai::test
