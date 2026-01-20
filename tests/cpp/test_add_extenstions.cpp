// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/genai/extensions.hpp"
#include "utils.hpp"

using namespace ov::genai::utils;
using map_type = std::map<std::string, int64_t>;

TEST(TestAddExtensions, test_extract_extensions) {
    ov::AnyMap properties = {
        ov::genai::extensions(std::vector<std::filesystem::path>{"path_extension1", "path_extension2"})};
    ov::genai::PathExtensions pathExtentions{"path_extension1", "path_extension2"};

    EXPECT_EQ(extract_extensions(properties), pathExtentions);
}

TEST(TestAddExtensions, test_add_extensions_to_core) {
    std::string exe_path = "";
#ifdef __linux__
    exe_path = "../openvino_genai/libopenvino_tokenizers.so";
#elif defined(_WIN32)
    exe_path = "../openvino_genai/libopenvino_tokenizers.dll";
#endif
    if (!exe_path.empty()) {
        ov::AnyMap properties1 = {ov::genai::extensions(std::vector<std::filesystem::path>{exe_path.c_str()})};
        auto extensions1 = extract_extensions(properties1);
        EXPECT_NO_THROW(add_extensions_to_core(extensions1));
    }
    // Use intentionally non-existent, platform-agnostic extension paths to trigger error handling.
    ov::AnyMap properties2 = {ov::genai::extensions(
        std::vector<std::filesystem::path>{"non_existent_extension1", "non_existent_extension2"})};
    ov::AnyMap properties3 = {ov::genai::extensions(std::vector<std::shared_ptr<ov::Extension>>{})};

    auto extensions2 = extract_extensions(properties2);
    auto extensions3 = extract_extensions(properties3);
    EXPECT_THROW(add_extensions_to_core(extensions2), ov::Exception);
    EXPECT_NO_THROW(add_extensions_to_core(extensions3));
}
