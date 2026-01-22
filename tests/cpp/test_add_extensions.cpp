// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/genai/extensions.hpp"
#include "utils.hpp"

using namespace ov::genai::utils;

TEST(TestAddExtensions, test_extract_extensions) {
    ov::AnyMap properties = {
        ov::genai::extensions(std::vector<std::filesystem::path>{"path_extension1", "path_extension2"})};
    ov::genai::PathExtensions pathExtensions{"path_extension1", "path_extension2"};

    EXPECT_EQ(extract_extensions(properties), pathExtensions);
}

TEST(TestAddExtensions, test_add_extensions_to_core) {
    // Use intentionally non-existent, platform-agnostic extension paths to trigger error handling.
    ov::AnyMap properties2 = {ov::genai::extensions(
        std::vector<std::filesystem::path>{"non_existent_extension1", "non_existent_extension2"})};
    ov::AnyMap properties3 = {ov::genai::extensions(std::vector<std::shared_ptr<ov::Extension>>{})};

    auto extensions2 = extract_extensions(properties2);
    auto extensions3 = extract_extensions(properties3);
    EXPECT_THROW(add_extensions_to_core(extensions2), ov::Exception);
    EXPECT_NO_THROW(add_extensions_to_core(extensions3));
}
