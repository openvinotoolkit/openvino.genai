// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/genai/extensions.hpp"
#include "utils.hpp"

using namespace ov::genai::utils;

TEST(TestAddExtensions, test_extract_extensions) {
    ov::AnyMap properties1 = {
        ov::genai::extensions(std::vector<std::filesystem::path>{"path_extension1", "path_extension2"})};
    ov::AnyMap properties2 = {
        ov::genai::extensions(std::vector<std::shared_ptr<ov::Extension>>{nullptr, nullptr})};
    ov::genai::ExtensionList extensionList1{"path_extension1", "path_extension2"};
    ov::genai::ExtensionList extensionList2{nullptr, nullptr};

    EXPECT_EQ(extract_extensions(properties1), extensionList1);
    EXPECT_EQ(extract_extensions(properties2), extensionList2);
}

TEST(TestAddExtensions, test_extract_extensions_to_core) {
    // Use intentionally non-existent, platform-agnostic extension paths to trigger error handling.
    ov::AnyMap properties1 = {ov::genai::extensions(
        std::vector<std::filesystem::path>{"non_existent_extension1", "non_existent_extension2"})};
    ov::AnyMap properties2 = {ov::genai::extensions(std::vector<std::shared_ptr<ov::Extension>>{})};

    EXPECT_THROW(extract_extensions_to_core(properties1), ov::Exception);
    EXPECT_NO_THROW(extract_extensions_to_core(properties2));
}
