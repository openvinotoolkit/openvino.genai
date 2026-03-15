// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/frontend/extension.hpp>
#include <openvino/openvino.hpp>

#include "openvino/genai/extensions.hpp"
#include "utils.hpp"

using namespace ov::genai::utils;

TEST(TestAddExtensions, test_extract_extensions) {
    ov::AnyMap properties_path = {ov::genai::extensions(std::vector<std::filesystem::path>{"non_existent_path"})};
    auto op = std::make_shared<ov::frontend::OpExtension<>>("Relu", "MyRelu");
    ov::AnyMap properties_op = {ov::genai::extensions(std::vector<std::shared_ptr<ov::Extension>>{op})};
    ov::genai::ExtensionList extension_path{"non_existent_path"};
    ov::genai::ExtensionList extension_op{op};

    EXPECT_EQ(extract_extensions(properties_path), extension_path);
    EXPECT_EQ(extract_extensions(properties_op), extension_op);
}

TEST(TestAddExtensions, test_extract_extensions_to_core) {
    // Use intentionally non-existent, platform-agnostic extension paths to trigger error handling.
    ov::AnyMap properties_path = {ov::genai::extensions(std::vector<std::filesystem::path>{"non_existent_path"})};
    ov::AnyMap properties_op = {ov::genai::extensions(
        std::vector<std::shared_ptr<ov::Extension>>{std::make_shared<ov::frontend::OpExtension<>>("Relu", "MyRelu")})};

    EXPECT_THROW(extract_extensions_to_core(properties_path), ov::Exception);
    EXPECT_NO_THROW(extract_extensions_to_core(properties_op));
}
