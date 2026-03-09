// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/frontend/extension.hpp>
#include <openvino/openvino.hpp>

#include "openvino/genai/extensions.hpp"
#include "utils.hpp"

using namespace ov::genai::utils;

static std::filesystem::path get_custom_extension_library_name() {
#if defined(_WIN32)
    return "openvino_custom_add_extension.dll";
#elif defined(__APPLE__)
    return "openvino_custom_add_extension.dylib";
#else
    return "openvino_custom_add_extension.so";
#endif
}

TEST(TestAddExtensions, test_extract_extensions) {
    const std::filesystem::path extension_library_name = get_custom_extension_library_name();
    ov::AnyMap properties_path = {ov::genai::extensions(std::vector<std::filesystem::path>{extension_library_name})};
    auto op = std::make_shared<ov::frontend::OpExtension<>>("Relu", "MyRelu");
    ov::AnyMap properties_op = {ov::genai::extensions(std::vector<std::shared_ptr<ov::Extension>>{op})};
    ov::genai::ExtensionList extensionList_path{extension_library_name};
    ov::genai::ExtensionList extensionList_op{op};

    EXPECT_EQ(extract_extensions(properties_path), extensionList_path);
    EXPECT_EQ(extract_extensions(properties_op), extensionList_op);
}

TEST(TestAddExtensions, test_extract_extensions_to_core) {
    const std::filesystem::path extension_library_name = get_custom_extension_library_name();
    ov::AnyMap properties_path = {ov::genai::extensions(std::vector<std::filesystem::path>{extension_library_name})};
    ov::AnyMap properties_op = {ov::genai::extensions(
        std::vector<std::shared_ptr<ov::Extension>>{std::make_shared<ov::frontend::OpExtension<>>("Relu", "MyRelu")})};

    EXPECT_NO_THROW(extract_extensions_to_core(properties_path));
    EXPECT_NO_THROW(extract_extensions_to_core(properties_op));
}
