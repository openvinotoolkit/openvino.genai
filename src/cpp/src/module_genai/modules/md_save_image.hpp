// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "module_genai/module.hpp"

namespace ov {
namespace genai {
namespace module {

class SaveImageModule : public IBaseModule {
    DeclareModuleConstructor(SaveImageModule);

private:
    bool initialize();
    std::string m_filename_prefix;
    std::string m_output_folder;
    std::atomic<size_t> m_sequence_number{0};

    std::string generate_filename();
    std::vector<std::string> save_tensor_as_image(const ov::Tensor& tensor, const std::string& filepath);
};

REGISTER_MODULE_CONFIG(SaveImageModule);

} // namespace module
} // namespace genai
} // namespace ov
