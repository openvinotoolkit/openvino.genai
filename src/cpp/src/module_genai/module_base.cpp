// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/module_base.hpp"

#include <filesystem>

namespace ov {
namespace genai {
namespace module {

namespace fs = std::filesystem;

IBaseModule::IBaseModule(const IBaseModuleDesc::PTR& desc) : module_desc(desc) {
    for (auto& input : desc->inputs) {
        this->inputs[input.name] = InputModule();
        this->inputs[input.name].parent_port_name = input.source_module_out_name;
    }
    for (auto& output : desc->outputs) {
        this->outputs[output.name] = OutputModule();
    }
}

void IBaseModule::prepare_inputs() {
    for (auto& input : this->inputs) {
        const auto& parent_port_name = input.second.parent_port_name;
        if (input.second.module_ptr.lock() != nullptr) {
            input.second.data = input.second.module_ptr.lock()->outputs[parent_port_name].data;
        }
    }
}

const std::string& IBaseModule::get_module_name() const {
    return module_desc->name;
}

std::string IBaseModuleDesc::get_full_path(const std::string& fn) {
    // Check if fn is absolute path or file exists
    if (fs::exists(fn) || fs::path(fn).is_absolute()) {
        return fn;
    }

    fs::path joined_path = config_root_path / fn;
    if (fs::exists(joined_path) || fs::path(joined_path).is_absolute()) {
        return joined_path.string();
    }
    OPENVINO_ASSERT(false, "File path is invalid: " + fn);
}

}  // namespace module
}  // namespace genai
}  // namespace ov
