// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/module_type.hpp"

#include <unordered_map>

namespace ov {
namespace genai {
namespace module {

const std::unordered_map<ModuleType, std::string> ModuleTypeConverter::kTypeToString = {
#define X(name, val) {ModuleType::name, #name},
    GENAI_MODULE_TYPE_LIST
#undef X
};

const std::unordered_map<std::string, ModuleType> ModuleTypeConverter::kStringToType = {
#define X(name, val) {#name, ModuleType::name},
    GENAI_MODULE_TYPE_LIST
#undef X
};

const std::unordered_map<ThreadMode, std::string> ThreadModeConverter::kModeToString = {
    {ThreadMode::AUTO, "AUTO"},
    {ThreadMode::AUTO, "auto"},
    {ThreadMode::SYNC, "SYNC"},
    {ThreadMode::SYNC, "sync"},
    {ThreadMode::ASYNC, "ASYNC"},
    {ThreadMode::ASYNC, "async"},
};
const std::unordered_map<std::string, ThreadMode> ThreadModeConverter::kStringToMode =
    ThreadModeConverter::create_string_to_mode_map();

}  // namespace module
}  // namespace genai
}  // namespace ov