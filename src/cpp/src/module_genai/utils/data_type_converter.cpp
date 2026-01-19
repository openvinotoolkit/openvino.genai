// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/module_data_type.hpp"

#include <unordered_map>

namespace ov {
namespace genai {

namespace module {

const std::unordered_map<DataType, std::string> DataTypeConverter::kTypeToString = {
#define OV_GENAI_MODULE_DATA_TYPE_TO_STRING_ITEM(name, value) {DataType::name, #name},
    OV_GENAI_MODULE_DATA_TYPE_LIST(OV_GENAI_MODULE_DATA_TYPE_TO_STRING_ITEM)
#undef OV_GENAI_MODULE_DATA_TYPE_TO_STRING_ITEM
};

const std::unordered_map<std::string, DataType> DataTypeConverter::kStringToType = []() {
    auto map = DataTypeConverter::create_string_to_type_map();
    return map;
}();

}  // namespace module
}  // namespace genai
}  // namespace ov