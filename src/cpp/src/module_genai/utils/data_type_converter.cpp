// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/utils/data_type_converter.hpp"

#include <unordered_map>

namespace ov {
namespace genai {

namespace module {

const std::unordered_map<DataType, std::string> DataTypeConverter::kTypeToString = {
    {DataType::Unknown, "Unknown"},
    {DataType::OVTensor, "OVTensor"},
    {DataType::VecOVTensor, "VecOVTensor"},
    {DataType::OVRemoteTensor, "OVRemoteTensor"},
    {DataType::VecOVRemoteTensor, "VecOVRemoteTensor"},
    {DataType::String, "String"},
    {DataType::VecString, "VecString"},
    {DataType::Int, "Int"},
    {DataType::VecInt, "VecInt"},
    {DataType::VecVecInt, "VecVecInt"},
    {DataType::Float, "Float"},
    {DataType::VecFloat, "VecFloat"}};

const std::unordered_map<std::string, DataType> DataTypeConverter::kStringToType =
    DataTypeConverter::create_string_to_type_map();

}  // namespace module
}  // namespace genai
}  // namespace ov