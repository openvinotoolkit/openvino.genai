// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <string>
#include <string_view>

#include "openvino/core/except.hpp"

namespace ov {
namespace genai {
namespace module {

// Single source of truth for DataType names/values.
// Used to generate both the enum and the string conversion tables.
#ifndef OV_GENAI_MODULE_DATA_TYPE_LIST
#    define OV_GENAI_MODULE_DATA_TYPE_LIST(X) \
        X(Unknown, 0)                         \
        X(OVTensor, 1)                        \
        X(VecOVTensor, 2)                     \
        X(OVRemoteTensor, 3)                  \
        X(VecOVRemoteTensor, 4)               \
        X(String, 10)                         \
        X(VecString, 11)                      \
        X(Int, 20)                            \
        X(VecInt, 21)                         \
        X(VecVecInt, 22)                      \
        X(Float, 30)                          \
        X(VecFloat, 31)
#endif

enum class DataType : int {
#define OV_GENAI_MODULE_DATA_TYPE_ENUM_ITEM(name, value) name = value,
    OV_GENAI_MODULE_DATA_TYPE_LIST(OV_GENAI_MODULE_DATA_TYPE_ENUM_ITEM)
#undef OV_GENAI_MODULE_DATA_TYPE_ENUM_ITEM
};

struct DataTypeConverter {
private:
    static const std::unordered_map<DataType, std::string> kTypeToString;

    static const std::unordered_map<std::string, DataType> kStringToType;

    static std::unordered_map<std::string, DataType> create_string_to_type_map() {
        std::unordered_map<std::string, DataType> map;
        for (const auto& pair : kTypeToString) {
            map[pair.second] = pair.first;
        }
        return map;
    }

public:
    static std::string toString(DataType type) {
        auto it = kTypeToString.find(type);
        OPENVINO_ASSERT(it != kTypeToString.end(), "Unknown DataType value: " + std::to_string(static_cast<int>(type)));
        return it->second;
    }

    static DataType fromString(const std::string& str) {
        std::string_view sv{str};
        const auto begin = sv.find_first_not_of(" \t\n\r");
        const auto end = sv.find_last_not_of(" \t\n\r");
        OPENVINO_ASSERT(begin != std::string_view::npos, "Unknown DataType string: <empty>");
        sv = sv.substr(begin, end - begin + 1);

        auto it = kStringToType.find(std::string(sv));
        OPENVINO_ASSERT(it != kStringToType.end(), "Unknown DataType string: " + std::string(sv));
        return it->second;
    }
};

inline std::string to_string(DataType type) {
    return DataTypeConverter::toString(type);
}

inline DataType data_type_from_string(const std::string& str) {
    return DataTypeConverter::fromString(str);
}

}  // namespace module
}  // namespace genai
}  // namespace ov
