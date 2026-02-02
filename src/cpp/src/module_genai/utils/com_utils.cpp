// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "com_utils.hpp"

namespace ov::genai::module {
namespace utils {

bool check_env_variable(const std::string& var_name) {
    const char* env_p = std::getenv(var_name.c_str());
    if (env_p != nullptr && (std::string(env_p) == "true" || std::string(env_p) == "TRUE" ||
                             std::string(env_p) == "1" || std::string(env_p) == "True")) {
        return true;
    }

    return false;
}

}  // namespace utils
}  // namespace ov::genai::module