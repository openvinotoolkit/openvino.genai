// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>

namespace ov::genai::common_bindings::utils {

template <typename T, typename U>
std::vector<float> get_ms(const T& instance, U T::*member) {
    // Converts c++ duration to float so that it can be used in API bindings.
    std::vector<float> res;
    const auto& durations = instance.*member;
    res.reserve(durations.size());
    std::transform(durations.begin(), durations.end(), std::back_inserter(res), [](const auto& duration) {
        return duration.count();
    });
    return res;
}

template <typename T, typename U>
std::vector<double> timestamp_to_ms(const T& instance, U T::*member) {
    // Converts c++ duration to double so that it can be used in API bindings.
    // Use double instead of float bacuse timestamp in ms contains 14 digits
    // while float only allows to store ~7 significant digits.
    // And the current timestamp (number of secs from 1970) is already 11 digits.
    std::vector<double> res;
    const auto& timestamps = instance.*member;
    res.reserve(timestamps.size());
    std::transform(timestamps.begin(), timestamps.end(), std::back_inserter(res), [](const auto& timestamp) {
        return std::chrono::duration<double, std::milli>(timestamp.time_since_epoch()).count();
    });
    return res;
}

}  // namespace ov::genai::common_bindings::utils
