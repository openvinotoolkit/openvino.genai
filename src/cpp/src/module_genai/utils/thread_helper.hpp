// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <future>
#include <chrono>
#include <thread>
#include "openvino/runtime/compiled_model.hpp"

namespace ov::genai::module::thread_utils {

inline std::future<bool> load_model_weights_async(ov::CompiledModel& compiled_model) {
    auto load_fun = [&]() -> bool {
        // compiled_model.load_model_weights();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return true;
    };
    return std::async(std::launch::async, load_fun);
}

inline void load_model_weights_finish(std::future<bool>& result_future) {
    result_future.get();
}

inline std::future<bool> release_model_weights_async(ov::CompiledModel& compiled_model) {
    auto load_fun = [&]() -> bool {
        // compiled_model.release_model_weights();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return true;
    };
    return std::async(std::launch::async, load_fun);
}

inline void release_model_weights_finish(std::future<bool>& result_future) {
    result_future.get();
}

}  // namespace ov::genai::module::thread_utils