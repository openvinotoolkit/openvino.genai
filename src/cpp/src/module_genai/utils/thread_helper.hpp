// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <future>
#include <thread>

#include "openvino/runtime/compiled_model.hpp"
#include "profiler.hpp"

namespace ov::genai::module::thread_utils {

#ifndef ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
#    define ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT 0  // Current multiple threads may cause GPU crash.
#endif

#ifdef ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS
inline std::future<bool> load_model_weights_async(ov::CompiledModel compiled_model) {
    auto load_fun = [compiled_model]() mutable -> bool {
        PROFILE(pm, "load_model_weights async");
        compiled_model.load_model_weights();
        // infer_request = compiled_model.create_infer_request();
        return true;
    };
    return std::async(std::launch::async, std::move(load_fun));
}

inline std::future<bool> release_model_weights_async(ov::CompiledModel compiled_model, ov::InferRequest infer_request) {
    auto release_fun = [compiled_model, infer_request]() mutable -> bool {
        PROFILE(pm, "release_model_weights async");
        infer_request = ov::InferRequest();  // reset infer request to release the reference to the model weights
        compiled_model.release_model_weights();
        return true;
    };
    return std::async(std::launch::async, std::move(release_fun));
}
#endif

}  // namespace ov::genai::module::thread_utils