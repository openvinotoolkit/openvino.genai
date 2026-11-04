// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "include/base/async_worker.hpp"
#include "openvino/genai/streamer_base.hpp"

struct InferenceThreadContext {
    InferenceThreadContext(std::shared_ptr<PipelineOperationGuard> operation, std::string thread_name)
        : operation(std::move(operation)),
          thread_name(std::move(thread_name)) {}

    void record_callback_exception(std::exception_ptr exception);
    std::exception_ptr callback_exception();

    std::thread native_thread;
    std::unique_ptr<ThreadSafeCallback> callback;
    std::unique_ptr<ThreadSafeCallback> streamer;
    std::unique_ptr<ThreadSafeCallback> speech_streamer;
    std::shared_ptr<PipelineOperationGuard> operation;
    std::function<JsResultProducer()> run_generate;
    std::string thread_name;

private:
    std::mutex m_exception_mutex;
    std::exception_ptr m_callback_exception;
};

std::function<ov::genai::StreamingStatus(std::string)> make_text_streamer(InferenceThreadContext* context);
ov::genai::StreamingStatus to_streaming_status(const Napi::Value& value);

void start_inference(const Napi::CallbackInfo& info,
                     const Napi::Function& callback,
                     std::unique_ptr<InferenceThreadContext> context);
