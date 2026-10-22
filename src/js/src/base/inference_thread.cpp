// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/base/inference_thread.hpp"

#include <cmath>
#include <iostream>

#include "openvino/core/except.hpp"

void InferenceThreadContext::record_callback_exception(std::exception_ptr exception) {
    std::lock_guard<std::mutex> lock(m_exception_mutex);
    if (!m_callback_exception) {
        m_callback_exception = std::move(exception);
    }
}

std::exception_ptr InferenceThreadContext::callback_exception() {
    std::lock_guard<std::mutex> lock(m_exception_mutex);
    return m_callback_exception;
}

ov::genai::StreamingStatus to_streaming_status(const Napi::Value& value) {
    if (value.IsUndefined()) {
        return ov::genai::StreamingStatus::RUNNING;
    }
    OPENVINO_ASSERT(value.IsNumber(), "Streamer must return StreamingStatus or undefined");
    const double status = value.As<Napi::Number>().DoubleValue();
    OPENVINO_ASSERT(std::isfinite(status) && std::floor(status) == status &&
                        (status == static_cast<int>(ov::genai::StreamingStatus::RUNNING) ||
                         status == static_cast<int>(ov::genai::StreamingStatus::STOP) ||
                         status == static_cast<int>(ov::genai::StreamingStatus::CANCEL) ||
                         status == static_cast<int>(ov::genai::StreamingStatus::TOOL_CALL_STOP)),
                    "Streamer returned an invalid StreamingStatus");
    return static_cast<ov::genai::StreamingStatus>(static_cast<int>(status));
}

std::function<ov::genai::StreamingStatus(std::string)> make_text_streamer(InferenceThreadContext* context) {
    return [context](std::string word) {
        try {
            return context->streamer->call_and_wait<ov::genai::StreamingStatus>(
                [word = std::move(word)](Napi::Env env, Napi::Function callback) {
                    return callback.Call({Napi::String::New(env, word)});
                },
                to_streaming_status);
        } catch (const std::exception&) {
            context->record_callback_exception(std::current_exception());
            return ov::genai::StreamingStatus::CANCEL;
        }
    };
}

namespace {

void perform_generate_thread(InferenceThreadContext* context) {
    JsResultProducer produce;
    std::exception_ptr exception;
    try {
        produce = context->run_generate();
        exception = context->callback_exception();
    } catch (const std::exception&) {
        exception = context->callback_exception();
        if (!exception) {
            exception = std::current_exception();
        }
    }
    const napi_status status = context->callback->call(
        [context, produce = std::move(produce), exception](Napi::Env env, Napi::Function callback) {
            complete_pipeline_callback(env,
                                       callback,
                                       produce,
                                       context->operation,
                                       exception,
                                       context->thread_name + " error. ");
        });
    if (status != napi_ok && status != napi_closing) {
        std::cerr << "Failed to deliver " << context->thread_name << " completion. Status: " << status << std::endl;
    }
    if (context->streamer) {
        context->streamer->release();
    }
    if (context->speech_streamer) {
        context->speech_streamer->release();
    }
    context->callback->release();
}

}  // namespace

void start_inference(const Napi::CallbackInfo& info,
                     const Napi::Function& callback,
                     std::unique_ptr<InferenceThreadContext> context) {
    auto owner = std::shared_ptr<InferenceThreadContext>(std::move(context));
    OPENVINO_ASSERT(owner->run_generate, "InferenceThreadContext.run_generate is not set");
    owner->callback = std::make_unique<ThreadSafeCallback>(info.Env(),
                                                            callback,
                                                            owner->thread_name,
                                                            true,
                                                            [owner] {
                                                                if (owner->native_thread.joinable()) {
                                                                    owner->native_thread.join();
                                                                }
                                                            });
    try {
        owner->native_thread = std::thread(perform_generate_thread, owner.get());
    } catch (const std::exception&) {
        owner->operation.reset();
        owner->callback->release();
        throw;
    }
}
