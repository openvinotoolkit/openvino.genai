// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/base/inference_thread.hpp"

#include <exception>
#include <future>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"

namespace {

// Reports an error from the inference thread back to JS through the callback ThreadSafeFunction.
void report_inference_thread_error(InferenceThreadContext* context, const std::string& message) {
    auto status = context->callback_tsfn.BlockingCall(
        [thread_name = context->thread_name, message](Napi::Env env, Napi::Function js_callback) {
            try {
                js_callback.Call({Napi::Error::New(env, thread_name + " error. " + message).Value(), env.Null()});
            } catch (const std::exception& err) {
                std::cerr << "The callback failed when returning an error from " << thread_name << ". Details:\n"
                          << err.what() << std::endl;
                std::cerr << "Original error message:\n" << message << std::endl;
            }
        });
    if (status != napi_ok) {
        std::cerr << "The BlockingCall failed with status " << status << " when trying to return an error from "
                  << context->thread_name << "." << std::endl;
        std::cerr << "Original error message:\n" << message << std::endl;
    }
}

// Releases the ThreadSafeFunctions held by the context once the inference thread is done.
void release_inference_thread_safe_functions(InferenceThreadContext* context) {
    if (context->streamer_tsfn.has_value()) {
        context->streamer_tsfn->Release();
    }
    // callback_tsfn's finalizer joins the worker thread and deletes the context, so its Release() must be the last
    // action that touches the context.
    context->callback_tsfn.Release();
}

// Joins the exceptions collected from JS callbacks into a single message prefixed with the given header.
std::string combine_callback_exceptions(const std::vector<std::string>& exceptions, const std::string& header) {
    std::string combined = header + "\n";
    for (size_t i = 0; i < exceptions.size(); ++i) {
        combined += "[" + std::to_string(i + 1) + "] " + exceptions[i] + "\n";
    }
    return combined;
}

// Delivers the final result to JS through the callback ThreadSafeFunction. produce_js builds the Napi value on the
// Node.js thread. Any failure is reported through report_inference_thread_error.
void report_inference_thread_result(InferenceThreadContext* context, JsResultProducer produce_js) {
    auto producer = std::make_shared<JsResultProducer>(std::move(produce_js));
    auto final_callback_error = std::make_shared<std::string>();
    napi_status status = context->callback_tsfn.BlockingCall(
        [producer, final_callback_error](Napi::Env env, Napi::Function js_callback) {
            try {
                js_callback.Call({env.Null(), (*producer)(env)});
            } catch (const std::exception& err) {
                *final_callback_error = "The final callback failed. Details:\n" + std::string(err.what());
            }
        });
    if (status != napi_ok) {
        report_inference_thread_error(
            context, "The final BlockingCall failed with status " + std::to_string(static_cast<int>(status)));
    } else if (!final_callback_error->empty()) {
        report_inference_thread_error(context, *final_callback_error);
    }
}

}  // namespace

std::function<ov::genai::StreamingStatus(std::string)> make_text_streamer(InferenceThreadContext* context) {
    return [context](std::string word) -> ov::genai::StreamingStatus {
        std::promise<ov::genai::StreamingStatus> result_promise;
        napi_status status = context->streamer_tsfn->BlockingCall(
            [word, &result_promise, context](Napi::Env env, Napi::Function js_callback) {
                try {
                    auto callback_result = js_callback.Call({Napi::String::New(env, word)});
                    if (callback_result.IsNumber()) {
                        result_promise.set_value(
                            static_cast<ov::genai::StreamingStatus>(callback_result.As<Napi::Number>().Int32Value()));
                    } else {
                        result_promise.set_value(ov::genai::StreamingStatus::RUNNING);
                    }
                } catch (const std::exception& err) {
                    context->callback_exceptions.push_back(err.what());
                    result_promise.set_value(ov::genai::StreamingStatus::CANCEL);
                }
            });
        if (status != napi_ok) {
            context->callback_exceptions.push_back("The streamer callback BlockingCall failed with status: " +
                                                   std::to_string(static_cast<int>(status)));
            return ov::genai::StreamingStatus::CANCEL;
        }
        return result_promise.get_future().get();
    };
}

void perform_generate_thread(InferenceThreadContext* context) {
    try {
        OPENVINO_ASSERT(context->run_generate, "InferenceThreadContext.run_generate is not set");
        JsResultProducer produce_js = context->run_generate();
        context->on_finished();
        *context->is_generating = false;

        if (!context->callback_exceptions.empty()) {
            report_inference_thread_error(
                context,
                combine_callback_exceptions(context->callback_exceptions, context->streamer_exception_header));
        } else {
            report_inference_thread_result(context, std::move(produce_js));
        }
    } catch (const std::exception& error) {
        context->on_finished();
        *context->is_generating = false;
        report_inference_thread_error(context, error.what());
    }
    release_inference_thread_safe_functions(context);
}
