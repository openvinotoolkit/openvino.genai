// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/image_generation_inference_thread.hpp"

#include <exception>
#include <future>
#include <string>

#include <napi.h>

#include "include/helper.hpp"

namespace {

// Invokes the JS step callback and resolves result_promise with the value it returns (directly or via a Promise).
void invoke_js_step_callback(Napi::Env env,
                             Napi::Function js_callback,
                             size_t step,
                             size_t num_steps,
                             ov::Tensor& latent,
                             const std::shared_ptr<std::promise<bool>>& result_promise,
                             InferenceThreadContext* context) {
    auto record_and_stop = [result_promise, context](const std::string& message) {
        context->callback_exceptions.push_back(message);
        result_promise->set_value(true);  // stop generation on any callback failure
    };
    try {
        auto js_result = js_callback.Call({Napi::Number::New(env, static_cast<double>(step)),
                                           Napi::Number::New(env, static_cast<double>(num_steps)),
                                           cpp_to_js<ov::Tensor, Napi::Value>(env, latent)});
        if (js_result.IsBoolean()) {
            result_promise->set_value(js_result.As<Napi::Boolean>().Value());
        } else if (js_result.IsPromise()) {
            Napi::Object promise = js_result.As<Napi::Object>();
            Napi::Function then = promise.Get("then").As<Napi::Function>();
            auto on_fulfilled = Napi::Function::New(env, [result_promise, record_and_stop](const Napi::CallbackInfo& cb) {
                if (cb.Length() > 0 && cb[0].IsBoolean()) {
                    result_promise->set_value(cb[0].As<Napi::Boolean>().Value());
                } else {
                    record_and_stop("Step callback must resolve to a boolean.");
                }
            });
            auto on_rejected = Napi::Function::New(env, [record_and_stop](const Napi::CallbackInfo& cb) {
                std::string message = "Step callback promise rejected";
                if (cb.Length() > 0 && cb[0].IsObject()) {
                    Napi::Value msg = cb[0].As<Napi::Object>().Get("message");
                    if (msg.IsString()) {
                        message = msg.As<Napi::String>().Utf8Value();
                    }
                }
                record_and_stop(message);
            });
            then.Call(promise, {on_fulfilled, on_rejected});
        } else {
            record_and_stop("Step callback must return a boolean or a Promise<boolean>.");
        }
    } catch (const std::exception& err) {
        record_and_stop(err.what());
    }
}

}  // namespace

std::function<bool(size_t, size_t, ov::Tensor&)> make_image_generation_step_callback(
    InferenceThreadContext* context,
    std::shared_ptr<std::atomic<bool>> is_busy) {
    return [context, is_busy](size_t step, size_t num_steps, ov::Tensor& latent) -> bool {
        auto result_promise = std::make_shared<std::promise<bool>>();
        auto result_future = result_promise->get_future();
        // Release the inference request while the JS step callback runs so it may call decode().
        is_busy->store(false);
        napi_status status = context->streamer_tsfn->BlockingCall(
            [step, num_steps, &latent, result_promise, context](Napi::Env env, Napi::Function js_callback) {
                invoke_js_step_callback(env, js_callback, step, num_steps, latent, result_promise, context);
            });
        if (status != napi_ok) {
            is_busy->store(true);
            context->callback_exceptions.push_back("Step callback BlockingCall failed with status: " +
                                                   std::to_string(static_cast<int>(status)));
            return true;  // stop
        }
        bool stop = result_future.get();
        is_busy->store(true);
        return stop;
    };
}
