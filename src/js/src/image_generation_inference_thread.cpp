// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/image_generation_inference_thread.hpp"

std::function<bool(size_t, size_t, ov::Tensor&)> make_image_generation_step_callback(
    InferenceThreadContext* context,
    std::shared_ptr<PipelineState> state) {
    return [context, state = std::move(state)](size_t step, size_t num_steps, ov::Tensor& latent) {
        const PipelineStepGuard step_guard(state);
        try {
            return context->streamer->call_and_wait<bool>(
                [step, num_steps, latent](Napi::Env env, Napi::Function callback) {
                    return call_js_callback(callback,
                                            {Napi::Number::New(env, static_cast<double>(step)),
                                             Napi::Number::New(env, static_cast<double>(num_steps)),
                                             cpp_to_js<ov::Tensor, Napi::Value>(env, latent)});
                },
                [](const Napi::Value& value) {
                    OPENVINO_ASSERT(value.IsBoolean(), "Step callback must return a boolean or a Promise<boolean>");
                    return value.As<Napi::Boolean>().Value();
                },
                true);
        } catch (const std::exception&) {
            context->record_callback_exception(std::current_exception());
            return true;
        }
    };
}
