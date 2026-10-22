// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include <napi.h>

#include "include/base/pipeline_state.hpp"
#include "include/base/thread_safe_callback.hpp"
#include "include/helper.hpp"

using JsResultProducer = std::function<Napi::Value(Napi::Env)>;

template <typename Method>
Napi::Value with_pipeline_errors(const Napi::CallbackInfo& info, Method method) {
    try {
        return method();
    } catch (const Napi::Error& error) {
        error.ThrowAsJavaScriptException();
    } catch (const std::exception&) {
        callback_error(info.Env(), std::current_exception()).ThrowAsJavaScriptException();
    }
    return info.Env().Undefined();
}

inline void complete_pipeline_callback(Napi::Env env,
                                       const Napi::Function& callback,
                                       const JsResultProducer& produce,
                                       std::shared_ptr<PipelineOperationGuard>& operation,
                                       std::exception_ptr exception = {},
                                       const std::string& prefix = {}) {
    Napi::Value result = env.Undefined();
    Napi::Value error = env.Null();
    if (exception) {
        error = callback_error(env, exception, prefix).Value();
    } else {
        try {
            result = produce(env);
            if (env.IsExceptionPending()) {
                error = env.GetAndClearPendingException().Value();
            }
        } catch (const Napi::Error& conversion_error) {
            error = conversion_error.Value();
        } catch (const std::exception&) {
            error = callback_error(env, std::current_exception(), prefix).Value();
        }
    }
    operation.reset();
    callback.Call({error, error.IsNull() ? result : env.Null()});
}

class PipelineWorker final : public Napi::AsyncWorker {
public:
    PipelineWorker(const Napi::Object& receiver,
                   const Napi::Function& callback,
                   std::shared_ptr<PipelineOperationGuard> operation,
                   std::function<JsResultProducer()> execute)
        : Napi::AsyncWorker(receiver, callback),
          m_operation(std::move(operation)),
          m_execute(std::move(execute)),
          m_completion(std::make_unique<ThreadSafeCallback>(callback.Env(),
                                                            callback,
                                                            "GenAI_pipeline_worker",
                                                            true,
                                                            std::function<void()>{},
                                                            receiver)) {}

    void Execute() override {
        JsResultProducer produce;
        std::exception_ptr exception;
        try {
            produce = m_execute();
        } catch (const std::exception&) {
            exception = std::current_exception();
        }
        const napi_status status = m_completion->call(
            [produce = std::move(produce), operation = std::move(m_operation), exception](
                Napi::Env env,
                Napi::Function callback) mutable {
                complete_pipeline_callback(env, callback, produce, operation, exception);
            });
        if (status != napi_ok && status != napi_closing) {
            std::cerr << "Failed to deliver a GenAI pipeline worker completion. Status: " << status << std::endl;
        }
        m_completion->release();
    }

    void OnWorkComplete(Napi::Env, napi_status) override {
        // libuv also completes work during environment shutdown, when JavaScript calls are forbidden.
        Destroy();
    }

private:
    std::shared_ptr<PipelineOperationGuard> m_operation;
    std::function<JsResultProducer()> m_execute;
    std::unique_ptr<ThreadSafeCallback> m_completion;
};

inline void queue_pipeline_worker(const Napi::CallbackInfo& info,
                                  const Napi::Function& callback,
                                  std::shared_ptr<PipelineOperationGuard> operation,
                                  std::function<JsResultProducer()> execute) {
    auto worker = std::make_unique<PipelineWorker>(info.This().As<Napi::Object>(),
                                                    callback,
                                                    std::move(operation),
                                                    std::move(execute));
    worker->Queue();
    worker.release();
}

template <typename Pipeline, typename Factory>
void queue_pipeline_init(const Napi::CallbackInfo& info,
                         const Napi::Function& callback,
                         std::shared_ptr<PipelineOperationGuard> operation,
                         std::shared_ptr<Pipeline>& destination,
                         Factory factory) {
    // PipelineWorker retains the receiver until publication on the JavaScript thread.
    queue_pipeline_worker(info, callback, std::move(operation), [target = &destination, factory = std::move(factory)] {
        auto pipeline = factory();
        return [target, pipeline = std::move(pipeline)](Napi::Env env) -> Napi::Value {
            *target = pipeline;
            return env.Undefined();
        };
    });
}

template <typename Pipeline>
Napi::Value initialize_pipeline(const Napi::CallbackInfo& info,
                                std::shared_ptr<Pipeline>& pipeline,
                                const std::shared_ptr<PipelineState>& state) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(!pipeline, "Pipeline is already initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::INITIALIZE, "Pipeline is already initializing");
        VALIDATE_ARGS_COUNT(info, 4, "init()");
        auto model_path = js_to_cpp<std::filesystem::path>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        queue_pipeline_init(info,
                            info[3].As<Napi::Function>(),
                            std::move(operation),
                            pipeline,
                            [model_path = std::move(model_path),
                             device = std::move(device),
                             properties = std::move(properties)] {
                                return std::make_shared<Pipeline>(model_path, device, properties);
                            });
        return env.Undefined();
    });
}

template <typename Pipeline>
Napi::Value get_pipeline_generation_config(const Napi::CallbackInfo& info,
                                           const std::shared_ptr<Pipeline>& pipeline,
                                           const std::shared_ptr<PipelineState>& state,
                                           const std::string& name) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipeline, name, " is not initialized");
        const PipelineOperationGuard operation(state, PipelineOperation::ACCESS, "Pipeline is busy");
        const auto& config = pipeline->get_generation_config();
        return cpp_to_js<std::decay_t<decltype(config)>, Napi::Value>(info.Env(), config);
    });
}

template <typename Pipeline>
Napi::Value set_pipeline_generation_config(const Napi::CallbackInfo& info,
                                           const std::shared_ptr<Pipeline>& pipeline,
                                           const std::shared_ptr<PipelineState>& state,
                                           const std::string& name) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipeline, name, " is not initialized");
        const PipelineOperationGuard operation(state, PipelineOperation::ACCESS, "Pipeline is busy");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        OPENVINO_ASSERT(!info[0].IsUndefined() && !info[0].IsNull(), "Generation config cannot be undefined or null");
        OPENVINO_ASSERT(info[0].IsObject() && !info[0].IsArray(), "Generation config must be an object");
        using Config = std::decay_t<decltype(pipeline->get_generation_config())>;
        if constexpr (std::is_same_v<Config, ov::genai::ASRGenerationConfig>) {
            // ASR already merges partial updates to preserve required stopping criteria.
            auto config = pipeline->get_generation_config();
            config.update_generation_config(js_to_cpp<ov::AnyMap>(info.Env(), info[0]));
            pipeline->set_generation_config(config);
        } else {
            pipeline->set_generation_config(js_to_cpp<Config>(info.Env(), info[0]));
        }
        return info.Env().Undefined();
    });
}

template <typename Pipeline>
Napi::Value get_pipeline_performance_metrics(const Napi::CallbackInfo& info,
                                            const std::shared_ptr<Pipeline>& pipeline,
                                            const std::shared_ptr<PipelineState>& state,
                                            const std::string& name) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipeline, name, " is not initialized");
        const PipelineOperationGuard operation(state, PipelineOperation::ACCESS, "Pipeline is busy");
        const auto& metrics = pipeline->get_performance_metrics();
        return cpp_to_js<std::decay_t<decltype(metrics)>, Napi::Value>(info.Env(), metrics);
    });
}
