// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2image_pipeline/pipeline_wrapper.hpp"

#include <filesystem>
#include <future>
#include <iostream>
#include <optional>
#include <thread>

#include "include/helper.hpp"
#include "include/text2image_pipeline/init_worker.hpp"

namespace {

struct Text2ImageTsfnContext {
    Text2ImageTsfnContext(std::string prompt,
                          ov::AnyMap generation_properties,
                          std::shared_ptr<std::atomic<bool>> is_generating)
        : prompt(std::move(prompt)),
          generation_properties(std::move(generation_properties)),
          is_generating(std::move(is_generating)) {}

    std::thread native_thread;
    Napi::ThreadSafeFunction callback_tsfn;
    std::optional<Napi::ThreadSafeFunction> streamer_tsfn;
    std::string prompt;
    ov::AnyMap generation_properties;
    std::vector<std::string> callback_exceptions;
    std::shared_ptr<std::atomic<bool>> is_generating;
    std::shared_ptr<ov::genai::Text2ImagePipeline> pipe = nullptr;
};

void text2image_perform_inference_thread(Text2ImageTsfnContext* context) {
    auto report_error = [context](const std::string& message) {
        auto status = context->callback_tsfn.BlockingCall([message](Napi::Env env, Napi::Function js_callback) {
            try {
                js_callback.Call({Napi::Error::New(env, "text2image_perform_inference_thread error. " + message).Value(),
                                  env.Null()});
            } catch (const std::exception& err) {
                std::cerr << "The callback failed when returning an error from text2image_perform_inference_thread. Details:\n"
                          << err.what() << std::endl;
                std::cerr << "Original error message:\n" << message << std::endl;
            }
        });
        if (status != napi_ok) {
            std::cerr << "The BlockingCall failed with status " << status
                      << " when trying to return an error from text2image_perform_inference_thread." << std::endl;
            std::cerr << "Original error message:\n" << message << std::endl;
        }
    };
    auto finalize = [context]() {
        context->callback_tsfn.Release();
        if (context->streamer_tsfn.has_value()) {
            context->streamer_tsfn->Release();
        }
    };

    try {
        if (context->streamer_tsfn.has_value()) {
            context->generation_properties["callback"] =
                std::function<bool(size_t, size_t, ov::Tensor&)>(
                    [context](size_t step, size_t num_steps, ov::Tensor& latent) -> bool {
                        std::promise<bool> result_promise;
                        napi_status status = context->streamer_tsfn->BlockingCall(
                            [step, num_steps, &result_promise, context](
                                Napi::Env env, Napi::Function js_callback) {
                                try {
                                    auto js_result =
                                        js_callback.Call({Napi::Number::New(env, static_cast<double>(step)),
                                                          Napi::Number::New(env, static_cast<double>(num_steps))});
                                    result_promise.set_value(js_result.IsBoolean() &&
                                                             js_result.As<Napi::Boolean>().Value());
                                } catch (const std::exception& err) {
                                    context->callback_exceptions.push_back(err.what());
                                    result_promise.set_value(true);  // stop on exception
                                }
                            });
                        if (status != napi_ok) {
                            context->callback_exceptions.push_back(
                                "Step callback BlockingCall failed with status: " +
                                std::to_string(static_cast<int>(status)));
                            return true;  // stop
                        }
                        return result_promise.get_future().get();
                    });
        }

        ov::Tensor result = context->pipe->generate(context->prompt, context->generation_properties);
        context->is_generating->store(false);

        if (!context->callback_exceptions.empty()) {
            std::string combined_error = "Step callback exceptions occurred:\n";
            for (size_t i = 0; i < context->callback_exceptions.size(); ++i) {
                combined_error += "[" + std::to_string(i + 1) + "] " + context->callback_exceptions[i] + "\n";
            }
            report_error(combined_error);
        } else {
            auto final_result = std::make_shared<ov::Tensor>(std::move(result));
            auto final_callback_error = std::make_shared<std::string>();

            napi_status status = context->callback_tsfn.BlockingCall(
                [final_result, final_callback_error](Napi::Env env, Napi::Function js_callback) {
                    try {
                        js_callback.Call({env.Null(), cpp_to_js<ov::Tensor, Napi::Value>(env, *final_result)});
                    } catch (const std::exception& err) {
                        *final_callback_error = "The final callback failed. Details:\n" + std::string(err.what());
                    }
                });
            if (status != napi_ok) {
                report_error("The final BlockingCall failed with status " +
                             std::to_string(static_cast<int>(status)));
            } else if (!final_callback_error->empty()) {
                report_error(*final_callback_error);
            }
        }
    } catch (const std::exception& ex) {
        context->is_generating->store(false);
        report_error(ex.what());
    }

    finalize();
}

}  // namespace

Text2ImagePipelineWrapper::Text2ImagePipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Text2ImagePipelineWrapper>(info) {}

Napi::Function Text2ImagePipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Text2ImagePipeline",
                       {
                           InstanceMethod("init", &Text2ImagePipelineWrapper::init),
                           InstanceMethod("generate", &Text2ImagePipelineWrapper::generate),
                           InstanceMethod("getPerformanceMetrics",
                                          &Text2ImagePipelineWrapper::get_performance_metrics),
                           InstanceMethod("getGenerationConfig", &Text2ImagePipelineWrapper::get_generation_config),
                           InstanceMethod("setGenerationConfig", &Text2ImagePipelineWrapper::set_generation_config),
                       });
}

Napi::Value Text2ImagePipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!this->is_initializing->load(), "Pipeline is already initializing");
        this->is_initializing->store(true);

        VALIDATE_ARGS_COUNT(info, 4, "init()");
        auto model_path = js_to_cpp<std::filesystem::path>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        auto async_worker = new Text2ImageInitWorker(callback,
                                                     this->pipe,
                                                     this->is_initializing,
                                                     std::move(model_path),
                                                     std::move(device),
                                                     std::move(properties));
        async_worker->Queue();
    } catch (const std::exception& ex) {
        this->is_initializing->store(false);
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value Text2ImagePipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2ImagePipeline is not initialized");
        OPENVINO_ASSERT(!this->is_generating->load(), "Another generate is already in progress");
        this->is_generating->store(true);

        // generate(prompt, properties, streamer, doneCallback)
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        OPENVINO_ASSERT(info[0].IsString(), "Text2ImagePipeline.generate() expects a prompt string.");
        OPENVINO_ASSERT(info[3].IsFunction(), "generate done callback is not a function");

        auto prompt = js_to_cpp<std::string>(env, info[0]);
        auto generation_properties = js_to_cpp<ov::AnyMap>(env, info[1]);
        Napi::Function callback = info[3].As<Napi::Function>();

        auto* context =
            new Text2ImageTsfnContext(std::move(prompt), std::move(generation_properties), this->is_generating);
        context->pipe = this->pipe;

        // streamer at index 2 can be a function or null/undefined
        OPENVINO_ASSERT(info[2].IsUndefined() || info[2].IsNull() || info[2].IsFunction(),
                        "Text2ImagePipeline.generate() expects streamer to be a function, null or undefined.");
        if (!info[2].IsUndefined() && !info[2].IsNull() && info[2].IsFunction()) {
            context->streamer_tsfn =
                Napi::ThreadSafeFunction::New(env, info[2].As<Napi::Function>(), "Text2Image_streamer", 0, 1);
        }

        context->callback_tsfn =
            Napi::ThreadSafeFunction::New(env, callback, "Text2Image_generate_callback", 0, 1, [context](Napi::Env) {
                if (context->native_thread.joinable()) {
                    context->native_thread.join();
                }
                delete context;
            });

        context->native_thread = std::thread(text2image_perform_inference_thread, context);
    } catch (const std::exception& ex) {
        this->is_generating->store(false);
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value Text2ImagePipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2ImagePipeline is not initialized");
        return cpp_to_js<ov::genai::ImageGenerationConfig, Napi::Value>(env, this->pipe->get_generation_config());
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value Text2ImagePipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2ImagePipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        OPENVINO_ASSERT(!info[0].IsUndefined() && !info[0].IsNull(), "Generation config cannot be undefined or null");
        this->pipe->set_generation_config(js_to_cpp<ov::genai::ImageGenerationConfig>(env, info[0]));
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value Text2ImagePipelineWrapper::get_performance_metrics(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2ImagePipeline is not initialized");
        return cpp_to_js<ov::genai::ImageGenerationPerfMetrics, Napi::Value>(
            env,
            this->pipe->get_performance_metrics());
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}
