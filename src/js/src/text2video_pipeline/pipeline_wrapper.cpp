// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2video_pipeline/pipeline_wrapper.hpp"

#include <future>

#include "include/addon.hpp"
#include "include/helper.hpp"
#include "include/text2video_pipeline/init_worker.hpp"

struct Text2VideoTsfnContext {
    Text2VideoTsfnContext(std::string prompt, std::shared_ptr<bool> is_generating)
        : prompt(prompt),
          is_generating(is_generating) {};
    ~Text2VideoTsfnContext() {};

    std::thread native_thread;
    Napi::ThreadSafeFunction callback;

    std::string prompt;
    std::shared_ptr<bool> is_generating;
    std::shared_ptr<ov::genai::Text2VideoPipeline> pipe = nullptr;
    std::shared_ptr<ov::AnyMap> properties = nullptr;
};

void text2videoPerformInferenceThread(Text2VideoTsfnContext* context) {
    auto report_error = [context](const std::string& message) {
        auto status = context->callback.BlockingCall([message](Napi::Env env, Napi::Function jsCallback) {
            try {
                jsCallback.Call(
                    {Napi::Error::New(env, "text2videoPerformInferenceThread error. " + message).Value(), env.Null()});
            } catch (std::exception& err) {
                std::cerr << "The callback failed when attempting to return an error from "
                             "text2videoPerformInferenceThread. Details:\n"
                          << err.what() << std::endl;
                std::cerr << "Original error message:\n" << message << std::endl;
            }
        });
        if (status != napi_ok) {
            std::cerr << "The BlockingCall failed with status " << status
                      << " when trying to return an error from text2videoPerformInferenceThread." << std::endl;
            std::cerr << "Original error message:\n" << message << std::endl;
        }
    };
    auto finalize = [context]() {
        *context->is_generating = false;
        context->callback.Release();
    };
    try {
        ov::genai::VideoGenerationResult result = context->pipe->generate(context->prompt, *context->properties);

        napi_status status =
            context->callback.BlockingCall([result, &report_error](Napi::Env env, Napi::Function jsCallback) {
                try {
                    auto result_obj = cpp_to_js<ov::genai::VideoGenerationResult, Napi::Value>(env, result);

                    jsCallback.Call({
                        env.Null(),
                        result_obj
                    });
                } catch (std::exception& err) {
                    report_error("The final callback failed. Details:\n" + std::string(err.what()));
                }
            });

        if (status != napi_ok) {
            report_error("The final BlockingCall failed with status " + std::to_string(status));
        }
        finalize();
    } catch (std::exception& e) {
        report_error(e.what());
        finalize();
    }
}

Text2VideoPipelineWrapper::Text2VideoPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Text2VideoPipelineWrapper>(info) {};

Napi::Function Text2VideoPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Text2VideoPipeline",
                       {InstanceMethod("init", &Text2VideoPipelineWrapper::init),
                        InstanceMethod("generate", &Text2VideoPipelineWrapper::generate),
                        InstanceMethod("getGenerationConfig", &Text2VideoPipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &Text2VideoPipelineWrapper::set_generation_config)});
}

Napi::Value Text2VideoPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!*this->is_initializing, "Pipeline is already initializing");
        VALIDATE_ARGS_COUNT(info, 4, "init()");
        const std::string model_path = js_to_cpp<std::string>(env, info[0]);
        const std::string device = js_to_cpp<std::string>(env, info[1]);
        const auto& properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        Text2VideoInitWorker* asyncWorker =
            new Text2VideoInitWorker(callback, this->pipe, this->is_initializing, model_path, device, properties);
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value Text2VideoPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2VideoPipeline is not initialized");
        OPENVINO_ASSERT(!*this->is_generating, "Another generation is already in progress");
        *this->is_generating = true;
        VALIDATE_ARGS_COUNT(info, 3, "generate()");
        Text2VideoTsfnContext* context = nullptr;

        // Arguments: prompt, properties, callback
        auto prompt = js_to_cpp<std::string>(env, info[0]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[1]);
        OPENVINO_ASSERT(info[2].IsFunction(), "generate callback is not a function");
        auto callback = info[2].As<Napi::Function>();

        context = new Text2VideoTsfnContext(prompt, this->is_generating);
        context->pipe = this->pipe;
        context->properties = std::make_shared<ov::AnyMap>(properties);

        context->callback =
            Napi::ThreadSafeFunction::New(env,
                                          callback,                            // JavaScript function called asynchronously
                                          "T2V_generate_callback",             // Name
                                          0,                                   // Unlimited queue
                                          1,                                   // Only one thread will use this initially
                                          [context, this](Napi::Env) {         // Finalizer used to clean threads up
                                              context->native_thread.join();
                                              delete context;
                                          });
        context->native_thread = std::thread(text2videoPerformInferenceThread, context);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        *this->is_generating = false;
    }
    return env.Undefined();
}

Napi::Value Text2VideoPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2VideoPipeline is not initialized");
        const auto& config = this->pipe->get_generation_config();

        return cpp_to_js<ov::genai::VideoGenerationConfig, Napi::Value>(env, config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value Text2VideoPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2VideoPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        OPENVINO_ASSERT(info[0].IsObject(), "setGenerationConfig expects an object argument");

        auto config = js_to_cpp<ov::genai::VideoGenerationConfig>(env, info[0]);
        this->pipe->set_generation_config(config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}
