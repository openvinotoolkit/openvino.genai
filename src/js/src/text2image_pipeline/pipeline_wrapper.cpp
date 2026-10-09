// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2image_pipeline/pipeline_wrapper.hpp"

#include <filesystem>
#include <thread>

#include "include/helper.hpp"
#include "include/image_decode_worker.hpp"
#include "include/image_generation_inference_thread.hpp"
#include "include/text2image_pipeline/init_worker.hpp"

Text2ImagePipelineWrapper::Text2ImagePipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Text2ImagePipelineWrapper>(info) {}

Napi::Function Text2ImagePipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Text2ImagePipeline",
                       {
                           InstanceMethod("init", &Text2ImagePipelineWrapper::init),
                           InstanceMethod("generate", &Text2ImagePipelineWrapper::generate),
                           InstanceMethod("decode", &Text2ImagePipelineWrapper::decode),
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
        OPENVINO_ASSERT(!this->is_busy->load() && !this->is_generating->load(),
                        "generate() cannot run while another generate() or decode() is in progress");
        this->is_busy->store(true);
        this->is_generating->store(true);

        // generate(prompt, properties, streamer, doneCallback)
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        OPENVINO_ASSERT(info[0].IsString(), "Text2ImagePipeline.generate() expects a prompt string.");
        OPENVINO_ASSERT(info[3].IsFunction(), "generate done callback is not a function");

        auto prompt = js_to_cpp<std::string>(env, info[0]);
        auto generation_properties = js_to_cpp<ov::AnyMap>(env, info[1]);
        Napi::Function callback = info[3].As<Napi::Function>();

        auto pipe = this->pipe;
        auto is_busy = this->is_busy;
        auto* context = new InferenceThreadContext(this->is_generating, "text2image_perform_inference_thread");
        context->streamer_exception_header = "Step callback exceptions occurred:";
        context->on_finished = [is_busy] {
            is_busy->store(false);
        };
        context->run_generate = [context, pipe, is_busy, prompt = std::move(prompt),
                                 properties = std::move(generation_properties)]() mutable -> JsResultProducer {
            if (context->streamer_tsfn.has_value()) {
                properties["callback"] = make_image_generation_step_callback(context, is_busy);
            }
            ov::Tensor result = pipe->generate(prompt, properties);
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return cpp_to_js<ov::Tensor, Napi::Value>(env, result);
            };
        };

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

        context->native_thread = std::thread(perform_generate_thread, context);
    } catch (const std::exception& ex) {
        this->is_busy->store(false);
        this->is_generating->store(false);
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value Text2ImagePipelineWrapper::decode(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2ImagePipeline is not initialized");
        OPENVINO_ASSERT(!this->is_busy->load(),
                        "decode() cannot run while another generate() or decode() is in progress");
        VALIDATE_ARGS_COUNT(info, 2, "decode()");
        auto latent = js_to_cpp<ov::Tensor>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "decode callback is not a function");
        Napi::Function callback = info[1].As<Napi::Function>();

        this->is_busy->store(true);
        auto* async_worker =
            new ImageDecodeWorker<ov::genai::Text2ImagePipeline>(callback, this->pipe, std::move(latent), this->is_busy);
        async_worker->Queue();
    } catch (const std::exception& ex) {
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
