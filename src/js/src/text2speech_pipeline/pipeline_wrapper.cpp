// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2speech_pipeline/pipeline_wrapper.hpp"

#include <iostream>
#include <memory>
#include <thread>

#include "include/base/inference_thread.hpp"
#include "include/helper.hpp"
#include "include/text2speech_pipeline/init_worker.hpp"

Text2SpeechPipelineWrapper::Text2SpeechPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Text2SpeechPipelineWrapper>(info) {}

Napi::Function Text2SpeechPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Text2SpeechPipeline",
                       {
                           InstanceMethod("init", &Text2SpeechPipelineWrapper::init),
                           InstanceMethod("generate", &Text2SpeechPipelineWrapper::generate),
                           InstanceMethod("getGenerationConfig", &Text2SpeechPipelineWrapper::get_generation_config),
                           InstanceMethod("setGenerationConfig", &Text2SpeechPipelineWrapper::set_generation_config),
                       });
}

Napi::Value Text2SpeechPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!this->is_initializing->load(), "Pipeline is already initializing");
        this->is_initializing->store(true);

        VALIDATE_ARGS_COUNT(info, 4, "init()");
        auto model_path = js_to_cpp<std::string>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        auto async_worker = new Text2SpeechInitWorker(callback,
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

Napi::Value Text2SpeechPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2SpeechPipeline is not initialized");
        OPENVINO_ASSERT(!this->is_generating->load(), "Another generate is already in progress");
        this->is_generating->store(true);

        VALIDATE_ARGS_COUNT(info, 4, "generate()");

        auto inputs = js_to_cpp<ov::genai::StringInputs>(env, info[0]);

        ov::Tensor speaker_embedding;

        if (!info[1].IsNull() && !info[1].IsUndefined()) {
            speaker_embedding = js_to_cpp<ov::Tensor>(env, info[1]);
        }

        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);

        OPENVINO_ASSERT(info[3].IsFunction(), "generate callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        auto* context = new InferenceThreadContext(this->is_generating,
                                                   "text2speechPerformInferenceThread",
                                                   "Streamer exceptions occurred:");
        auto pipe = this->pipe;
        context->run_generate = [pipe, inputs = std::move(inputs), speaker_embedding = std::move(speaker_embedding),
                                 properties = std::move(properties)]() mutable -> JsResultProducer {
            ov::genai::Text2SpeechDecodedResults result;
            std::visit(overloaded{[&](const std::string& text) {
                                      result = pipe->generate(text, speaker_embedding, properties);
                                  },
                                  [&](const std::vector<std::string>& texts) {
                                      result = pipe->generate(texts, speaker_embedding, properties);
                                  }},
                       inputs);
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return to_text2speech_decoded_result(env, result);
            };
        };

        context->callback_tsfn =
            Napi::ThreadSafeFunction::New(env, callback, "Text2Speech_generate_callback", 0, 1, [context](Napi::Env) {
                if (context->native_thread.joinable()) {
                    context->native_thread.join();
                }
                delete context;
            });

        context->native_thread = std::thread(perform_generate_thread, context);
    } catch (const std::exception& ex) {
        this->is_generating->store(false);
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value Text2SpeechPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2SpeechPipeline is not initialized");
        auto config = this->pipe->get_generation_config();
        return cpp_to_js<ov::genai::SpeechGenerationConfig, Napi::Value>(env, config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value Text2SpeechPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2SpeechPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        if (info[0].IsUndefined() || info[0].IsNull()) {
            OPENVINO_THROW("Generation config cannot be undefined or null");
        }
        this->pipe->set_generation_config(js_to_cpp<ov::genai::SpeechGenerationConfig>(env, info[0]));
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}
