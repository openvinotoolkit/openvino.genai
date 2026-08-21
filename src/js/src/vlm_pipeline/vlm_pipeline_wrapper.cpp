// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/vlm_pipeline/vlm_pipeline_wrapper.hpp"

#include "include/addon.hpp"
#include "include/base/inference_thread.hpp"
#include "include/helper.hpp"
#include "include/tokenizer.hpp"
#include "include/vlm_pipeline/finish_chat_worker.hpp"
#include "include/vlm_pipeline/init_worker.hpp"
#include "include/vlm_pipeline/perf_metrics.hpp"
#include "include/vlm_pipeline/start_chat_worker.hpp"

VLMPipelineWrapper::VLMPipelineWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<VLMPipelineWrapper>(info) {};

Napi::Function VLMPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "VLMPipeline",
                       {InstanceMethod("init", &VLMPipelineWrapper::init),
                        InstanceMethod("generate", &VLMPipelineWrapper::generate),
                        InstanceMethod("getTokenizer", &VLMPipelineWrapper::get_tokenizer),
                        InstanceMethod("startChat", &VLMPipelineWrapper::start_chat),
                        InstanceMethod("finishChat", &VLMPipelineWrapper::finish_chat),
                        InstanceMethod("setChatTemplate", &VLMPipelineWrapper::set_chat_template),
                        InstanceMethod("getGenerationConfig", &VLMPipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &VLMPipelineWrapper::set_generation_config)});
}

Napi::Value VLMPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!*this->is_initializing, "Pipeline is already initializing");
        VALIDATE_ARGS_COUNT(info, 4, "init()");
        auto model_path = js_to_cpp<std::filesystem::path>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        auto callback = info[3].As<Napi::Function>();

        auto* asyncWorker = new VLMInitWorker(callback,
                                              this->pipe,
                                              this->is_initializing,
                                              std::move(model_path),
                                              std::move(device),
                                              std::move(properties));
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value VLMPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "VLMPipeline is not initialized");
        OPENVINO_ASSERT(!*this->is_generating, "Another generation is already in progress");
        *this->is_generating = true;
        VALIDATE_ARGS_COUNT(info, 6, "generate()");

        // Arguments: prompt or ChatHistory, images, videos, streamer, generationConfig, callback
        auto inputs = js_to_cpp<VLMGenerateInputs>(env, info[0]);
        auto images = js_to_cpp<std::vector<ov::Tensor>>(env, info[1]);
        auto videos = js_to_cpp<std::vector<ov::Tensor>>(env, info[2]);
        auto streamer_arg = info[3];
        OPENVINO_ASSERT(streamer_arg.IsFunction() || streamer_arg.IsUndefined(),
                        "streamer must be a function or undefined");
        auto generation_config = js_to_cpp<ov::AnyMap>(env, info[4]);
        OPENVINO_ASSERT(info[5].IsFunction(), "generate callback is not a function");
        auto callback = info[5].As<Napi::Function>();

        auto* context =
            new InferenceThreadContext(this->is_generating, "vlmPerformInferenceThread", "Streamer exceptions occurred:");
        auto pipe = this->pipe;
        context->run_generate = [context, pipe, inputs = std::move(inputs), images = std::move(images),
                                 videos = std::move(videos),
                                 generation_config = std::move(generation_config)]() mutable -> JsResultProducer {
            ov::genai::GenerationConfig config;
            config.update_generation_config(generation_config);

            ov::genai::StreamerVariant streamer = std::monostate();
            if (context->streamer_tsfn.has_value()) {
                streamer = make_text_streamer(context);
            }

            ov::genai::VLMDecodedResults result;
            std::visit(overloaded{[&](const std::string& prompt) {
                                      result = pipe->generate(prompt, images, videos, config, streamer);
                                  },
                                  [&](const ov::genai::ChatHistory& history) {
                                      result = pipe->generate(history, images, videos, config, streamer);
                                  }},
                       inputs);

            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return to_vlm_decoded_result(env, result);
            };
        };

        context->callback_tsfn = Napi::ThreadSafeFunction::New(env,
                                                              callback,
                                                              "VLM_generate_callback",
                                                              0,
                                                              1,
                                                              [context](Napi::Env) {
                                                                  if (context->native_thread.joinable()) {
                                                                      context->native_thread.join();
                                                                  }
                                                                  delete context;
                                                              });
        if (!streamer_arg.IsUndefined()) {
            context->streamer_tsfn =
                Napi::ThreadSafeFunction::New(env, streamer_arg.As<Napi::Function>(), "VLM_generate_streamer", 0, 1);
        }
        context->native_thread = std::thread(perform_generate_thread, context);
    } catch (const std::exception& ex) {
        *this->is_generating = false;
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value VLMPipelineWrapper::start_chat(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "VLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 2, "startChat()");
        auto system_message = js_to_cpp<std::string>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "startChat callback is not a function");
        auto callback = info[1].As<Napi::Function>();

        VLMStartChatWorker* asyncWorker = new VLMStartChatWorker(callback, this->pipe, system_message);
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value VLMPipelineWrapper::finish_chat(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "VLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "finishChat()");
        OPENVINO_ASSERT(info[0].IsFunction(), "finishChat callback is not a function");
        Napi::Function callback = info[0].As<Napi::Function>();

        VLMFinishChatWorker* asyncWorker = new VLMFinishChatWorker(callback, this->pipe);
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value VLMPipelineWrapper::get_tokenizer(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "VLMPipeline is not initialized");
        auto tokenizer = this->pipe->get_tokenizer();
        return TokenizerWrapper::wrap(env, tokenizer);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value VLMPipelineWrapper::set_chat_template(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "VLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setChatTemplate()");
        auto chat_template = js_to_cpp<std::string>(env, info[0]);
        this->pipe->set_chat_template(chat_template);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value VLMPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "VLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        ov::genai::GenerationConfig config = js_to_cpp<ov::genai::GenerationConfig>(env, info[0]);
        this->pipe->set_generation_config(config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value VLMPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "VLMPipeline is not initialized");
        ov::genai::GenerationConfig config = this->pipe->get_generation_config();
        return cpp_to_js<ov::genai::GenerationConfig, Napi::Value>(env, config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}
