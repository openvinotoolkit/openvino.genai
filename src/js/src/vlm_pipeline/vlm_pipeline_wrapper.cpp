// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/vlm_pipeline/vlm_pipeline_wrapper.hpp"

#include "include/base/inference_thread.hpp"
#include "include/tokenizer.hpp"

VLMPipelineWrapper::VLMPipelineWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<VLMPipelineWrapper>(info) {}

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
    return initialize_pipeline(info, pipe, state);
}

Napi::Value VLMPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "VLMPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::GENERATE, "Another generation is already in progress");
        VALIDATE_ARGS_COUNT(info, 6, "generate()");
        auto inputs = js_to_cpp<VLMGenerateInputs>(env, info[0]);
        auto images = js_to_cpp<std::vector<ov::Tensor>>(env, info[1]);
        auto videos = js_to_cpp<std::vector<ov::Tensor>>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction() || info[3].IsUndefined(), "streamer must be a function or undefined");
        auto properties = js_to_cpp<ov::AnyMap>(env, info[4]);
        OPENVINO_ASSERT(info[5].IsFunction(), "generate callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "vlmPerformInferenceThread");
        if (info[3].IsFunction()) {
            context->streamer = std::make_unique<ThreadSafeCallback>(env, info[3].As<Napi::Function>(), "VLM_streamer");
        }
        context->run_generate = [context = context.get(),
                                 pipe = pipe,
                                 inputs = std::move(inputs),
                                 images = std::move(images),
                                 videos = std::move(videos),
                                 properties = std::move(properties)]() mutable -> JsResultProducer {
            ov::genai::GenerationConfig config;
            config.update_generation_config(properties);
            ov::genai::StreamerVariant streamer = std::monostate();
            if (context->streamer) {
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
        start_inference(info, info[5].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}

Napi::Value VLMPipelineWrapper::start_chat(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "VLMPipeline is not initialized");
        auto operation =
            std::make_shared<PipelineOperationGuard>(state, PipelineOperation::ACCESS, "Pipeline is busy");
        VALIDATE_ARGS_COUNT(info, 2, "startChat()");
        auto system_message = js_to_cpp<std::string>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "startChat callback is not a function");
        queue_pipeline_worker(info,
                              info[1].As<Napi::Function>(),
                              std::move(operation),
                              [pipe = pipe, system_message = std::move(system_message)] {
                                  pipe->start_chat(system_message);
                                  return [](Napi::Env env) -> Napi::Value {
                                      return env.Undefined();
                                  };
                              });
        return env.Undefined();
    });
}

Napi::Value VLMPipelineWrapper::finish_chat(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipe, "VLMPipeline is not initialized");
        auto operation =
            std::make_shared<PipelineOperationGuard>(state, PipelineOperation::ACCESS, "Pipeline is busy");
        VALIDATE_ARGS_COUNT(info, 1, "finishChat()");
        OPENVINO_ASSERT(info[0].IsFunction(), "finishChat callback is not a function");
        queue_pipeline_worker(info, info[0].As<Napi::Function>(), std::move(operation), [pipe = pipe] {
            pipe->finish_chat();
            return [](Napi::Env env) -> Napi::Value {
                return env.Undefined();
            };
        });
        return info.Env().Undefined();
    });
}

Napi::Value VLMPipelineWrapper::get_tokenizer(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipe, "VLMPipeline is not initialized");
        return TokenizerWrapper::wrap(info.Env(), pipe->get_tokenizer());
    });
}

Napi::Value VLMPipelineWrapper::set_chat_template(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipe, "VLMPipeline is not initialized");
        const PipelineOperationGuard operation(state, PipelineOperation::ACCESS, "Pipeline is busy");
        VALIDATE_ARGS_COUNT(info, 1, "setChatTemplate()");
        pipe->set_chat_template(js_to_cpp<std::string>(info.Env(), info[0]));
        return info.Env().Undefined();
    });
}

Napi::Value VLMPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    return get_pipeline_generation_config(info, pipe, state, "VLMPipeline");
}

Napi::Value VLMPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    return set_pipeline_generation_config(info, pipe, state, "VLMPipeline");
}
