// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/whisper_pipeline/pipeline_wrapper.hpp"

#include "include/base/inference_thread.hpp"
#include "include/tokenizer.hpp"

WhisperPipelineWrapper::WhisperPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<WhisperPipelineWrapper>(info) {}

Napi::Function WhisperPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "WhisperPipeline",
                       {InstanceMethod("init", &WhisperPipelineWrapper::init),
                        InstanceMethod("generate", &WhisperPipelineWrapper::generate),
                        InstanceMethod("getTokenizer", &WhisperPipelineWrapper::get_tokenizer),
                        InstanceMethod("getGenerationConfig", &WhisperPipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &WhisperPipelineWrapper::set_generation_config)});
}

Napi::Value WhisperPipelineWrapper::init(const Napi::CallbackInfo& info) {
    return initialize_pipeline(info, pipe, state);
}

Napi::Value WhisperPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "WhisperPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::GENERATE, "Another generate is already in progress");
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        auto raw_speech = js_to_cpp<std::vector<float>>(env, info[0]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[1]);
        OPENVINO_ASSERT(info[2].IsFunction() || info[2].IsUndefined(), "streamer callback is not a function");
        OPENVINO_ASSERT(info[3].IsFunction(), "generate callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "whisperPerformInferenceThread");
        if (info[2].IsFunction()) {
            context->streamer =
                std::make_unique<ThreadSafeCallback>(env, info[2].As<Napi::Function>(), "Whisper_streamer");
        }
        context->run_generate = [context = context.get(),
                                 pipe = pipe,
                                 raw_speech = std::move(raw_speech),
                                 properties = std::move(properties)]() mutable -> JsResultProducer {
            auto config = pipe->get_generation_config();
            config.update_generation_config(properties);
            ov::genai::StreamerVariant streamer = std::monostate();
            if (context->streamer) {
                streamer = make_text_streamer(context);
            }
            auto result = pipe->generate(raw_speech, config, streamer);
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return to_whisper_decoded_result(env, result);
            };
        };
        start_inference(info, info[3].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}

Napi::Value WhisperPipelineWrapper::get_tokenizer(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipe, "WhisperPipeline is not initialized");
        return TokenizerWrapper::wrap(info.Env(), pipe->get_tokenizer());
    });
}

Napi::Value WhisperPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    return get_pipeline_generation_config(info, pipe, state, "WhisperPipeline");
}

Napi::Value WhisperPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    return set_pipeline_generation_config(info, pipe, state, "WhisperPipeline");
}
