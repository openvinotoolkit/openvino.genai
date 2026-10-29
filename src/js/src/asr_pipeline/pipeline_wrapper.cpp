// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/asr_pipeline/pipeline_wrapper.hpp"

#include "include/base/inference_thread.hpp"
#include "include/tokenizer.hpp"

ASRPipelineWrapper::ASRPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ASRPipelineWrapper>(info) {}

Napi::Function ASRPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "ASRPipeline",
                       {InstanceMethod("init", &ASRPipelineWrapper::init),
                        InstanceMethod("generate", &ASRPipelineWrapper::generate),
                        InstanceMethod("getTokenizer", &ASRPipelineWrapper::get_tokenizer),
                        InstanceMethod("getGenerationConfig", &ASRPipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &ASRPipelineWrapper::set_generation_config)});
}

Napi::Value ASRPipelineWrapper::init(const Napi::CallbackInfo& info) {
    return initialize_pipeline(info, pipe, state);
}

Napi::Value ASRPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "ASRPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::GENERATE, "Another generate is already in progress");
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        auto raw_speech = js_to_cpp<std::vector<float>>(env, info[0]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[1]);
        OPENVINO_ASSERT(info[2].IsFunction() || info[2].IsUndefined(), "streamer callback is not a function");
        OPENVINO_ASSERT(info[3].IsFunction(), "generate callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "asr_inference_thread");
        if (info[2].IsFunction()) {
            context->streamer =
                std::make_unique<ThreadSafeCallback>(env, info[2].As<Napi::Function>(), "ASR_streamer");
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
            ov::genai::AudioInputs audio_inputs{std::move(raw_speech)};
            auto result = pipe->generate(audio_inputs, config, streamer);
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return to_asr_decoded_result(env, result);
            };
        };
        start_inference(info, info[3].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}

Napi::Value ASRPipelineWrapper::get_tokenizer(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipe, "ASRPipeline is not initialized");
        return TokenizerWrapper::wrap(info.Env(), pipe->get_tokenizer());
    });
}

Napi::Value ASRPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    return get_pipeline_generation_config(info, pipe, state, "ASRPipeline");
}

Napi::Value ASRPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    return set_pipeline_generation_config(info, pipe, state, "ASRPipeline");
}
