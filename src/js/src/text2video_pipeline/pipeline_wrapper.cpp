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
                    auto result_obj = Napi::Object::New(env);

                    // Convert video tensor
                    result_obj.Set("video", cpp_to_js<ov::Tensor, Napi::Value>(env, result.video));

                    // Convert performance metrics to a plain JS object
                    auto perf_obj = Napi::Object::New(env);
                    auto& perf = const_cast<ov::genai::VideoGenerationPerfMetrics&>(result.performance_stat);

                    perf_obj.Set("loadTime", Napi::Number::New(env, perf.get_load_time()));
                    perf_obj.Set("generateDuration", Napi::Number::New(env, perf.get_generate_duration()));

                    auto iter_dur = perf.get_iteration_duration();
                    auto iter_obj = Napi::Object::New(env);
                    iter_obj.Set("mean", Napi::Number::New(env, iter_dur.mean));
                    iter_obj.Set("std", Napi::Number::New(env, iter_dur.std));
                    perf_obj.Set("iterationDuration", iter_obj);

                    auto trans_dur = perf.get_transformer_infer_duration();
                    auto trans_obj = Napi::Object::New(env);
                    trans_obj.Set("mean", Napi::Number::New(env, trans_dur.mean));
                    trans_obj.Set("std", Napi::Number::New(env, trans_dur.std));
                    perf_obj.Set("transformerInferDuration", trans_obj);

                    perf_obj.Set("vaeDecoderInferDuration",
                                 Napi::Number::New(env, perf.get_vae_decoder_infer_duration()));

                    result_obj.Set("perfMetrics", perf_obj);

                    jsCallback.Call({
                        env.Null(),  // Error should be null in normal case
                        result_obj   // Return result object
                    });
                } catch (std::exception& err) {
                    report_error("The final callback failed. Details:\n" + std::string(err.what()));
                }
            });

        if (status != napi_ok) {
            report_error("The final BlockingCall failed with status " + status);
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

        auto config_obj = Napi::Object::New(env);
        config_obj.Set("guidance_scale", Napi::Number::New(env, config.guidance_scale));
        config_obj.Set("height", Napi::Number::New(env, config.height));
        config_obj.Set("width", Napi::Number::New(env, config.width));
        config_obj.Set("num_inference_steps", Napi::Number::New(env, config.num_inference_steps));
        config_obj.Set("num_videos_per_prompt", Napi::Number::New(env, static_cast<double>(config.num_videos_per_prompt)));
        config_obj.Set("num_frames", Napi::Number::New(env, static_cast<double>(config.num_frames)));
        config_obj.Set("max_sequence_length", Napi::Number::New(env, config.max_sequence_length));

        if (config.negative_prompt.has_value()) {
            config_obj.Set("negative_prompt", Napi::String::New(env, config.negative_prompt.value()));
        }
        if (config.guidance_rescale.has_value()) {
            config_obj.Set("guidance_rescale", Napi::Number::New(env, config.guidance_rescale.value()));
        }
        if (config.frame_rate.has_value()) {
            config_obj.Set("frame_rate", Napi::Number::New(env, config.frame_rate.value()));
        }

        return config_obj;
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

        auto js_obj = info[0].As<Napi::Object>();
        auto config = this->pipe->get_generation_config();

        if (js_obj.Has("negative_prompt") && !js_obj.Get("negative_prompt").IsUndefined()) {
            config.negative_prompt = js_to_cpp<std::string>(env, js_obj.Get("negative_prompt"));
        }
        if (js_obj.Has("guidance_scale") && !js_obj.Get("guidance_scale").IsUndefined()) {
            config.guidance_scale = js_obj.Get("guidance_scale").As<Napi::Number>().FloatValue();
        }
        if (js_obj.Has("height") && !js_obj.Get("height").IsUndefined()) {
            config.height = js_to_cpp<int64_t>(env, js_obj.Get("height"));
        }
        if (js_obj.Has("width") && !js_obj.Get("width").IsUndefined()) {
            config.width = js_to_cpp<int64_t>(env, js_obj.Get("width"));
        }
        if (js_obj.Has("num_inference_steps") && !js_obj.Get("num_inference_steps").IsUndefined()) {
            config.num_inference_steps = js_to_cpp<int64_t>(env, js_obj.Get("num_inference_steps"));
        }
        if (js_obj.Has("num_videos_per_prompt") && !js_obj.Get("num_videos_per_prompt").IsUndefined()) {
            config.num_videos_per_prompt =
                static_cast<size_t>(js_obj.Get("num_videos_per_prompt").As<Napi::Number>().Int64Value());
        }
        if (js_obj.Has("num_frames") && !js_obj.Get("num_frames").IsUndefined()) {
            config.num_frames = static_cast<size_t>(js_obj.Get("num_frames").As<Napi::Number>().Int64Value());
        }
        if (js_obj.Has("max_sequence_length") && !js_obj.Get("max_sequence_length").IsUndefined()) {
            config.max_sequence_length = js_obj.Get("max_sequence_length").As<Napi::Number>().Int32Value();
        }
        if (js_obj.Has("guidance_rescale") && !js_obj.Get("guidance_rescale").IsUndefined()) {
            config.guidance_rescale = js_obj.Get("guidance_rescale").As<Napi::Number>().FloatValue();
        }
        if (js_obj.Has("frame_rate") && !js_obj.Get("frame_rate").IsUndefined()) {
            config.frame_rate = js_obj.Get("frame_rate").As<Napi::Number>().FloatValue();
        }

        this->pipe->set_generation_config(config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}
