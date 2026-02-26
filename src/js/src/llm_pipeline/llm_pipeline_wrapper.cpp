#include "include/llm_pipeline/llm_pipeline_wrapper.hpp"

#include <future>

#include "include/addon.hpp"
#include "include/helper.hpp"
#include "include/llm_pipeline/finish_chat_worker.hpp"
#include "include/llm_pipeline/init_worker.hpp"
#include "include/llm_pipeline/start_chat_worker.hpp"
#include "include/perf_metrics.hpp"
#include "include/tokenizer.hpp"

struct TsfnContext {
    TsfnContext(GenerateInputs inputs, std::shared_ptr<bool> is_generating)
        : inputs(inputs),
          is_generating(is_generating) {};
    ~TsfnContext() {};

    std::thread native_thread;
    Napi::ThreadSafeFunction generate_tsfn;
    std::optional<Napi::ThreadSafeFunction> streamer_tsfn;

    GenerateInputs inputs;
    std::shared_ptr<bool> is_generating;
    std::shared_ptr<ov::genai::LLMPipeline> pipe = nullptr;
    std::shared_ptr<ov::AnyMap> generation_config = nullptr;
    std::shared_ptr<ov::AnyMap> options = nullptr;
};

void performInferenceThread(TsfnContext* context) {
    auto report_error = [context](const std::string& message) {
        auto status = context->generate_tsfn.BlockingCall([message](Napi::Env env, Napi::Function jsCallback) {
            try {
                jsCallback.Call(
                    {Napi::Error::New(env, "performInferenceThread error. " + message).Value(), env.Null()});
            } catch (std::exception& err) {
                std::cerr << "The callback failed when attempting to return an error from performInferenceThread. "
                             "Details:\n"
                          << err.what() << std::endl;
                std::cerr << "Original error message:\n" << message << std::endl;
            }
        });
        if (status != napi_ok) {
            std::cerr << "The BlockingCall failed with status " << status
                      << " when trying to return an error from performInferenceThread." << std::endl;
            std::cerr << "Original error message:\n" << message << std::endl;
        }
    };
    auto finalize = [context]() {
        context->generate_tsfn.Release();
        if (context->streamer_tsfn.has_value()) {
            context->streamer_tsfn->Release();
        }
    };
    std::vector<std::string> streamer_exceptions;
    ov::genai::DecodedResults result;
    // Run inference
    try {
        ov::genai::GenerationConfig config;
        config.update_generation_config(*context->generation_config);

        ov::genai::StreamerVariant streamer = std::monostate();
        if (context->streamer_tsfn.has_value()) {
            streamer = [context, &streamer_exceptions](std::string word) {
                std::promise<ov::genai::StreamingStatus> resultPromise;
                napi_status status = context->streamer_tsfn->BlockingCall(
                    [word, &resultPromise, &streamer_exceptions](Napi::Env env, Napi::Function jsCallback) {
                        try {
                            auto callback_result = jsCallback.Call({Napi::String::New(env, word)});
                            if (callback_result.IsNumber()) {
                                resultPromise.set_value(static_cast<ov::genai::StreamingStatus>(
                                    callback_result.As<Napi::Number>().Int32Value()));
                            } else {
                                resultPromise.set_value(ov::genai::StreamingStatus::RUNNING);
                            }
                        } catch (std::exception& err) {
                            streamer_exceptions.push_back(err.what());
                            resultPromise.set_value(ov::genai::StreamingStatus::CANCEL);
                        }
                    });

                if (status != napi_ok) {
                    streamer_exceptions.push_back("The streamer callback BlockingCall failed with the status: " +
                                                  status);
                    return ov::genai::StreamingStatus::CANCEL;
                }

                return resultPromise.get_future().get();
            };
        }

        std::visit(overloaded{[context, config, streamer, &result](ov::genai::StringInputs& inputs) {
                                  result = context->pipe->generate(inputs, config, streamer);
                              },
                              [context, config, streamer, &result](ov::genai::ChatHistory& inputs) {
                                  result = context->pipe->generate(inputs, config, streamer);
                              },
                              [&](auto&) {
                                  OPENVINO_THROW("Unsupported type for generate inputs.");
                              }},
                   context->inputs);

    } catch (std::exception& e) {
        report_error(e.what());
    }
    // should be called right after inference to release the flag asap
    *context->is_generating = false;

    // Call callback with result or error
    try {
        if (!streamer_exceptions.empty()) {
            // If there were exceptions from the streamer, report them all as a single error and finish without result
            std::string combined_error = "Streamer exceptions occurred:\n";
            for (size_t i = 0; i < streamer_exceptions.size(); ++i) {
                combined_error += "[" + std::to_string(i + 1) + "] " + streamer_exceptions[i] + "\n";
            }
            report_error(combined_error);
        } else {
            // If no exceptions from streamer, call the final callback with the result
            napi_status status =
                context->generate_tsfn.BlockingCall([result, &report_error](Napi::Env env, Napi::Function jsCallback) {
                    try {
                        jsCallback.Call({
                            env.Null(),                     // Error should be null in normal case
                            to_decoded_result(env, result)  // Return DecodedResults as the final result
                        });
                    } catch (std::exception& err) {
                        report_error("The final callback failed. Details:\n" + std::string(err.what()));
                    }
                });

            if (status != napi_ok) {
                report_error("The final BlockingCall failed with status " + status);
            }
        }
    } catch (std::exception& e) {
        report_error(e.what());
    }
    finalize();
}

LLMPipelineWrapper::LLMPipelineWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<LLMPipelineWrapper>(info) {};

Napi::Function LLMPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "LLMPipeline",
                       {InstanceMethod("init", &LLMPipelineWrapper::init),
                        InstanceMethod("generate", &LLMPipelineWrapper::generate),
                        InstanceMethod("getTokenizer", &LLMPipelineWrapper::get_tokenizer),
                        InstanceMethod("getGenerationConfig", &LLMPipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &LLMPipelineWrapper::set_generation_config),
                        InstanceMethod("startChat", &LLMPipelineWrapper::start_chat),
                        InstanceMethod("finishChat", &LLMPipelineWrapper::finish_chat)});
}

Napi::Value LLMPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!*this->is_initializing, "Pipeline is already initializing");
        VALIDATE_ARGS_COUNT(info, 4, "init()");
        const auto model_path = js_to_cpp<std::string>(env, info[0]);
        const auto device = js_to_cpp<std::string>(env, info[1]);
        const auto& properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        auto callback = info[3].As<Napi::Function>();

        InitWorker* asyncWorker =
            new InitWorker(callback, this->pipe, this->is_initializing, model_path, device, properties);
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    TsfnContext* context = nullptr;

    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        OPENVINO_ASSERT(!*this->is_generating, "Another generation is already in progress");
        *this->is_generating = true;
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        auto inputs = js_to_cpp<GenerateInputs>(env, info[0]);
        auto generation_config = js_to_cpp<ov::AnyMap>(env, info[1]);
        OPENVINO_ASSERT(info[2].IsFunction() || info[2].IsUndefined(), "streamer callback is not a function");
        auto streamer = info[2];
        OPENVINO_ASSERT(info[3].IsFunction(), "generate callback is not a function");
        auto callback = info[3].As<Napi::Function>();

        context = new TsfnContext(inputs, this->is_generating);
        context->pipe = this->pipe;
        context->generation_config = std::make_shared<ov::AnyMap>(generation_config);
        // Create a ThreadSafeFunction
        context->generate_tsfn =
            Napi::ThreadSafeFunction::New(env,
                                          callback,
                                          "LLM_generate_callback",  // Name
                                          0,                        // Unlimited queue
                                          1,                        // Only one thread will use this initially
                                          [context](Napi::Env) {    // Finalizer used to clean threads up
                                              context->native_thread.join();
                                              delete context;
                                          });
        if (streamer.IsFunction()) {
            context->streamer_tsfn = Napi::ThreadSafeFunction::New(env,
                                                                   streamer.As<Napi::Function>(),
                                                                   "LLM_generate_streamer",  // Name
                                                                   0,                        // Unlimited queue
                                                                   1);  // Only one thread will use this initially
        }
        context->native_thread = std::thread(performInferenceThread, context);
    } catch (const std::exception& ex) {
        *this->is_generating = false;
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::start_chat(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 2, "startChat()");
        auto system_message = js_to_cpp<std::string>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "startChat callback is not a function");
        auto callback = info[1].As<Napi::Function>();

        auto* asyncWorker = new StartChatWorker(callback, this->pipe, system_message);
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::finish_chat(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "finishChat()");
        OPENVINO_ASSERT(info[0].IsFunction(), "finishChat callback is not a function");
        auto callback = info[0].As<Napi::Function>();

        FinishChatWorker* asyncWorker = new FinishChatWorker(callback, this->pipe);
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::get_tokenizer(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        auto tokenizer = this->pipe->get_tokenizer();
        return TokenizerWrapper::wrap(env, tokenizer);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        return cpp_to_js<ov::genai::GenerationConfig, Napi::Value>(env, this->pipe->get_generation_config());
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        this->pipe->set_generation_config(js_to_cpp<ov::genai::GenerationConfig>(env, info[0]));
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}
