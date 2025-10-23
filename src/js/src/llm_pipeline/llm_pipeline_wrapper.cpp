#include "include/llm_pipeline/llm_pipeline_wrapper.hpp"

#include <future>
#include "include/addon.hpp"
#include "include/helper.hpp"
#include "include/perf_metrics.hpp"
#include "include/llm_pipeline/start_chat_worker.hpp"
#include "include/llm_pipeline/finish_chat_worker.hpp"
#include "include/llm_pipeline/init_worker.hpp"
#include "include/tokenizer.hpp"

struct TsfnContext {
    TsfnContext(ov::genai::StringInputs prompt) : prompt(prompt) {};
    ~TsfnContext() {};

    std::thread native_thread;
    Napi::ThreadSafeFunction tsfn;

    ov::genai::StringInputs prompt;
    std::shared_ptr<ov::genai::LLMPipeline> pipe = nullptr;
    std::shared_ptr<ov::AnyMap> generation_config = nullptr;
    std::shared_ptr<ov::AnyMap> options = nullptr;
};

Napi::Object create_decoded_results_object(Napi::Env env, const ov::genai::DecodedResults& result) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("texts", cpp_to_js<std::vector<std::string>, Napi::Value>(env, result.texts));
    obj.Set("scores", cpp_to_js<std::vector<float>, Napi::Value>(env, result.scores));
    obj.Set("perfMetrics", PerfMetricsWrapper::wrap(env, result.perf_metrics));
    obj.Set("subword", Napi::String::New(env, result));
    return obj;
}

void performInferenceThread(TsfnContext* context) {
    try {
        ov::genai::GenerationConfig config;
        config.update_generation_config(*context->generation_config);

        auto disableStreamer = false;
        if (context->options->find("disableStreamer") != context->options->end()) {
            auto value = (*context->options)["disableStreamer"];
            if (value.is<bool>()) {
                disableStreamer = value.as<bool>();
            } else {
                OPENVINO_THROW("disableStreamer option should be boolean");
            }
        }

        ov::genai::StreamerVariant streamer = std::monostate();
        if (!disableStreamer) {
            streamer = [context](std::string word) {
                std::promise<ov::genai::StreamingStatus> resultPromise;
                napi_status status = context->tsfn.BlockingCall([word, &resultPromise](Napi::Env env, Napi::Function jsCallback) {
                    try {
                        auto callback_result = jsCallback.Call({
                            Napi::Boolean::New(env, false),
                            Napi::String::New(env, word)
                        });
                        if (callback_result.IsNumber()) {
                            resultPromise.set_value(static_cast<ov::genai::StreamingStatus>(callback_result.As<Napi::Number>().Int32Value()));
                        } else {
                            resultPromise.set_value(ov::genai::StreamingStatus::RUNNING);
                        }
                    } catch(std::exception& err) {
                        Napi::Error::Fatal("performInferenceThread callback error. Details:" , err.what());
                    }
                });
                if (status != napi_ok) {
                    // Handle error
                    Napi::Error::Fatal("performInferenceThread error", "napi_status != napi_ok");
                }

                // Return flag corresponds whether generation should be stopped.
                return resultPromise.get_future().get();
            };
        }

        auto result = context->pipe->generate(context->prompt, config, streamer);
        napi_status status = context->tsfn.BlockingCall([result](Napi::Env env, Napi::Function jsCallback) {
            jsCallback.Call({Napi::Boolean::New(env, true), create_decoded_results_object(env, result)});
        });

        if (status != napi_ok) {
            // Handle error
            Napi::Error::Fatal("performInferenceThread error", "napi_status != napi_ok");
        }

        context->tsfn.Release();
    }
    catch(std::exception& e) {
        Napi::Error::Fatal("performInferenceThread error" , e.what());

        context->tsfn.Release();
    }
}

LLMPipelineWrapper::LLMPipelineWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<LLMPipelineWrapper>(info) {};

Napi::Function LLMPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "LLMPipeline",
                       {InstanceMethod("init", &LLMPipelineWrapper::init),
                        InstanceMethod("generate", &LLMPipelineWrapper::generate),
                        InstanceMethod("getTokenizer", &LLMPipelineWrapper::get_tokenizer),
                        InstanceMethod("startChat", &LLMPipelineWrapper::start_chat),
                        InstanceMethod("finishChat", &LLMPipelineWrapper::finish_chat)});
}

Napi::Value LLMPipelineWrapper::init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const std::string model_path = info[0].ToString();
    const std::string device = info[1].ToString();
    const auto& properties = js_to_cpp<ov::AnyMap>(info.Env(), info[2]);
    Napi::Function callback = info[3].As<Napi::Function>();

    InitWorker* asyncWorker = new InitWorker(callback, this->pipe, model_path, device, properties);
    asyncWorker->Queue();

    return info.Env().Undefined();
}

Napi::Value LLMPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    TsfnContext* context = nullptr;

    try {
        ov::genai::StringInputs prompt = js_to_cpp<ov::genai::StringInputs>(env, info[0]);
        auto generation_config = js_to_cpp<ov::AnyMap>(info.Env(), info[2]);
        ov::AnyMap options;
        if (info.Length() == 4) {
            options = js_to_cpp<ov::AnyMap>(info.Env(), info[3]);
        }

        context = new TsfnContext(prompt);
        context->pipe = this->pipe;
        context->generation_config = std::make_shared<ov::AnyMap>(generation_config);
        context->options = std::make_shared<ov::AnyMap>(options);
        // Create a ThreadSafeFunction
        context->tsfn = Napi::ThreadSafeFunction::New(
            env,
            info[1].As<Napi::Function>(),   // JavaScript function called asynchronously
            "TSFN",                         // Name
            0,                              // Unlimited queue
            1,                              // Only one thread will use this initially
            [context](Napi::Env) {          // Finalizer used to clean threads up
                // std::cout << "Finalize TFSN" << std::endl;
                context->native_thread.join();
                delete context;
            }
        );
        context->native_thread = std::thread(performInferenceThread, context);

        return Napi::Boolean::New(env, false);
    } catch(Napi::TypeError& type_err) {
        throw type_err;
    } catch(std::exception& err) {
        std::cout << "Catch in the thread: '" << err.what() << "'" << std::endl;
        if (context != nullptr) {
            context->tsfn.Release();
        }

        throw Napi::Error::New(env, err.what());
    }

    return Napi::Boolean::New(env, true);
}

Napi::Value LLMPipelineWrapper::start_chat(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    OPENVINO_ASSERT(
        info.Length() == 2 && info[0].IsString() && info[1].IsFunction(),
        "startChat expects 2 arguments: system_message and callback function"
    );
    auto system_message = js_to_cpp<std::string>(info.Env(), info[0]);
    auto callback = info[1].As<Napi::Function>();

    StartChatWorker* asyncWorker = new StartChatWorker(callback, this->pipe, system_message);
    asyncWorker->Queue();

    return info.Env().Undefined();
}

Napi::Value LLMPipelineWrapper::finish_chat(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Function callback = info[0].As<Napi::Function>();

    FinishChatWorker* asyncWorker = new FinishChatWorker(callback, this->pipe);
    asyncWorker->Queue();

    return info.Env().Undefined();
}

Napi::Value LLMPipelineWrapper::get_tokenizer(const Napi::CallbackInfo& info) {
    auto tokenizer = this->pipe->get_tokenizer();

    return TokenizerWrapper::wrap(info.Env(), tokenizer);
}
