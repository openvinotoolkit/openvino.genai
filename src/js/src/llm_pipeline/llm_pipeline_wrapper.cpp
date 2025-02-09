#include "include/helper.hpp"

#include "include/llm_pipeline/llm_pipeline_wrapper.hpp"
#include "include/llm_pipeline/start_chat_worker.hpp"
#include "include/llm_pipeline/finish_chat_worker.hpp"
#include "include/llm_pipeline/init_worker.hpp"

struct TsfnContext {
    TsfnContext(std::string prompt) : prompt(prompt) {};
    ~TsfnContext() {};

    std::thread native_thread;
    Napi::ThreadSafeFunction tsfn;

    std::string prompt;
    std::shared_ptr<ov::genai::LLMPipeline> pipe = nullptr;
    std::shared_ptr<ov::AnyMap> options = nullptr;
};

void performInferenceThread(TsfnContext* context) {
    try {
        ov::genai::GenerationConfig config;
        config.update_generation_config(*context->options);

        std::function<bool(std::string)> streamer = [context](std::string word) {
            napi_status status = context->tsfn.BlockingCall([word](Napi::Env env, Napi::Function jsCallback) {
                try {
                    jsCallback.Call({
                        Napi::Boolean::New(env, false),
                        Napi::String::New(env, word)
                    });
                } catch(std::exception& err) {
                    Napi::Error::Fatal("performInferenceThread callback error. Details:" , err.what());
                }
            });
            if (status != napi_ok) {
                // Handle error
                Napi::Error::Fatal("performInferenceThread error", "napi_status != napi_ok");
            }

            // Return flag corresponds whether generation should be stopped.
            // false means continue generation.
            return false;
        };

        context->pipe->generate(context->prompt, config, streamer);
        napi_status status = context->tsfn.BlockingCall([](Napi::Env env, Napi::Function jsCallback) {
            jsCallback.Call({
                Napi::Boolean::New(env, true),
            });
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
                        InstanceMethod("startChat", &LLMPipelineWrapper::start_chat),
                        InstanceMethod("finishChat", &LLMPipelineWrapper::finish_chat)});
}

Napi::Value LLMPipelineWrapper::init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const std::string model_path = info[0].ToString();
    const std::string device = info[1].ToString();
    Napi::Function callback = info[2].As<Napi::Function>();

    InitWorker* asyncWorker = new InitWorker(callback, this->pipe, model_path, device);
    asyncWorker->Queue();

    return info.Env().Undefined();
}

Napi::Value LLMPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    TsfnContext* context = nullptr;

    try {
        std::string prompt = info[0].ToString();
        ov::AnyMap options;
        if (info.Length() == 3) {
            options = to_anyMap(info.Env(), info[2]);
        }

        context = new TsfnContext(prompt);
        context->pipe = this->pipe;
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
    Napi::Function callback = info[0].As<Napi::Function>();

    StartChatWorker* asyncWorker = new StartChatWorker(callback, this->pipe);
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
