#include "include/addon.hpp"
#include "include/helper.hpp"
#include "include/tokenizer.hpp"

TokenizerWrapper::TokenizerWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TokenizerWrapper>(info) {};

Napi::Function TokenizerWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
        "Tokenizer",
        {
            InstanceMethod("applyChatTemplate", &TokenizerWrapper::apply_chat_template),
            InstanceMethod("getBosToken", &TokenizerWrapper::get_bos_token),
            InstanceMethod("getBosTokenId", &TokenizerWrapper::get_bos_token_id),
            InstanceMethod("getEosToken", &TokenizerWrapper::get_eos_token),
            InstanceMethod("getEosTokenId", &TokenizerWrapper::get_eos_token_id),
            InstanceMethod("getPadToken", &TokenizerWrapper::get_pad_token),
            InstanceMethod("getPadTokenId", &TokenizerWrapper::get_pad_token_id),
        }
    );
}

Napi::Object TokenizerWrapper::wrap(Napi::Env env, ov::genai::Tokenizer tokenizer) {
    const auto& prototype = env.GetInstanceData<AddonData>()->tokenizer;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to CompiledModel prototype.");
    }
    auto obj = prototype.New({});
    const auto tw = Napi::ObjectWrap<TokenizerWrapper>::Unwrap(obj);
    tw->_tokenizer = tokenizer;
    return obj;
}

Napi::Value TokenizerWrapper::apply_chat_template(const Napi::CallbackInfo& info) {
    try {
        auto history = js_to_cpp<ov::genai::ChatHistory>(info.Env(), info[0]);
        OPENVINO_ASSERT(!info[1].IsUndefined() && info[1].IsBoolean(), "The argument 'addGenerationPrompt' must be a boolean");
        bool add_generation_prompt = info[1].ToBoolean();
        std::string chat_template = "";
        if (!info[2].IsUndefined()) {
            chat_template = info[2].ToString().Utf8Value();
        }
        std::optional<ov::genai::JsonContainer> tools;
        if (!info[3].IsUndefined()) {
            tools = ov::genai::JsonContainer::from_json_string(json_stringify(info.Env(), info[3]));
        }
        std::optional<ov::genai::JsonContainer> extra_context;
        if (!info[4].IsUndefined()) {
            extra_context = ov::genai::JsonContainer::from_json_string(json_stringify(info.Env(), info[4]));
        }
        auto result = this->_tokenizer.apply_chat_template(history, add_generation_prompt, chat_template, tools, extra_context);
        return Napi::String::New(info.Env(), result);
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::get_bos_token(const Napi::CallbackInfo& info) {
    try {
        return Napi::String::New(info.Env(), this->_tokenizer.get_bos_token());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::get_bos_token_id(const Napi::CallbackInfo& info) {
    try {
        return Napi::Number::New(info.Env(), this->_tokenizer.get_bos_token_id());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::get_eos_token(const Napi::CallbackInfo& info) {
    try {
        return Napi::String::New(info.Env(), this->_tokenizer.get_eos_token());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::get_eos_token_id(const Napi::CallbackInfo& info) {
    try {
        return Napi::Number::New(info.Env(), this->_tokenizer.get_eos_token_id());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::get_pad_token(const Napi::CallbackInfo& info) {
    try {
        return Napi::String::New(info.Env(), this->_tokenizer.get_pad_token());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::get_pad_token_id(const Napi::CallbackInfo& info) {
    try {
        return Napi::Number::New(info.Env(), this->_tokenizer.get_pad_token_id());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}
