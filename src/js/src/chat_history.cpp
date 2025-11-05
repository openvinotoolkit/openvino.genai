#include "include/chat_history.hpp"
#include "include/helper.hpp"

Napi::Function ChatHistoryWrap::get_class(Napi::Env env) {
    return DefineClass(env, "ChatHistory", {
        InstanceMethod("push", &ChatHistoryWrap::push_back),
        InstanceMethod("pop", &ChatHistoryWrap::pop_back),
        InstanceMethod("getMessages", &ChatHistoryWrap::get_messages),
        InstanceMethod("setMessages", &ChatHistoryWrap::set_messages),
        InstanceMethod("clear", &ChatHistoryWrap::clear),
        InstanceMethod("size", &ChatHistoryWrap::size),
        InstanceMethod("empty", &ChatHistoryWrap::empty),
        InstanceMethod("setTools", &ChatHistoryWrap::set_tools),
        InstanceMethod("getTools", &ChatHistoryWrap::get_tools),
        InstanceMethod("setExtraContext", &ChatHistoryWrap::set_extra_context),
        InstanceMethod("getExtraContext", &ChatHistoryWrap::get_extra_context)
    });
}

ChatHistoryWrap::ChatHistoryWrap(const Napi::CallbackInfo& info) 
    : Napi::ObjectWrap<ChatHistoryWrap>(info) {
    auto env = info.Env();

    try {
        OPENVINO_ASSERT(info.Length() <= 1, "ChatHistory constructor accepts zero or one argument with messages array");

        if (info.Length() == 0 || info[0].IsUndefined()) {
            m_chat_history = ov::genai::ChatHistory();
        } else {
            m_chat_history = ov::genai::ChatHistory(js_to_cpp<ov::genai::JsonContainer>(env, info[0]));
        }
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    }
}

ov::genai::ChatHistory& ChatHistoryWrap::get_value() {
    return m_chat_history;
}

Napi::Value ChatHistoryWrap::push_back(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        OPENVINO_ASSERT(info.Length() == 1, "ChatHistory.push requires one argument with message object");

        m_chat_history.push_back(js_to_cpp<ov::genai::JsonContainer>(env, info[0]));
        return info.This();
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

void ChatHistoryWrap::pop_back(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        m_chat_history.pop_back();
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    }
}

Napi::Value ChatHistoryWrap::get_messages(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        return cpp_to_js<ov::genai::JsonContainer, Napi::Value>(env, m_chat_history.get_messages());
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value ChatHistoryWrap::set_messages(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        OPENVINO_ASSERT(info.Length() == 1, "ChatHistory.setMessages requires one argument with messages array");

        m_chat_history.get_messages() = js_to_cpp<ov::genai::JsonContainer>(env, info[0]);
        return info.This();
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

void ChatHistoryWrap::clear(const Napi::CallbackInfo& info) {
    try {
        m_chat_history.clear();
    } catch (const std::exception& e) {
        Napi::Error::New(info.Env(), e.what()).ThrowAsJavaScriptException();
    }
}

Napi::Value ChatHistoryWrap::size(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        return Napi::Number::New(env, m_chat_history.size());
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value ChatHistoryWrap::empty(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        return Napi::Boolean::New(env, m_chat_history.empty());
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value ChatHistoryWrap::set_tools(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        OPENVINO_ASSERT(info.Length() == 1, "ChatHistory.setTools requires one argument with tools object");

        m_chat_history.set_tools(js_to_cpp<ov::genai::JsonContainer>(env, info[0]));
        return info.This();
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value ChatHistoryWrap::get_tools(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        return cpp_to_js<ov::genai::JsonContainer, Napi::Value>(env, m_chat_history.get_tools());
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value ChatHistoryWrap::set_extra_context(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        OPENVINO_ASSERT(info.Length() == 1, "ChatHistory.setExtraContext requires one argument with extra context object");

        m_chat_history.set_extra_context(js_to_cpp<ov::genai::JsonContainer>(env, info[0]));
        return info.This();
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value ChatHistoryWrap::get_extra_context(const Napi::CallbackInfo& info) {
    auto env = info.Env();

    try {
        return cpp_to_js<ov::genai::JsonContainer, Napi::Value>(env, m_chat_history.get_extra_context());
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}
