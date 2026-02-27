#pragma once

#include <napi.h>
#include "openvino/genai/chat_history.hpp"

class ChatHistoryWrap : public Napi::ObjectWrap<ChatHistoryWrap> {
public:
    static Napi::Function get_class(Napi::Env env);
    
    ChatHistoryWrap(const Napi::CallbackInfo& info);
    
    ov::genai::ChatHistory& get_value();

private:
    Napi::Value push_back(const Napi::CallbackInfo& info);
    void pop_back(const Napi::CallbackInfo& info);
    Napi::Value get_messages(const Napi::CallbackInfo& info);
    Napi::Value set_messages(const Napi::CallbackInfo& info);
    void clear(const Napi::CallbackInfo& info);
    Napi::Value size(const Napi::CallbackInfo& info);
    Napi::Value empty(const Napi::CallbackInfo& info);
    Napi::Value set_tools(const Napi::CallbackInfo& info);
    Napi::Value get_tools(const Napi::CallbackInfo& info);
    Napi::Value set_extra_context(const Napi::CallbackInfo& info);
    Napi::Value get_extra_context(const Napi::CallbackInfo& info);
    
    ov::genai::ChatHistory m_chat_history;
};
