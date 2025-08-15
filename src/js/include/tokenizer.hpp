#pragma once

#include <napi.h>
#include "openvino/genai/tokenizer.hpp"

class TokenizerWrapper : public Napi::ObjectWrap<TokenizerWrapper> {
public:
    TokenizerWrapper(const Napi::CallbackInfo& info);
    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, ov::genai::Tokenizer tokenizer);
    Napi::Value apply_chat_template(const Napi::CallbackInfo& info);
    Napi::Value get_bos_token(const Napi::CallbackInfo& info);
    Napi::Value get_bos_token_id(const Napi::CallbackInfo& info);
    Napi::Value get_eos_token(const Napi::CallbackInfo& info);
    Napi::Value get_eos_token_id(const Napi::CallbackInfo& info);
    Napi::Value get_pad_token(const Napi::CallbackInfo& info);
    Napi::Value get_pad_token_id(const Napi::CallbackInfo& info);
private:
    ov::genai::Tokenizer _tokenizer;
};
