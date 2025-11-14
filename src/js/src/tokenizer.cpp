#include "include/addon.hpp"
#include "include/helper.hpp"
#include "include/tokenizer.hpp"

TokenizerWrapper::TokenizerWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TokenizerWrapper>(info) {
    if (info.Length() == 0) {
        return;
    }

    auto env = info.Env();
    try {
        if (info.Length() == 1 || info.Length() == 2) {
            OPENVINO_ASSERT(info[0].IsString(), "Tokenizer constructor expects 'tokenizerPath' to be a string");
            const auto tokenizer_path = js_to_cpp<std::string>(env, info[0]);
            ov::AnyMap properties;
            if (info.Length() == 2) {
                properties = js_to_cpp<ov::AnyMap>(env, info[1]);
            }
            this->_tokenizer = ov::genai::Tokenizer(tokenizer_path, properties);
            return;
        }

        OPENVINO_ASSERT(info.Length() == 4 || info.Length() == 5,
                        "Tokenizer constructor expects 1-2 arguments (path[, properties]) or 4-5 arguments (models, tensors[, properties])");
        OPENVINO_ASSERT(info[0].IsString(), "The argument 'tokenizerModel' must be a string");
        OPENVINO_ASSERT(info[1].IsObject(), "The argument 'tokenizerWeights' must be an OpenVINO Tensor");
        OPENVINO_ASSERT(info[2].IsString(), "The argument 'detokenizerModel' must be a string");
        OPENVINO_ASSERT(info[3].IsObject(), "The argument 'detokenizerWeights' must be an OpenVINO Tensor");

        const auto tokenizer_model = js_to_cpp<std::string>(env, info[0]);
        const auto tokenizer_weights = js_to_cpp<ov::Tensor>(env, info[1]);
        const auto detokenizer_model = js_to_cpp<std::string>(env, info[2]);
        const auto detokenizer_weights = js_to_cpp<ov::Tensor>(env, info[3]);
        ov::AnyMap properties;
        if (info.Length() == 5) {
            properties = js_to_cpp<ov::AnyMap>(env, info[4]);
        }

        this->_tokenizer = ov::genai::Tokenizer(
            tokenizer_model,
            tokenizer_weights,
            detokenizer_model,
            detokenizer_weights,
            properties
        );
    } catch (const std::exception& err) {
        Napi::Error::New(env, err.what()).ThrowAsJavaScriptException();
    }
}

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
            InstanceMethod("getChatTemplate", &TokenizerWrapper::get_chat_template),
            InstanceMethod("getOriginalChatTemplate", &TokenizerWrapper::get_original_chat_template),
            InstanceMethod("setChatTemplate", &TokenizerWrapper::set_chat_template),
            InstanceMethod("supportsPairedInput", &TokenizerWrapper::supports_paired_input),
            InstanceMethod("decode", &TokenizerWrapper::decode),
            InstanceMethod("encode", &TokenizerWrapper::encode),
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
        OPENVINO_ASSERT(info.Length() >= 2,
                       "Tokenizer.applyChatTemplate requires at least two arguments: chatHistory and addGenerationPrompt");
        OPENVINO_ASSERT(info[0].IsObject(), "The argument 'chatHistory' must be an object");
        OPENVINO_ASSERT(info[1].IsBoolean(), "The argument 'addGenerationPrompt' must be a boolean");

        ov::genai::ChatHistory history;
        if (is_chat_history(info.Env(), info[0])) {
            history = unwrap<ov::genai::ChatHistory>(info.Env(), info[0]);
        } else {
            history = ov::genai::ChatHistory(js_to_cpp<ov::genai::JsonContainer>(info.Env(), info[0]));
        }

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
        return Napi::BigInt::New(info.Env(), this->_tokenizer.get_bos_token_id());
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
        return Napi::BigInt::New(info.Env(), this->_tokenizer.get_eos_token_id());
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
        return Napi::BigInt::New(info.Env(), this->_tokenizer.get_pad_token_id());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::encode(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(info.Length() >= 1, "Tokenizer.encode requires at least one argument: text or prompts");

        // Parse encoding options from the last argument if it's an object
        ov::AnyMap tokenization_params;
        auto count_text_args = info.Length();
        
        if (info[count_text_args - 1].IsObject() && !info[count_text_args - 1].IsArray()) {
            tokenization_params = js_to_cpp<ov::AnyMap>(env, info[count_text_args - 1]);
            count_text_args--;
        }

        ov::genai::TokenizedInputs result;

        // Handle different input types
        if (info[0].IsString()) {
            // Single string
            auto text = js_to_cpp<std::string>(env, info[0]);
            result = this->_tokenizer.encode(text, tokenization_params);
        } else if (count_text_args == 1 && info[0].IsArray()) {
            auto arr = info[0].As<Napi::Array>();
            
            // Check if it's array of pairs [[str, str], ...]
            if (arr.Length() > 0 && arr.Get(uint32_t(0)).IsArray()) {
                // Array of pairs
                std::vector<std::pair<std::string, std::string>> paired_prompts;
                for (uint32_t i = 0; i < arr.Length(); ++i) {
                    OPENVINO_ASSERT(arr.Get(i).IsArray(), "Each pair must be an array");
                    auto pair = arr.Get(i).As<Napi::Array>();
                    OPENVINO_ASSERT(pair.Length() == 2, "Each pair must contain exactly 2 strings");
                    paired_prompts.emplace_back(
                        js_to_cpp<std::string>(env, pair.Get(uint32_t(0))),
                        js_to_cpp<std::string>(env, pair.Get(uint32_t(1)))
                    );
                }
                result = this->_tokenizer.encode(paired_prompts, tokenization_params);
            } else {
                // Regular array of strings
                auto prompts = js_to_cpp<std::vector<std::string>>(env, info[0]);
                result = this->_tokenizer.encode(prompts, tokenization_params);
            }
        } else if (count_text_args == 2 && info[0].IsArray() && info[1].IsArray()) {
            // Two arrays (paired input: prompts_1, prompts_2)
            auto prompts1 = js_to_cpp<std::vector<std::string>>(env, info[0]);
            auto prompts2 = js_to_cpp<std::vector<std::string>>(env, info[1]);
            result = this->_tokenizer.encode(prompts1, prompts2, tokenization_params);
        } else {
            OPENVINO_THROW("Unsupported input type for encode. Expected: string, string[], [string, string][], or two string arrays");
        }

        return cpp_to_js<ov::genai::TokenizedInputs, Napi::Value>(env, result);
    } catch (std::exception& err) {
        Napi::Error::New(env, err.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value TokenizerWrapper::decode(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(info.Length() >= 1, "Tokenizer.decode requires at least one argument: tokens");

        ov::AnyMap detokenization_params;
        if (info.Length() >= 2) {
            const auto& options_candidate = info[1];
            detokenization_params = js_to_cpp<ov::AnyMap>(env, options_candidate);
        }

        // Handle different input types
        if (info[0].IsArray()) {
            auto arr = info[0].As<Napi::Array>();
            
            // Check if it's a 2D array (batch of sequences)
            if (arr.Length() > 0 && arr.Get(uint32_t(0)).IsArray()) {
                // Batch decoding: number[][] | bigint[][]
                std::vector<std::vector<int64_t>> batch_tokens;
                for (uint32_t i = 0; i < arr.Length(); ++i) {
                    batch_tokens.push_back(js_to_cpp<std::vector<int64_t>>(env, arr.Get(i)));
                }
                auto result = this->_tokenizer.decode(batch_tokens, detokenization_params);
                return cpp_to_js<std::vector<std::string>, Napi::Value>(env, result);
            } else {
                // Single sequence: number[] | bigint[]
                auto tokens = js_to_cpp<std::vector<int64_t>>(env, info[0]);
                auto result = this->_tokenizer.decode(tokens, detokenization_params);
                return Napi::String::New(env, result);
            }
        } else {
            // Tensor input
            auto tensor = js_to_cpp<ov::Tensor>(env, info[0]);
            auto result = this->_tokenizer.decode(tensor, detokenization_params);
            return cpp_to_js<std::vector<std::string>, Napi::Value>(env, result);
        }
    } catch (std::exception& err) {
        Napi::Error::New(env, err.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value TokenizerWrapper::get_chat_template(const Napi::CallbackInfo& info) {
    try {
        return Napi::String::New(info.Env(), this->_tokenizer.get_chat_template());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::get_original_chat_template(const Napi::CallbackInfo& info) {
    try {
        return Napi::String::New(info.Env(), this->_tokenizer.get_original_chat_template());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::set_chat_template(const Napi::CallbackInfo& info) {
    try {
        OPENVINO_ASSERT(info.Length() >= 1, "Tokenizer.setChatTemplate requires one argument: chatTemplate");
        OPENVINO_ASSERT(info[0].IsString(), "The argument 'chatTemplate' must be a string");

        this->_tokenizer.set_chat_template(js_to_cpp<std::string>(info.Env(), info[0]));
        return info.Env().Undefined();
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Value TokenizerWrapper::supports_paired_input(const Napi::CallbackInfo& info) {
    try {
        return Napi::Boolean::New(info.Env(), this->_tokenizer.supports_paired_input());
    } catch (std::exception& err) {
        Napi::Error::New(info.Env(), err.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}
