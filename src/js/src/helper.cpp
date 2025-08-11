#include "include/helper.hpp"

ov::AnyMap to_anyMap(const Napi::Env& env, const Napi::Value& val) {
    ov::AnyMap properties;
    if (!val.IsObject()) {
        OPENVINO_THROW("Passed Napi::Value must be an object.");
    }
    const auto& parameters = val.ToObject();
    const auto& keys = parameters.GetPropertyNames();

    for (uint32_t i = 0; i < keys.Length(); ++i) {
        const auto& property_name = static_cast<Napi::Value>(keys[i]).ToString().Utf8Value();

        const auto& any_value = js_to_cpp<ov::Any>(env, parameters.Get(property_name));

        properties.insert(std::make_pair(property_name, any_value));
    }

    return properties;
}

template <>
ov::Any js_to_cpp<ov::Any>(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsString()) {
        return ov::Any(value.ToString().Utf8Value());
    } else if (value.IsBigInt()) {
        Napi::BigInt big_value = value.As<Napi::BigInt>();
        bool is_lossless;
        int64_t big_num = big_value.Int64Value(&is_lossless);

        if (!is_lossless) {
            OPENVINO_THROW("Result of BigInt conversion to int64_t results in a loss of precision");
        }

        return ov::Any(big_num);
    } else if (value.IsNumber()) {
        Napi::Number num = value.ToNumber();

        if (is_napi_value_int(env, value)) {
            return ov::Any(num.Int32Value());
        } else {
            return ov::Any(num.DoubleValue());
        }
    } else if (value.IsBoolean()) {
        return ov::Any(static_cast<bool>(value.ToBoolean()));
    } else if (value.IsArray()) {
        return ov::Any(js_to_cpp<std::vector<std::string>>(env, value));
    } else if (value.IsObject()) {
        if (value.ToString().Utf8Value() == "[object Set]") {
            try {
                // try to cast to set of strings
                auto object_value = value.As<Napi::Object>();
                auto values = object_value.Get("values").As<Napi::Function>();
                auto iterator = values.Call(object_value, {}).As<Napi::Object>();
                auto next = iterator.Get("next").As<Napi::Function>();
                auto size = object_value.Get("size").As<Napi::Number>().Int32Value();

                std::set<std::string> set;
                for (uint32_t i = 0; i < size; ++i) {
                    auto item = next.Call(iterator, {}).As<Napi::Object>();
                    set.insert(item.Get("value").As<Napi::String>().Utf8Value());
                }

                return ov::Any(set);
            } catch (std::exception& e) {
                std::cerr << "Cannot convert to set: " << e.what() << std::endl;
            }
        }
    }
    OPENVINO_THROW("Cannot convert to ov::Any");
}

template <>
ov::AnyMap js_to_cpp<ov::AnyMap>(const Napi::Env& env, const Napi::Value& value) {
    std::map<std::string, ov::Any> result_map;
    const auto& object = value.ToObject();
    const auto& keys = object.GetPropertyNames();

    for (uint32_t i = 0; i < keys.Length(); ++i) {
        const std::string& key_name = keys.Get(i).ToString();
        result_map[key_name] = js_to_cpp<ov::Any>(env, object.Get(key_name));
    }

    return result_map;
}

template <>
std::vector<std::string> js_to_cpp<std::vector<std::string>>(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsArray()) {
        auto array = value.As<Napi::Array>();
        size_t arrayLength = array.Length();

        std::vector<std::string> nativeArray;
        for (uint32_t i = 0; i < arrayLength; ++i) {
            Napi::Value arrayItem = array[i];
            if (!arrayItem.IsString()) {
                OPENVINO_THROW(std::string("Passed array must contain only strings."));
            }
            nativeArray.push_back(arrayItem.As<Napi::String>().Utf8Value());
        }
        return nativeArray;

    } else {
        OPENVINO_THROW("Passed argument must be of type Array or TypedArray.");
    }
}

template <>
ov::genai::StringInputs js_to_cpp<ov::genai::StringInputs>(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsString()) {
        return value.As<Napi::String>().Utf8Value();
    } else if (value.IsArray()) {
        return js_to_cpp<std::vector<std::string>>(env, value);
    } else {
        OPENVINO_THROW("Passed argument must be a string or an array of strings");
    }
}

template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(const Napi::Env& env, const ov::genai::EmbeddingResult embedding_result) {
    return std::visit(overloaded {
        [env](std::vector<float> embed_vector) -> Napi::Value {
            auto vector_size = embed_vector.size();
            auto buffer = Napi::ArrayBuffer::New(env, vector_size * sizeof(float));
            std::memcpy(buffer.Data(), embed_vector.data(), vector_size * sizeof(float));
            Napi::Value typed_array = Napi::Float32Array::New(env, vector_size, buffer, 0);
            return typed_array;
        }, 
        [env](std::vector<int8_t> embed_vector) -> Napi::Value {
            auto buffer_size = embed_vector.size();
            auto buffer = Napi::ArrayBuffer::New(env, buffer_size * sizeof(int8_t));
            std::memcpy(buffer.Data(), embed_vector.data(), buffer_size * sizeof(int8_t));
            Napi::Value typed_array = Napi::Int8Array::New(env, buffer_size, buffer, 0);
            return typed_array;
        },
        [env](std::vector<uint8_t> embed_vector) -> Napi::Value {
            auto buffer_size = embed_vector.size();
            auto buffer = Napi::ArrayBuffer::New(env, buffer_size * sizeof(uint8_t));
            std::memcpy(buffer.Data(), embed_vector.data(), buffer_size * sizeof(uint8_t));
            Napi::Value typed_array = Napi::Uint8Array::New(env, buffer_size, buffer, 0);
            return typed_array;
        },
        [env](auto& args) -> Napi::Value { OPENVINO_THROW("Unsupported type for EmbeddingResult."); }
    }, embedding_result);
}

template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResults, Napi::Value>(const Napi::Env& env, const ov::genai::EmbeddingResults embedding_result) {
    return std::visit([env](auto& embed_vector) {
        auto js_result = Napi::Array::New(env, embed_vector.size());
        for (auto i = 0; i < embed_vector.size(); i++) {
            js_result[i] = cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(env, embed_vector[i]);
        }
        return js_result;
    }, embedding_result);
}

template <>
Napi::Value cpp_to_js<std::vector<std::string>, Napi::Value>(const Napi::Env& env,
                                                             const std::vector<std::string> value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::String::New(env, value[i]);
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<std::vector<float>, Napi::Value>(const Napi::Env& env,
                                                             const std::vector<float> value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::Number::New(env, value[i]);
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<std::vector<double>, Napi::Value>(const Napi::Env& env,
                                                             const std::vector<double> value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::Number::New(env, value[i]);
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<std::vector<size_t>, Napi::Value>(const Napi::Env& env,
                                                             const std::vector<size_t> value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::Number::New(env, value[i]);
    }
    return js_array;
}

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num) {
    return env.Global().Get("Number").ToObject().Get("isInteger").As<Napi::Function>().Call({num}).ToBoolean().Value();
}
