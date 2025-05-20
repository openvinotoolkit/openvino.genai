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

                std::set<std::string> set;
                bool done;
                do {
                    Napi::Object result = next.Call(iterator, {}).As<Napi::Object>();
                    done = !result.Get("done").As<Napi::Boolean>();
                    if (!done) {
                        auto v = result.Get("value").As<Napi::String>().Utf8Value();
                        set.insert(v);
                    }
                } while(done);
                return ov::Any(set);
            } catch (std::exception e) {
                std::cerr << "Cannot convert to set: " << e.what() << std::endl;
            }
        }
    }
    OPENVINO_THROW("Cannot convert to ov::Any");
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

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num) {
    return env.Global().Get("Number").ToObject().Get("isInteger").As<Napi::Function>().Call({num}).ToBoolean().Value();
}
