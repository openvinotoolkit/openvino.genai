#include "include/helper.hpp"

#include "include/addon.hpp"
#include "include/perf_metrics.hpp"

namespace {
constexpr const char* JS_SCHEDULER_CONFIG_KEY = "schedulerConfig";
constexpr const char* CPP_SCHEDULER_CONFIG_KEY = "scheduler_config";
constexpr const char* POOLING_TYPE_KEY = "pooling_type";
constexpr const char* STRUCTURED_OUTPUT_CONFIG_KEY = "structured_output_config";

}  // namespace

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
    if(value.IsUndefined() || value.IsNull()) {
        return result_map;
    }
    if (!value.IsObject()) {
        OPENVINO_THROW("Passed Napi::Value must be an object.");
    }
    const auto& object = value.ToObject();
    const auto& keys = object.GetPropertyNames();

    for (uint32_t i = 0; i < keys.Length(); ++i) {
        const std::string& key_name = keys.Get(i).ToString();
        auto value_by_key = object.Get(key_name);
        if (value_by_key.IsUndefined() || value_by_key.IsNull()) {
            continue;
        }
        if (key_name == JS_SCHEDULER_CONFIG_KEY) {
            result_map[CPP_SCHEDULER_CONFIG_KEY] = js_to_cpp<ov::genai::SchedulerConfig>(env, value_by_key);
        } else if (key_name == POOLING_TYPE_KEY) {
            result_map[key_name] =
                ov::genai::TextEmbeddingPipeline::PoolingType(value_by_key.ToNumber().Int32Value());
        } else if (key_name == STRUCTURED_OUTPUT_CONFIG_KEY) {
            result_map[key_name] = js_to_cpp<ov::genai::StructuredOutputConfig>(env, value_by_key);
        } else {
            result_map[key_name] = js_to_cpp<ov::Any>(env, value_by_key);
        }
    }

    return result_map;
}

template <>
std::string js_to_cpp<std::string>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsString(), "Passed argument must be of type String.");
    return value.As<Napi::String>().Utf8Value();
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
ov::genai::ChatHistory js_to_cpp<ov::genai::ChatHistory>(const Napi::Env& env, const Napi::Value& value) {
    auto incorrect_argument_message = "Chat history must be an array of JS objects";
    if (!value.IsArray()) {
        OPENVINO_THROW(incorrect_argument_message);
    }

    auto array = value.As<Napi::Array>();
    size_t arrayLength = array.Length();

    for (uint32_t i = 0; i < arrayLength; ++i) {
        Napi::Value arrayItem = array[i];
        if (!arrayItem.IsObject()) {
            OPENVINO_THROW(incorrect_argument_message);
        }
    }

    // TODO Consider using direct native JsonContainer conversion instead of string serialization
    auto messages = ov::genai::JsonContainer::from_json_string(json_stringify(env, value));
    return ov::genai::ChatHistory(messages);
}

template <>
ov::genai::SchedulerConfig js_to_cpp<ov::genai::SchedulerConfig>(const Napi::Env& env, const Napi::Value& value) {
    ov::genai::SchedulerConfig config;
    OPENVINO_ASSERT(value.IsObject(), "SchedulerConfig must be a JS object");
    auto obj = value.As<Napi::Object>();

    if (obj.Has("max_num_batched_tokens")) {
        config.max_num_batched_tokens = obj.Get("max_num_batched_tokens").ToNumber().Uint32Value();
    }
    if (obj.Has("num_kv_blocks")) {
        config.num_kv_blocks = obj.Get("num_kv_blocks").ToNumber().Uint32Value();
    }
    if (obj.Has("cache_size")) {
        config.cache_size = obj.Get("cache_size").ToNumber().Uint32Value();
    }
    if (obj.Has("dynamic_split_fuse")) {
        config.dynamic_split_fuse = obj.Get("dynamic_split_fuse").ToBoolean().Value();
    }

    return config;
}

template <>
ov::genai::StructuredOutputConfig::Tag js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsObject(), "Tag must be a JS object");
    auto obj = value.As<Napi::Object>();

    return ov::genai::StructuredOutputConfig::Tag(
        js_to_cpp<std::string>(env, obj.Get("begin")),
        js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, obj.Get("content")),
        js_to_cpp<std::string>(env, obj.Get("end"))
    );
}

template <>
ov::genai::StructuredOutputConfig::StructuralTag js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsString()) {
        return js_to_cpp<std::string>(env, value);
    }
    
    OPENVINO_ASSERT(value.IsObject(), "StructuralTag must be a JS object or string");
    auto obj = value.As<Napi::Object>();

    std::string tag_type = obj.Get("structuralTagType").ToString().Utf8Value();

    if (tag_type == "Regex") {
        return ov::genai::StructuredOutputConfig::Regex(
            js_to_cpp<std::string>(env, obj.Get("value"))
        );
    } else if (tag_type == "JSONSchema") {
        return ov::genai::StructuredOutputConfig::JSONSchema(
            js_to_cpp<std::string>(env, obj.Get("value"))
        );
    } else if (tag_type == "EBNF") {
        return ov::genai::StructuredOutputConfig::EBNF(
            js_to_cpp<std::string>(env, obj.Get("value"))
        );
    } else if (tag_type == "ConstString") {
        return ov::genai::StructuredOutputConfig::ConstString(
            js_to_cpp<std::string>(env, obj.Get("value"))
        );
    } else if (tag_type == "AnyText") {
        return ov::genai::StructuredOutputConfig::AnyText();
    } else if (tag_type == "QwenXMLParametersFormat") {
        return ov::genai::StructuredOutputConfig::QwenXMLParametersFormat(
            js_to_cpp<std::string>(env, obj.Get("jsonSchema"))
        );
    } else if (tag_type == "Concat") {
        std::vector<ov::genai::StructuredOutputConfig::StructuralTag> elements;
        auto js_elements = obj.Get("elements");
        OPENVINO_ASSERT(js_elements.IsArray(), "Concat StructuralTag 'elements' must be an array");
        auto js_array = js_elements.As<Napi::Array>();
        size_t arrayLength = js_array.Length();
        for (uint32_t i = 0; i < arrayLength; ++i) {
            elements.push_back(js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, js_array[i]));
        }
        return std::make_shared<ov::genai::StructuredOutputConfig::Concat>(elements);
    } else if (tag_type == "Union") {
        std::vector<ov::genai::StructuredOutputConfig::StructuralTag> elements;
        auto js_elements = obj.Get("elements");
        OPENVINO_ASSERT(js_elements.IsArray(), "Union StructuralTag 'elements' must be an array");
        auto js_array = js_elements.As<Napi::Array>();
        size_t arrayLength = js_array.Length();
        for (uint32_t i = 0; i < arrayLength; ++i) {
            elements.push_back(js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, js_array[i]));
        }
        return std::make_shared<ov::genai::StructuredOutputConfig::Union>(elements);
    } else if (tag_type == "Tag") {
        return  std::make_shared<ov::genai::StructuredOutputConfig::Tag>(js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(env, obj));
    } else if (tag_type == "TriggeredTags") {
        std::vector<ov::genai::StructuredOutputConfig::Tag> tags;
        auto js_tags = obj.Get("tags");
        auto triggers = js_to_cpp<std::vector<std::string>>(env, obj.Get("triggers"));
        auto at_least_one = obj.Get("atLeastOne");
        auto stop_after_first = obj.Get("stopAfterFirst");
        OPENVINO_ASSERT(
            at_least_one.IsBoolean() && stop_after_first.IsBoolean(),
            "TriggeredTags 'atLeastOne', and 'stopAfterFirst' must be booleans"
        );
        OPENVINO_ASSERT(js_tags.IsArray(), "TriggeredTags 'tags' must be an array");
        auto js_array = js_tags.As<Napi::Array>();
        size_t arrayLength = js_array.Length();
        for (uint32_t i = 0; i < arrayLength; ++i) {
            tags.push_back(js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(env, js_array[i]));
        }
        return std::make_shared<ov::genai::StructuredOutputConfig::TriggeredTags>(
            triggers,
            tags,
            at_least_one.As<Napi::Boolean>().Value(),
            stop_after_first.As<Napi::Boolean>().Value()
        );
    } else if (tag_type == "TagsWithSeparator") {
        std::vector<ov::genai::StructuredOutputConfig::Tag> tags;
        auto separator = js_to_cpp<std::string>(env, obj.Get("separator"));
        auto at_least_one = obj.Get("atLeastOne");
        auto stop_after_first = obj.Get("stopAfterFirst");
        OPENVINO_ASSERT(
            at_least_one.IsBoolean() && stop_after_first.IsBoolean(),
            "TagsWithSeparator 'atLeastOne', and 'stopAfterFirst' must be booleans"
        );

        auto js_tags = obj.Get("tags");
        OPENVINO_ASSERT(js_tags.IsArray(), "TagsWithSeparator 'tags' must be an array");
        auto js_array = js_tags.As<Napi::Array>();
        size_t arrayLength = js_array.Length();
        for (uint32_t i = 0; i < arrayLength; ++i) {
            tags.push_back(js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(env, js_array[i]));
        }

        return std::make_shared<ov::genai::StructuredOutputConfig::TagsWithSeparator>(
            tags,
            separator,
            at_least_one.As<Napi::Boolean>().Value(),
            stop_after_first.As<Napi::Boolean>().Value()
        );
    } else {
        OPENVINO_THROW("Unknown StructuralTag type: " + tag_type);
    }
}

template <>
ov::genai::StructuredOutputConfig js_to_cpp<ov::genai::StructuredOutputConfig>(const Napi::Env& env, const Napi::Value& value) {
    ov::genai::StructuredOutputConfig config;
    OPENVINO_ASSERT(value.IsObject(), "StructuredOutputConfig must be a JS object");
    auto obj = value.As<Napi::Object>();

    if (obj.Has("json_schema") && !obj.Get("json_schema").IsUndefined()) {
        config.json_schema = js_to_cpp<std::string>(env, obj.Get("json_schema"));
    }
    if (obj.Has("regex") && !obj.Get("regex").IsUndefined()) {
        config.regex = js_to_cpp<std::string>(env, obj.Get("regex"));
    }
    if (obj.Has("grammar") && !obj.Get("grammar").IsUndefined()) {
        config.grammar = js_to_cpp<std::string>(env, obj.Get("grammar"));
    }
    if (obj.Has("structural_tags_config") && !obj.Get("structural_tags_config").IsUndefined()) {
        config.structural_tags_config = js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, obj.Get("structural_tags_config"));
    }

    return config;
}

template <>
ov::genai::PerfMetrics& unwrap<ov::genai::PerfMetrics>(const Napi::Env& env, const Napi::Value& value) {
    const auto obj = value.As<Napi::Object>();
    const auto& prototype = env.GetInstanceData<AddonData>()->perf_metrics;

    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    OPENVINO_ASSERT(obj.InstanceOf(prototype.Value().As<Napi::Function>()),
                    "Passed argument is not of type PerfMetrics");

    const auto js_metrics = Napi::ObjectWrap<PerfMetricsWrapper>::Unwrap(obj);
    return js_metrics->get_value();
}

template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(const Napi::Env& env,
                                                               const ov::genai::EmbeddingResult embedding_result) {
    return std::visit(overloaded{[env](std::vector<float> embed_vector) -> Napi::Value {
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
                                 [env](auto& args) -> Napi::Value {
                                     OPENVINO_THROW("Unsupported type for EmbeddingResult.");
                                 }},
                      embedding_result);
}

template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResults, Napi::Value>(const Napi::Env& env,
                                                                const ov::genai::EmbeddingResults embedding_result) {
    return std::visit(
        [env](auto& embed_vector) {
            auto js_result = Napi::Array::New(env, embed_vector.size());
            for (auto i = 0; i < embed_vector.size(); i++) {
                js_result[i] = cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(env, embed_vector[i]);
            }
            return js_result;
        },
        embedding_result);
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
Napi::Value cpp_to_js<std::vector<float>, Napi::Value>(const Napi::Env& env, const std::vector<float> value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::Number::New(env, value[i]);
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<std::vector<double>, Napi::Value>(const Napi::Env& env, const std::vector<double> value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::Number::New(env, value[i]);
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<std::vector<size_t>, Napi::Value>(const Napi::Env& env, const std::vector<size_t> value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::Number::New(env, value[i]);
    }
    return js_array;
}

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num) {
    return env.Global().Get("Number").ToObject().Get("isInteger").As<Napi::Function>().Call({num}).ToBoolean().Value();
}

std::string json_stringify(const Napi::Env& env, const Napi::Value& value) {
    return env.Global()
        .Get("JSON")
        .ToObject()
        .Get("stringify")
        .As<Napi::Function>()
        .Call({ value })
        .ToString()
        .Utf8Value();
}
