// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>
#include <openvino/runtime/core.hpp>
#include "openvino/genai/generation_config.hpp"
#include "sampling/structured_output/structured_output_controller.hpp"
#include "tokenizer/tokenizer_impl.hpp"
#include "json_utils.hpp"
#include "utils.hpp"


namespace ov {
namespace genai {

ov::Property<size_t> rng_seed{"rng_seed"};

namespace {

using SOC = StructuredOutputConfig;
using StructuralTag = SOC::StructuralTag;
using ToolChoice = std::string;
using Json = nlohmann::ordered_json;

struct FunctionTool {
    std::string name;
    Json parameters = nullptr;
    std::optional<bool> strict;
};

struct BuiltinTool {
    std::string type;
    std::string name;
    Json parameters = nullptr;
};

struct NormalizedTools {
    std::vector<FunctionTool> functions;
    std::vector<BuiltinTool> builtins;
    ToolChoice choice = "auto";
};

using ModelFormatOptions = ModelStructuralTagOptions;

const Json& as_json(const JsonContainer& value) {
    return *static_cast<const Json*>(value._get_json_value_ptr());
}

std::string json_type(const Json& value) {
    if (value.is_null()) return "null";
    if (value.is_boolean()) return "boolean";
    if (value.is_number()) return "number";
    if (value.is_string()) return "string";
    if (value.is_array()) return "array";
    if (value.is_object()) return "object";
    return "unknown";
}

void assert_object(const Json& value, const std::string& name) {
    OPENVINO_ASSERT(value.is_object(), name, " must be an object, but got ", json_type(value), ".");
}

std::string require_string_field(const Json& value, const std::string& key, const std::string& context) {
    OPENVINO_ASSERT(value.contains(key), context, " must contain string field '", key, "'.");
    OPENVINO_ASSERT(value.at(key).is_string(), context, " field '", key, "' must be a string.");
    return value.at(key).get<std::string>();
}

std::optional<bool> optional_bool_field(const Json& value, const std::string& key, const std::string& context) {
    if (!value.contains(key) || value.at(key).is_null()) {
        return std::nullopt;
    }
    OPENVINO_ASSERT(value.at(key).is_boolean(), context, " field '", key, "' must be a boolean when present.");
    return value.at(key).get<bool>();
}

Json optional_parameters_field(const Json& value, const std::string& context) {
    if (!value.contains("parameters") || value.at("parameters").is_null()) {
        return nullptr;
    }
    OPENVINO_ASSERT(value.at("parameters").is_object(), context, " field 'parameters' must be an object or null.");
    return value.at("parameters");
}

std::string parameters_schema(const FunctionTool& tool) {
    if (tool.strict.has_value() && !*tool.strict) {
        return "true";
    }
    if (tool.parameters.is_null()) {
        return "true";
    }
    return tool.parameters.dump();
}

std::string parameters_schema(const BuiltinTool& tool) {
    if (tool.parameters.is_null()) {
        return "true";
    }
    return tool.parameters.dump();
}

std::vector<std::string> text_excludes(bool exclude_special_tokens, std::initializer_list<const char*> tokens) {
    if (!exclude_special_tokens) {
        return {};
    }
    std::vector<std::string> result;
    result.reserve(tokens.size());
    for (const auto* token : tokens) {
        result.emplace_back(token);
    }
    return result;
}

SOC::JSONSchema schema(const std::string& json_schema, const ModelFormatOptions& options, const std::string& style = "json") {
    return SOC::JSONSchema(json_schema, style, options.any_order);
}

std::shared_ptr<SOC::Concat> seq(std::initializer_list<StructuralTag> elements) {
    return std::make_shared<SOC::Concat>(std::vector<StructuralTag>(elements));
}

std::shared_ptr<SOC::Tag> tag_ptr(SOC::Tag tag) {
    return std::make_shared<SOC::Tag>(std::move(tag));
}

std::shared_ptr<SOC::TagsWithSeparator> separated(const std::vector<SOC::Tag>& tags,
                                                  const std::string& separator,
                                                  bool at_least_one = false) {
    return std::make_shared<SOC::TagsWithSeparator>(tags, separator, at_least_one);
}

std::shared_ptr<SOC::TriggeredTags> triggered(const std::vector<std::string>& triggers,
                                              const std::vector<SOC::Tag>& tags,
                                              const std::vector<std::string>& excludes,
                                              bool at_least_one = false) {
    return std::make_shared<SOC::TriggeredTags>(triggers, tags, at_least_one, false, excludes);
}

FunctionTool parse_function_tool(const Json& tool) {
    assert_object(tool, "function tool");
    OPENVINO_ASSERT(tool.contains("function"), "function tool must contain object field 'function'.");
    const Json& function = tool.at("function");
    assert_object(function, "function tool field 'function'");
    FunctionTool parsed;
    parsed.name = require_string_field(function, "name", "function tool field 'function'");
    parsed.parameters = optional_parameters_field(function, "function tool field 'function'");
    parsed.strict = optional_bool_field(function, "strict", "function tool field 'function'");
    return parsed;
}

BuiltinTool parse_builtin_tool(const Json& tool) {
    assert_object(tool, "builtin tool");
    BuiltinTool parsed;
    parsed.type = require_string_field(tool, "type", "builtin tool");
    if (tool.contains("name") && !tool.at("name").is_null()) {
        OPENVINO_ASSERT(tool.at("name").is_string(), "builtin tool field 'name' must be a string when present.");
        parsed.name = tool.at("name").get<std::string>();
    } else {
        parsed.name = parsed.type;
    }
    parsed.parameters = optional_parameters_field(tool, "builtin tool");
    return parsed;
}

std::pair<std::vector<FunctionTool>, std::vector<BuiltinTool>> parse_tools(const JsonContainer& tools_container) {
    const Json& tools = as_json(tools_container);
    OPENVINO_ASSERT(tools.is_array(), "The 'tools' argument must be a JSON array.");

    std::vector<FunctionTool> functions;
    std::vector<BuiltinTool> builtins;
    for (size_t i = 0; i < tools.size(); ++i) {
        const Json& tool = tools.at(i);
        OPENVINO_ASSERT(tool.is_object(), "tools[", i, "] must be an object, but got ", json_type(tool), ".");
        const std::string type = require_string_field(tool, "type", "tool");
        if (type == "function") {
            functions.push_back(parse_function_tool(tool));
        } else {
            builtins.push_back(parse_builtin_tool(tool));
        }
    }
    return {functions, builtins};
}

void filter_allowed_tools(std::vector<FunctionTool>& functions, std::vector<BuiltinTool>& builtins, const Json& allowed_tools) {
    assert_object(allowed_tools, "allowed_tools");
    OPENVINO_ASSERT(allowed_tools.contains("tools"), "allowed_tools must contain field 'tools'.");
    OPENVINO_ASSERT(allowed_tools.at("tools").is_array(), "allowed_tools.tools must be an array.");

    std::unordered_set<std::string> allowed_function_names;
    std::unordered_set<std::string> allowed_builtin_types;
    for (const auto& allowed_tool : allowed_tools.at("tools")) {
        assert_object(allowed_tool, "allowed tool reference");
        const std::string type = require_string_field(allowed_tool, "type", "allowed tool reference");
        if (type == "function") {
            OPENVINO_ASSERT(allowed_tool.contains("function"), "Allowed function tool references must include 'function'.");
            assert_object(allowed_tool.at("function"), "allowed function tool reference field 'function'");
            allowed_function_names.insert(require_string_field(allowed_tool.at("function"), "name", "allowed function tool reference field 'function'"));
        } else {
            allowed_builtin_types.insert(type);
        }
    }

    std::unordered_set<std::string> available_function_names;
    for (const auto& tool : functions) {
        available_function_names.insert(tool.name);
    }
    for (const auto& name : allowed_function_names) {
        OPENVINO_ASSERT(available_function_names.count(name) > 0,
                        "Allowed function tool is not found in the tools list: ", name, ".");
    }

    std::unordered_set<std::string> matched_builtin_types;
    for (const auto& tool : builtins) {
        if (allowed_builtin_types.count(tool.type) > 0) {
            matched_builtin_types.insert(tool.type);
        }
    }
    for (const auto& type : allowed_builtin_types) {
        OPENVINO_ASSERT(matched_builtin_types.count(type) > 0,
                        "Allowed builtin tool is not found in the tools list: ", type, ".");
    }

    functions.erase(std::remove_if(functions.begin(), functions.end(), [&](const FunctionTool& tool) {
        return allowed_function_names.count(tool.name) == 0;
    }), functions.end());
    builtins.erase(std::remove_if(builtins.begin(), builtins.end(), [&](const BuiltinTool& tool) {
        return allowed_builtin_types.count(tool.type) == 0;
    }), builtins.end());
}

NormalizedTools normalize_tool_choice(const JsonContainer& tools_container, const JsonContainer& tool_choice_container) {
    auto [functions, builtins] = parse_tools(tools_container);
    const Json& tool_choice = as_json(tool_choice_container);

    ToolChoice simplified = "auto";
    if (tool_choice.is_null()) {
        simplified = "auto";
    } else if (tool_choice.is_string()) {
        const std::string choice = tool_choice.get<std::string>();
        OPENVINO_ASSERT(choice == "auto" || choice == "none" || choice == "required",
                        "tool_choice string must be one of 'auto', 'none', or 'required'.");
        if (choice == "none") {
            functions.clear();
            builtins.clear();
            simplified = "auto";
        } else {
            simplified = choice;
        }
    } else if (tool_choice.is_object()) {
        const std::string type = require_string_field(tool_choice, "type", "tool_choice");
        if (type == "allowed_tools") {
            OPENVINO_ASSERT(tool_choice.contains("allowed_tools"), "tool_choice.allowed_tools must be present.");
            const Json& allowed_tools = tool_choice.at("allowed_tools");
            filter_allowed_tools(functions, builtins, allowed_tools);
            simplified = require_string_field(allowed_tools, "mode", "tool_choice.allowed_tools");
            OPENVINO_ASSERT(simplified == "auto" || simplified == "required",
                            "tool_choice.allowed_tools.mode must be 'auto' or 'required'.");
        } else if (type == "function") {
            OPENVINO_ASSERT(tool_choice.contains("function"), "tool_choice function object must include 'function'.");
            assert_object(tool_choice.at("function"), "tool_choice field 'function'");
            const std::string tool_name = require_string_field(tool_choice.at("function"), "name", "tool_choice field 'function'");
            functions.erase(std::remove_if(functions.begin(), functions.end(), [&](const FunctionTool& tool) {
                return tool.name != tool_name;
            }), functions.end());
            OPENVINO_ASSERT(!functions.empty(), "The tool with name '", tool_name, "' is not found in the tools list.");
            builtins.clear();
            simplified = "forced";
        } else {
            functions.clear();
            builtins.erase(std::remove_if(builtins.begin(), builtins.end(), [&](const BuiltinTool& tool) {
                return tool.type != type;
            }), builtins.end());
            OPENVINO_ASSERT(builtins.size() == 1,
                            "Builtin tool choice must match exactly one builtin tool, got ", builtins.size(), " matches.");
            simplified = "forced";
        }
    } else {
        OPENVINO_THROW("tool_choice must be a string, object, or null, but got ", json_type(tool_choice), ".");
    }

    OPENVINO_ASSERT(simplified != "required" || !functions.empty() || !builtins.empty(),
                    "The 'tools' list is empty, which is not allowed when 'tool_choice' is 'required'.");
    OPENVINO_ASSERT(simplified != "forced" || functions.size() + builtins.size() == 1,
                    "Forced tool choice must resolve to exactly one tool.");
    return {std::move(functions), std::move(builtins), simplified};
}

StructuralTag llama(const NormalizedTools& nt, const ModelFormatOptions& options) {
    constexpr const char* TOOL_NAME_PREFIX = "{\"name\": \"";
    constexpr const char* PARAMETERS_FIELD_PREFIX = "\", \"parameters\": ";
    constexpr const char* TOOL_OBJECT_BEGIN_PREFIX = "{\"name\": \"";
    constexpr const char* TOOL_OBJECT_PARAMETERS_PREFIX = "\", \"parameters\": ";
    constexpr const char* TOOLS_TRIGGER = "{\"name\": ";
    auto excludes = text_excludes(options.exclude_special_tokens, {"<think>", "</think>"});

    StructuralTag suffix = SOC::AnyText(excludes);
    if (nt.choice == "auto") {
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) {
            tags.emplace_back(std::string(TOOL_OBJECT_BEGIN_PREFIX) + tool.name + TOOL_OBJECT_PARAMETERS_PREFIX,
                              schema(parameters_schema(tool), options),
                              "}");
        }
        suffix = tags.empty() ? StructuralTag(SOC::AnyText(excludes))
                              : StructuralTag(triggered({TOOLS_TRIGGER}, tags, excludes));
    } else if (nt.choice == "forced") {
        OPENVINO_ASSERT(!nt.functions.empty(), "Forced tool choice must resolve to exactly one function tool for this model format.");
        suffix = std::make_shared<SOC::Tag>(std::string(TOOL_NAME_PREFIX) + nt.functions[0].name + PARAMETERS_FIELD_PREFIX,
                                            schema(parameters_schema(nt.functions[0]), options),
                                            "}");
    } else {
        OPENVINO_ASSERT(!nt.functions.empty(), "At least one function tool is required for this model format.");
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) {
            tags.emplace_back(std::string(TOOL_OBJECT_BEGIN_PREFIX) + tool.name + TOOL_OBJECT_PARAMETERS_PREFIX,
                              schema(parameters_schema(tool), options),
                              "}");
        }
        suffix = separated(tags, "", true);
    }
    return suffix;
}

StructuralTag kimi(const NormalizedTools& nt, const ModelFormatOptions& options) {
    constexpr const char* TOOL_CALL_BEGIN = "<|tool_call_begin|>";
    constexpr const char* TOOL_CALL_BEGIN_PREFIX = "<|tool_call_begin|>functions.";
    constexpr const char* TOOL_CALL_SUFFIX = ":";
    constexpr const char* TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>";
    constexpr const char* TOOL_CALL_END = "<|tool_call_end|>";
    constexpr const char* TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>";
    constexpr const char* TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>";
    constexpr const char* THINK_TAG_END = "</think>";
    auto excludes = text_excludes(options.exclude_special_tokens, {"<think>", "</think>"});

    auto make_tag = [&](const FunctionTool& tool) {
        return SOC::Tag(std::string(TOOL_CALL_BEGIN_PREFIX) + tool.name + TOOL_CALL_SUFFIX,
                        seq({SOC::Regex(R"(\d+)"), SOC::ConstString(TOOL_CALL_ARGUMENT_BEGIN), schema(parameters_schema(tool), options)}),
                        TOOL_CALL_END);
    };

    StructuralTag suffix = SOC::AnyText(excludes);
    if (nt.choice == "auto") {
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        if (!tags.empty()) {
            suffix = triggered({TOOL_CALLS_SECTION_BEGIN},
                               {SOC::Tag(TOOL_CALLS_SECTION_BEGIN, separated(tags, "", true), TOOL_CALLS_SECTION_END)},
                               text_excludes(options.exclude_special_tokens, {"<think>", "</think>", TOOL_CALL_BEGIN}));
        }
    } else if (nt.choice == "forced") {
        OPENVINO_ASSERT(!nt.functions.empty(), "Forced tool choice must resolve to exactly one function tool for this model format.");
        suffix = seq({SOC::ConstString(TOOL_CALLS_SECTION_BEGIN), tag_ptr(make_tag(nt.functions[0])), SOC::ConstString(TOOL_CALLS_SECTION_END)});
    } else {
        OPENVINO_ASSERT(!nt.functions.empty(), "At least one function tool is required for this model format.");
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        suffix = seq({SOC::ConstString(TOOL_CALLS_SECTION_BEGIN), separated(tags, "", true), SOC::ConstString(TOOL_CALLS_SECTION_END)});
    }
    if (!options.reasoning) return suffix;
    return seq({std::make_shared<SOC::Tag>("", SOC::AnyText(), THINK_TAG_END), suffix});
}

StructuralTag deepseek_r1(const NormalizedTools& nt, const ModelFormatOptions& options) {
    constexpr const char* TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>";
    constexpr const char* TOOL_CALLS_END = "<｜tool▁calls▁end｜>";
    constexpr const char* TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>";
    constexpr const char* TOOL_CALL_END = "<｜tool▁call▁end｜>";
    constexpr const char* TOOL_SEP = "<｜tool▁sep｜>";
    constexpr const char* JSON_RENDER_BEGIN = "\n```json\n";
    constexpr const char* JSON_RENDER_END = "\n```";
    constexpr const char* THINK_TAG_END = "</think>";
    auto excludes = text_excludes(options.exclude_special_tokens, {"<think>", "</think>"});
    auto make_tag = [&](const FunctionTool& tool) {
        return SOC::Tag(std::string(TOOL_CALL_BEGIN) + "function" + TOOL_SEP + tool.name + JSON_RENDER_BEGIN,
                        schema(parameters_schema(tool), options),
                        std::string(JSON_RENDER_END) + TOOL_CALL_END);
    };

    StructuralTag suffix = SOC::AnyText(excludes);
    if (nt.choice == "auto") {
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        if (!tags.empty()) {
            suffix = triggered({TOOL_CALLS_BEGIN},
                               {SOC::Tag(TOOL_CALLS_BEGIN, separated(tags, "\n", true), TOOL_CALLS_END)},
                               excludes);
        }
    } else if (nt.choice == "forced") {
        OPENVINO_ASSERT(!nt.functions.empty(), "Forced tool choice must resolve to exactly one function tool for this model format.");
        suffix = std::make_shared<SOC::Tag>(std::string(TOOL_CALLS_BEGIN) + TOOL_CALL_BEGIN + "function" + TOOL_SEP + nt.functions[0].name + JSON_RENDER_BEGIN,
                                            schema(parameters_schema(nt.functions[0]), options),
                                            std::string(JSON_RENDER_END) + TOOL_CALL_END + TOOL_CALLS_END);
    } else {
        OPENVINO_ASSERT(!nt.functions.empty(), "At least one function tool is required for this model format.");
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        suffix = std::make_shared<SOC::Tag>(TOOL_CALLS_BEGIN, separated(tags, "\n", true), TOOL_CALLS_END);
    }
    if (!options.reasoning) return suffix;
    return seq({std::make_shared<SOC::Tag>("", SOC::AnyText(), THINK_TAG_END), suffix});
}

StructuralTag deepseek_v3_1(const NormalizedTools& nt, const ModelFormatOptions& options) {
    constexpr const char* TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>";
    constexpr const char* TOOL_CALLS_END = "<｜tool▁calls▁end｜>";
    constexpr const char* TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>";
    constexpr const char* TOOL_CALL_END = "<｜tool▁call▁end｜>";
    constexpr const char* TOOL_SEP = "<｜tool▁sep｜>";
    constexpr const char* THINK_TAG_END = "</think>";
    auto excludes = text_excludes(options.exclude_special_tokens, {"<think>", "</think>"});
    auto make_tag = [&](const FunctionTool& tool) {
        return SOC::Tag(std::string(TOOL_CALL_BEGIN) + tool.name + TOOL_SEP,
                        schema(parameters_schema(tool), options),
                        TOOL_CALL_END);
    };

    StructuralTag suffix = SOC::AnyText(excludes);
    if (nt.choice == "auto") {
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        if (!tags.empty()) suffix = triggered({TOOL_CALLS_BEGIN}, {SOC::Tag(TOOL_CALLS_BEGIN, separated(tags, "", true), TOOL_CALLS_END)}, excludes);
    } else if (nt.choice == "forced") {
        OPENVINO_ASSERT(!nt.functions.empty(), "Forced tool choice must resolve to exactly one function tool for this model format.");
        suffix = std::make_shared<SOC::Tag>(std::string(TOOL_CALLS_BEGIN) + TOOL_CALL_BEGIN + nt.functions[0].name + TOOL_SEP,
                                            schema(parameters_schema(nt.functions[0]), options),
                                            std::string(TOOL_CALL_END) + TOOL_CALLS_END);
    } else {
        OPENVINO_ASSERT(!nt.functions.empty(), "At least one function tool is required for this model format.");
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        suffix = std::make_shared<SOC::Tag>(TOOL_CALLS_BEGIN, separated(tags, "", true), TOOL_CALLS_END);
    }
    if (!options.reasoning) return suffix;
    return seq({std::make_shared<SOC::Tag>("", SOC::AnyText(), THINK_TAG_END), suffix});
}

StructuralTag qwen_3_5(const NormalizedTools& nt, const ModelFormatOptions& options) {
    constexpr const char* TOOL_CALL_BEGIN_PREFIX = "<tool_call>\n<function=";
    constexpr const char* TOOL_CALL_BEGIN_SUFFIX = ">\n";
    constexpr const char* TOOL_CALL_END = "\n</function>\n</tool_call>";
    constexpr const char* TOOL_CALL_TRIGGER = "<tool_call>\n<function=";
    constexpr const char* THINK_TAG_END = "</think>";
    constexpr const char* THINK_SUFFIX = "\n\n";
    auto excludes = text_excludes(options.exclude_special_tokens, {"<think>", "</think>"});
    auto make_tag = [&](const FunctionTool& tool) {
        return SOC::Tag(std::string(TOOL_CALL_BEGIN_PREFIX) + tool.name + TOOL_CALL_BEGIN_SUFFIX,
                        schema(parameters_schema(tool), options, "qwen_xml"),
                        TOOL_CALL_END);
    };
    StructuralTag suffix = SOC::AnyText(excludes);
    if (nt.choice == "auto") {
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        if (!tags.empty()) suffix = triggered({TOOL_CALL_TRIGGER}, tags, excludes);
    } else if (nt.choice == "forced") {
        OPENVINO_ASSERT(!nt.functions.empty(), "Forced tool choice must resolve to exactly one function tool for this model format.");
        suffix = std::make_shared<SOC::Tag>(make_tag(nt.functions[0]));
    } else {
        OPENVINO_ASSERT(!nt.functions.empty(), "At least one function tool is required for this model format.");
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        suffix = separated(tags, "\n", true);
    }
    if (!options.reasoning) return suffix;
    return seq({seq({std::make_shared<SOC::Tag>("", SOC::AnyText(), THINK_TAG_END), SOC::ConstString(THINK_SUFFIX)}), suffix});
}

StructuralTag qwen_3(const NormalizedTools& nt, const ModelFormatOptions& options) {
    constexpr const char* TOOL_CALL_BEGIN_PREFIX = "<tool_call>\n{\"name\": \"";
    constexpr const char* ARGUMENTS_FIELD_PREFIX = "\", \"arguments\": ";
    constexpr const char* TOOL_CALL_END = "}\n</tool_call>";
    constexpr const char* TOOL_CALL_TRIGGER = "<tool_call>";
    constexpr const char* THINK_TAG_END = "</think>";
    constexpr const char* THINK_SUFFIX = "\n\n";
    auto excludes = text_excludes(options.exclude_special_tokens, {"<think>", "</think>"});
    auto make_tag = [&](const FunctionTool& tool) {
        return SOC::Tag(std::string(TOOL_CALL_BEGIN_PREFIX) + tool.name + ARGUMENTS_FIELD_PREFIX,
                        schema(parameters_schema(tool), options),
                        TOOL_CALL_END);
    };
    StructuralTag suffix = SOC::AnyText(excludes);
    if (nt.choice == "auto") {
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        if (!tags.empty()) suffix = triggered({TOOL_CALL_TRIGGER}, tags, excludes);
    } else if (nt.choice == "forced") {
        OPENVINO_ASSERT(!nt.functions.empty(), "Forced tool choice must resolve to exactly one function tool for this model format.");
        suffix = std::make_shared<SOC::Tag>(make_tag(nt.functions[0]));
    } else {
        OPENVINO_ASSERT(!nt.functions.empty(), "At least one function tool is required for this model format.");
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        suffix = separated(tags, "\n", true);
    }
    if (!options.reasoning) return suffix;
    return seq({seq({std::make_shared<SOC::Tag>("", SOC::AnyText(), THINK_TAG_END), SOC::ConstString(THINK_SUFFIX)}), suffix});
}

StructuralTag harmony(const NormalizedTools& nt, const ModelFormatOptions& options) {
    constexpr const char* CALL_END = "<|call|>";
    constexpr const char* FINAL_BEGIN = "<|channel|>final<|message|>";
    const std::vector<std::string> FINAL_END = {"<|end|>", "<|return|>"};
    constexpr const char* ANALYSIS_BEGIN = "<|channel|>analysis<|message|>";
    constexpr const char* TAG_SEPARATOR = "<|start|>assistant";

    auto function_tags = [&](const FunctionTool& tool) {
        StructuralTag content = schema(parameters_schema(tool), options);
        return std::vector<SOC::Tag>{
            SOC::Tag(std::string("<|channel|>commentary to=functions.") + tool.name + "<|constrain|>json<|message|>", content, CALL_END),
            SOC::Tag(std::string(" to=functions.") + tool.name + "<|channel|>commentary <|constrain|>json<|message|>", content, CALL_END),
            SOC::Tag(std::string(" to=functions.") + tool.name + "<|channel|>commentary json<|message|>", content, CALL_END),
        };
    };
    auto builtin_tags = [&](const BuiltinTool& tool) {
        StructuralTag content = schema(parameters_schema(tool), options);
        return std::vector<SOC::Tag>{
            SOC::Tag(std::string("<|channel|>commentary to=") + tool.name + " code<|message|>", content, CALL_END),
            SOC::Tag(std::string(" to=") + tool.name + "<|channel|>commentary code<|message|>", content, CALL_END),
        };
    };

    std::vector<SOC::Tag> tags;
    auto append = [&](const std::vector<SOC::Tag>& more) { tags.insert(tags.end(), more.begin(), more.end()); };
    if (nt.choice == "auto") {
        for (const auto& tool : nt.functions) append(function_tags(tool));
        for (const auto& tool : nt.builtins) append(builtin_tags(tool));
        tags.emplace_back(FINAL_BEGIN, SOC::AnyText(), FINAL_END);
    } else if (nt.choice == "forced") {
        if (!nt.builtins.empty()) {
            append(builtin_tags(nt.builtins[0]));
        } else if (!nt.functions.empty()) {
            append(function_tags(nt.functions[0]));
        } else {
            OPENVINO_THROW("Forced tool choice must resolve to exactly one tool.");
        }
    } else {
        for (const auto& tool : nt.builtins) append(builtin_tags(tool));
        for (const auto& tool : nt.functions) append(function_tags(tool));
        OPENVINO_ASSERT(!tags.empty(), "At least one tool is required for this model format.");
    }
    if (options.reasoning) {
        tags.emplace_back(ANALYSIS_BEGIN, SOC::AnyText(), FINAL_END);
    }
    return separated(tags, TAG_SEPARATOR);
}

StructuralTag dsml_xml(const NormalizedTools& nt,
                       const ModelFormatOptions& options,
                       const std::string& calls_begin,
                       const std::string& calls_end,
                       const std::string& calls_trigger) {
    constexpr const char* INVOKE_BEGIN_PREFIX = "<｜DSML｜invoke name=\"";
    constexpr const char* INVOKE_BEGIN_SUFFIX = "\">\n";
    constexpr const char* INVOKE_END = "</｜DSML｜invoke>\n";
    constexpr const char* TOOL_CALLS_PREFIX = "\n\n";
    constexpr const char* THINK_TAG_END = "</think>";
    auto excludes = text_excludes(options.exclude_special_tokens, {"<think>", "</think>"});
    auto make_tag = [&](const FunctionTool& tool) {
        return SOC::Tag(std::string(INVOKE_BEGIN_PREFIX) + tool.name + INVOKE_BEGIN_SUFFIX,
                        schema(parameters_schema(tool), options, "deepseek_xml"),
                        INVOKE_END);
    };
    StructuralTag suffix = SOC::AnyText(excludes);
    if (nt.choice == "auto") {
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        if (!tags.empty()) suffix = triggered({calls_trigger}, {SOC::Tag(calls_begin, separated(tags, "", true), calls_end)}, excludes);
    } else if (nt.choice == "forced") {
        OPENVINO_ASSERT(!nt.functions.empty(), "Forced tool choice must resolve to exactly one function tool for this model format.");
        suffix = seq({SOC::ConstString(TOOL_CALLS_PREFIX + calls_begin), tag_ptr(make_tag(nt.functions[0])), SOC::ConstString(calls_end)});
    } else {
        OPENVINO_ASSERT(!nt.functions.empty(), "At least one function tool is required for this model format.");
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        suffix = seq({SOC::ConstString(TOOL_CALLS_PREFIX + calls_begin), separated(tags, "", true), SOC::ConstString(calls_end)});
    }
    if (!options.reasoning) return suffix;
    return seq({std::make_shared<SOC::Tag>("", SOC::AnyText(), THINK_TAG_END), suffix});
}

StructuralTag minimax(const NormalizedTools& nt, const ModelFormatOptions& options) {
    constexpr const char* INVOKE_BEGIN_PREFIX = "<invoke name=\"";
    constexpr const char* INVOKE_BEGIN_SUFFIX = "\">\n";
    constexpr const char* INVOKE_END = "</invoke>\n";
    constexpr const char* TOOL_CALL_BEGIN = "<minimax:tool_call>\n";
    constexpr const char* TOOL_CALL_END = "</minimax:tool_call>";
    constexpr const char* TOOL_CALL_TRIGGER = "<minimax:tool_call>";
    constexpr const char* THINK_TAG_END = "</think>";
    constexpr const char* THINK_SUFFIX = "\n\n";
    constexpr const char* EMPTY_THINK_CONTENT = "\n</think>\n\n";
    auto excludes = text_excludes(options.exclude_special_tokens, {"<think>", "</think>"});
    auto make_tag = [&](const FunctionTool& tool) {
        return SOC::Tag(std::string(INVOKE_BEGIN_PREFIX) + tool.name + INVOKE_BEGIN_SUFFIX,
                        schema(parameters_schema(tool), options, "minimax_xml"),
                        INVOKE_END);
    };
    StructuralTag suffix = SOC::AnyText(excludes);
    if (nt.choice == "auto") {
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        if (!tags.empty()) suffix = triggered({TOOL_CALL_TRIGGER}, {SOC::Tag(TOOL_CALL_BEGIN, separated(tags, "", true), TOOL_CALL_END)}, excludes);
    } else if (nt.choice == "forced") {
        OPENVINO_ASSERT(!nt.functions.empty(), "Forced tool choice must resolve to exactly one function tool for this model format.");
        suffix = seq({SOC::ConstString(std::string("\n") + TOOL_CALL_BEGIN), tag_ptr(make_tag(nt.functions[0])), SOC::ConstString(TOOL_CALL_END)});
    } else {
        OPENVINO_ASSERT(!nt.functions.empty(), "At least one function tool is required for this model format.");
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        suffix = seq({SOC::ConstString(std::string("\n") + TOOL_CALL_BEGIN), separated(tags, "", true), SOC::ConstString(TOOL_CALL_END)});
    }
    StructuralTag think_tag = options.reasoning ? StructuralTag(std::make_shared<SOC::Tag>("", SOC::AnyText(), THINK_TAG_END))
                                                : StructuralTag(SOC::ConstString(EMPTY_THINK_CONTENT));
    return seq({think_tag, SOC::ConstString(THINK_SUFFIX), suffix});
}

StructuralTag glm_4_7(const NormalizedTools& nt, const ModelFormatOptions& options) {
    constexpr const char* TOOL_CALL_BEGIN_PREFIX = "<tool_call>";
    constexpr const char* TOOL_CALL_END = "</tool_call>";
    constexpr const char* TOOL_CALL_TRIGGER = "<tool_call>";
    constexpr const char* THINK_TAG_END = "</think>";
    auto reasoning_excludes = text_excludes(options.exclude_special_tokens,
                                            {"<think>", "</think>", TOOL_CALL_BEGIN_PREFIX, TOOL_CALL_END,
                                             "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"});
    auto text_ex = text_excludes(options.exclude_special_tokens,
                                 {"<think>", "</think>", TOOL_CALL_END, "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"});
    auto make_tag = [&](const FunctionTool& tool) {
        return SOC::Tag(std::string(TOOL_CALL_BEGIN_PREFIX) + tool.name,
                        schema(parameters_schema(tool), options, "glm_xml"),
                        TOOL_CALL_END);
    };
    StructuralTag suffix = SOC::AnyText(reasoning_excludes);
    if (nt.choice == "auto") {
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        if (!tags.empty()) suffix = triggered({TOOL_CALL_TRIGGER}, tags, text_ex);
    } else if (nt.choice == "forced") {
        OPENVINO_ASSERT(!nt.functions.empty(), "Forced tool choice must resolve to exactly one function tool for this model format.");
        suffix = std::make_shared<SOC::Tag>(make_tag(nt.functions[0]));
    } else {
        OPENVINO_ASSERT(!nt.functions.empty(), "At least one function tool is required for this model format.");
        std::vector<SOC::Tag> tags;
        for (const auto& tool : nt.functions) tags.push_back(make_tag(tool));
        suffix = separated(tags, "", true);
    }
    if (!options.reasoning) return suffix;
    return seq({std::make_shared<SOC::Tag>("", SOC::AnyText(reasoning_excludes), THINK_TAG_END), suffix});
}

using Builder = std::function<StructuralTag(const NormalizedTools&, const ModelFormatOptions&)>;

const std::vector<std::string>& supported_model_formats() {
    static const std::vector<std::string> formats = {
        "llama", "kimi", "deepseek_r1", "deepseek_v3_1", "qwen_3_5", "qwen_3_coder",
        "qwen_3", "harmony", "deepseek_v3_2", "minimax", "glm_4_7", "deepseek_v4"};
    return formats;
}

std::string supported_model_formats_string() {
    std::ostringstream oss;
    const auto& formats = supported_model_formats();
    for (size_t i = 0; i < formats.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << formats[i];
    }
    return oss.str();
}

const std::unordered_map<std::string, Builder>& registry() {
    static const std::unordered_map<std::string, Builder> builders = {
        {"llama", llama},
        {"kimi", kimi},
        {"deepseek_r1", deepseek_r1},
        {"deepseek_v3_1", deepseek_v3_1},
        {"qwen_3_5", qwen_3_5},
        {"qwen_3_coder", qwen_3_5},
        {"qwen_3", qwen_3},
        {"harmony", harmony},
        {"deepseek_v3_2", [](const NormalizedTools& nt, const ModelFormatOptions& options) {
             return dsml_xml(nt, options, "<｜DSML｜function_calls>\n", "</｜DSML｜function_calls>", "<｜DSML｜function_calls>");
         }},
        {"minimax", minimax},
        {"glm_4_7", glm_4_7},
        {"deepseek_v4", [](const NormalizedTools& nt, const ModelFormatOptions& options) {
             return dsml_xml(nt, options, "<｜DSML｜tool_calls>\n", "</｜DSML｜tool_calls>", "<｜DSML｜tool_calls>");
         }},
    };
    return builders;
}

}  // namespace

GenerationConfig::GenerationConfig(const std::filesystem::path& json_path) {
    using utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path, "' with generation config");

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "eos_token_id", eos_token_id);
    read_json_param(data, "max_new_tokens", max_new_tokens);
    read_json_param(data, "max_length", max_length);
    // note that ignore_eos is not present in HF GenerationConfig
    read_json_param(data, "ignore_eos", ignore_eos);
    read_json_param(data, "min_new_tokens", min_new_tokens);
    read_json_param(data, "stop_strings", stop_strings);
    // note that include_stop_str_in_output is not present in HF GenerationConfig
    read_json_param(data, "include_stop_str_in_output", include_stop_str_in_output);
    // note that stop_token_ids is not present in HF GenerationConfig, but some generation_config.json define
    // multiple eos_token_id (e.g. https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/generation_config.json)
    // so, we need to read them as 'stop_token_ids'
    std::vector<int64_t> ordered_stop_token_ids;
    read_json_param(data, "eos_token_id", ordered_stop_token_ids);

    if (!ordered_stop_token_ids.empty()) {
        for (int64_t stop_token_id : ordered_stop_token_ids)
            stop_token_ids.insert(stop_token_id);

        if (eos_token_id == -1) {
            eos_token_id = ordered_stop_token_ids[0];
        }
    }

    // note that echo is not present in HF GenerationConfig
    read_json_param(data, "echo", echo);
    // note that logprobs is not present in HF GenerationConfig
    read_json_param(data, "logprobs", logprobs);

    // penalties
    read_json_param(data, "repetition_penalty", repetition_penalty);
    // note that frequency_penalty is not present in HF GenerationConfig
    read_json_param(data, "frequency_penalty", frequency_penalty);
    // note that presence_penalty is not present in HF GenerationConfig
    read_json_param(data, "presence_penalty", presence_penalty);

    // beam search
    read_json_param(data, "num_beam_groups", num_beam_groups);
    read_json_param(data, "num_beams", num_beams);
    read_json_param(data, "diversity_penalty", diversity_penalty);
    read_json_param(data, "length_penalty", length_penalty);
    read_json_param(data, "num_return_sequences", num_return_sequences);
    read_json_param(data, "no_repeat_ngram_size", no_repeat_ngram_size);

    if (data.contains("early_stopping")) {
        auto field_type = data["early_stopping"].type();
        if (field_type == nlohmann::json::value_t::string && data["early_stopping"] == "never") {
            stop_criteria = StopCriteria::NEVER;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == true) {
            stop_criteria = StopCriteria::EARLY;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == false) {
            stop_criteria = StopCriteria::HEURISTIC;
        }
    }

    // multinomial
    read_json_param(data, "do_sample", do_sample);
    read_json_param(data, "temperature", temperature);
    read_json_param(data, "top_p", top_p);
    read_json_param(data, "top_k", top_k);
    read_json_param(data, "min_p", min_p);

    // assistant generation
    read_json_param(data, "assistant_confidence_threshold", assistant_confidence_threshold);
    read_json_param(data, "num_assistant_tokens", num_assistant_tokens);
    read_json_param(data, "max_ngram_size", max_ngram_size);

    // tree search
    read_json_param(data, "branching_factor", branching_factor);
    read_json_param(data, "tree_depth", tree_depth);

    // append EOS to stop_token_ids
    if (eos_token_id != -1)
        set_eos_token_id(eos_token_id);
}

void GenerationConfig::set_eos_token_id(size_t tokenizer_eos_token_id) {
    eos_token_id = tokenizer_eos_token_id;
    stop_token_ids.insert(eos_token_id);
}

void GenerationConfig::update_generation_config(const ov::AnyMap& properties) {
    using utils::read_anymap_param;

    // stop conditions
    read_anymap_param(properties, "eos_token_id", eos_token_id);
    read_anymap_param(properties, "max_new_tokens", max_new_tokens);
    read_anymap_param(properties, "max_length", max_length);
    read_anymap_param(properties, "ignore_eos", ignore_eos);
    read_anymap_param(properties, "min_new_tokens", min_new_tokens);
    read_anymap_param(properties, "stop_strings", stop_strings);
    read_anymap_param(properties, "include_stop_str_in_output", include_stop_str_in_output);
    read_anymap_param(properties, "stop_token_ids", stop_token_ids);
    if (eos_token_id != -1) {
        set_eos_token_id(eos_token_id);
    }

    // generic
    read_anymap_param(properties, "echo", echo);
    read_anymap_param(properties, "logprobs", logprobs);
    read_anymap_param(properties, "num_return_sequences", num_return_sequences);
    read_anymap_param(properties, "adapters", adapters);
    read_anymap_param(properties, "apply_chat_template", apply_chat_template);

    // penalties
    read_anymap_param(properties, "frequency_penalty", frequency_penalty);
    read_anymap_param(properties, "presence_penalty", presence_penalty);
    read_anymap_param(properties, "repetition_penalty", repetition_penalty);

    // beam search
    read_anymap_param(properties, "num_beam_groups", num_beam_groups);
    read_anymap_param(properties, "num_beams", num_beams);
    read_anymap_param(properties, "diversity_penalty", diversity_penalty);
    read_anymap_param(properties, "length_penalty", length_penalty);
    read_anymap_param(properties, "stop_criteria", stop_criteria);
    read_anymap_param(properties, "no_repeat_ngram_size", no_repeat_ngram_size);

    // multinomial
    read_anymap_param(properties, "do_sample", do_sample);
    read_anymap_param(properties, "temperature", temperature);
    read_anymap_param(properties, "top_p", top_p);
    read_anymap_param(properties, "top_k", top_k);
    read_anymap_param(properties, "min_p", min_p);
    // TODO: add support of 'generator' property similar to Image generation
    read_anymap_param(properties, "rng_seed", rng_seed);

    // assistant generation
    read_anymap_param(properties, "assistant_confidence_threshold", assistant_confidence_threshold);
    read_anymap_param(properties, "num_assistant_tokens", num_assistant_tokens);
    read_anymap_param(properties, "max_ngram_size", max_ngram_size);

    // Structured output
    read_anymap_param(properties, "structured_output_config", structured_output_config);
    read_anymap_param(properties, "parsers", parsers);

    // CDPruner
    read_anymap_param(properties, "pruning_ratio", pruning_ratio);
    read_anymap_param(properties, "relevance_weight", relevance_weight);

    // tree search
    read_anymap_param(properties, "branching_factor", branching_factor);
    read_anymap_param(properties, "tree_depth", tree_depth);
}


StructuralTagItem::StructuralTagItem(const ov::AnyMap& properties) {
    update_config(properties);
}

void StructuralTagItem::update_config(const ov::AnyMap& properties) {
    using utils::read_anymap_param;

    read_anymap_param(properties, "begin", begin);
    read_anymap_param(properties, "schema", schema);
    read_anymap_param(properties, "end", end);
}


std::string StructuralTagItem::to_string() const {
    return "StructuralTagItem(begin=" + begin +
           ", schema=" + schema +
           ", end=" + end + ")";
}


StructuralTagsConfig::StructuralTagsConfig(const ov::AnyMap& properties) {
    update_config(properties);
}


void StructuralTagsConfig::update_config(const ov::AnyMap& properties) {
    using utils::read_anymap_param;

    read_anymap_param(properties, "structural_tags", structural_tags);
    read_anymap_param(properties, "triggers", triggers);
}


std::string StructuralTagsConfig::to_string() const {
    std::ostringstream tags_repr;
    tags_repr << "[";
    for (auto it = structural_tags.begin(); it != structural_tags.end(); ++it) {
        if (it != structural_tags.begin()) tags_repr << ", ";
        tags_repr << it->to_string();
    }
    tags_repr << "]";

    std::ostringstream triggers_repr;
    triggers_repr << "[";
    for (auto it = triggers.begin(); it != triggers.end(); ++it) {
        if (it != triggers.begin()) triggers_repr << ", ";
        triggers_repr << *it;
    }
    triggers_repr << "]";

    return "StructuralTagsConfig(structural_tags=" + tags_repr.str() +
           ", triggers=" + triggers_repr.str() + ")";
}

std::string StructuralTagsConfig::to_json() const {
    std::vector<StructuredOutputConfig::Tag> tags;
    tags.reserve(structural_tags.size());
    for (const auto& tag : structural_tags) {
        tags.emplace_back(tag.begin, StructuredOutputConfig::JSONSchema{tag.schema}, tag.end);
    }
    return StructuredOutputConfig::TriggeredTags(triggers, tags, false, false).to_json();
}

StructuredOutputConfig::StructuredOutputConfig(const ov::AnyMap& properties) {
    update_config(properties);
    validate();
}

void StructuredOutputConfig::update_config(const ov::AnyMap& properties) {
    using utils::read_anymap_param;

    read_anymap_param(properties, "json_schema", json_schema);
    read_anymap_param(properties, "regex", regex);
    read_anymap_param(properties, "grammar", grammar);
    read_anymap_param(properties, "structural_tags_config", structural_tags_config);
    read_anymap_param(properties, "compound_grammar", compound_grammar);
    read_anymap_param(properties, "backend", backend);
    read_anymap_param(properties, "enable_jump_forward", enable_jump_forward);
}

StructuredOutputConfig StructuredOutputConfig::from_model_format(const std::string& model_format,
                                                                  const JsonContainer& tools,
                                                                  const JsonContainer& tool_choice,
                                                                  const ModelStructuralTagOptions& options) {
    const auto& builders = registry();
    const auto it = builders.find(model_format);
    OPENVINO_ASSERT(it != builders.end(),
                    "Unknown format type: ", model_format,
                    ", supported types: ", supported_model_formats_string());

    StructuredOutputConfig config;
    config.structural_tags_config = it->second(normalize_tool_choice(tools, tool_choice), options);
    return config;
}

size_t GenerationConfig::get_max_new_tokens(size_t prompt_length) const {
    // max_new_tokens has priority over max_length, only if max_new_tokens was not specified use max_length
    if (max_new_tokens != SIZE_MAX) {
        return max_new_tokens;
    } else {
        OPENVINO_ASSERT(max_length > prompt_length, "Internal error: generation_config.max_length should be bigger than number of prompt tokens");
        return max_length - prompt_length;
    }
}

bool GenerationConfig::is_greedy_decoding() const {
    return !do_sample && !is_beam_search() && !is_tree_search();
}

bool GenerationConfig::is_beam_search() const {
    return num_beams > 1;
}

bool GenerationConfig::is_tree_search() const {
    return tree_depth > 0;
}

bool GenerationConfig::is_multinomial() const {
    return do_sample;
}

bool GenerationConfig::is_assisting_generation() const {
    return assistant_confidence_threshold > 0 || num_assistant_tokens > 0;
}

bool GenerationConfig::is_structured_output_generation() const {
    return structured_output_config.has_value();
}

bool GenerationConfig::is_prompt_lookup() const {
    return max_ngram_size > 0 && num_assistant_tokens > 0;
}

void GenerationConfig::validate() const {
    OPENVINO_ASSERT(num_return_sequences > 0, "num_return_sequences must be greater than 0");

    // Stop conditions

    OPENVINO_ASSERT(eos_token_id == -1 || stop_token_ids.find(eos_token_id) != stop_token_ids.end(),
        "'stop_token_ids' must contain 'eos_token_id'. Please, call 'set_eos_token_id' with 'eos_token_id' value");

    auto stop_token_ids_it = std::find_if(stop_token_ids.begin(), stop_token_ids.end(), [] (int64_t stop_token_id) -> bool {
        return stop_token_id < 0;
    });
    OPENVINO_ASSERT(stop_token_ids_it == stop_token_ids.end(), "'stop_token_ids' must be non-negative, but it contains a value ", *stop_token_ids_it);

    OPENVINO_ASSERT(!ignore_eos || max_new_tokens != SIZE_MAX || max_length != SIZE_MAX,
                    "ignore_eos is true, in this case either 'max_new_tokens', or 'max_length' should be defined.");

    OPENVINO_ASSERT(eos_token_id != -1 || !stop_token_ids.empty() || !stop_strings.empty() || max_new_tokens != SIZE_MAX || max_length != SIZE_MAX,
                    "Either 'eos_token_id', or 'stop_token_ids', or 'stop_strings', or 'max_new_tokens', or 'max_length' should be defined.");

    OPENVINO_ASSERT(max_new_tokens > 0 || (max_new_tokens == 0 && echo), "'max_new_tokens' must be greater than 0, if `echo` is set, 0 is also accepted");
    OPENVINO_ASSERT(min_new_tokens <= max_new_tokens, "min_new_tokens must be less or equal max_new_tokens");

    // Sampling strategies

    OPENVINO_ASSERT(num_return_sequences == 1 || (is_multinomial() || is_beam_search()), 
        "'num_return_sequences' can be more than 1 only in case of beam search or multinomial sampling, but got ", num_return_sequences);

    // generic penalties, but not supported by beam search currently
    if (!is_beam_search()) {
        OPENVINO_ASSERT(frequency_penalty >= -2.0f && frequency_penalty <= 2.0f, "'frequence_penalty' penalty must be within [-2.0; 2.0], but got ", frequency_penalty);
        OPENVINO_ASSERT(presence_penalty >= -2.0f && presence_penalty <= 2.0f, "'presence_penalty' penalty must be within [-2.0; 2.0], but got ", presence_penalty);
        OPENVINO_ASSERT(repetition_penalty > 0.0f, "'repetition_penalty' must be a strictly positive float, but got ", repetition_penalty);
    } else {
        OPENVINO_ASSERT(frequency_penalty == 0.0f, "'frequency_penalty' is not currently supported by beam search and should be 0.0f, but got ", frequency_penalty);
        OPENVINO_ASSERT(presence_penalty == 0.0f, "'presence_penalty' is not currently supported by beam search and should be 0.0f, but got ", presence_penalty);
        OPENVINO_ASSERT(repetition_penalty == 1.0f, "'repetition_penalty' is not currently supported by beam search and should be 1.0f, but got ", repetition_penalty);
    }

    if (is_multinomial()) {
        OPENVINO_ASSERT(top_p > 0 && top_p <= 1.0f, "When 'do_sample' is true, top_p must be a positive float > 0.0 and <= 1.0, but got ", top_p);
        OPENVINO_ASSERT(temperature > 0, "When 'do_sample' is true, temperature must be a strictly positive float, but got ", temperature);
        OPENVINO_ASSERT(min_p >= 0.0f && min_p < 1.0f, "When 'do_sample' is true, min_p must be in [0.0, 1.0), but got ", min_p);
    } else {
        // parameters requiring multinomial
        // OPENVINO_ASSERT(top_k == std::numeric_limits<size_t>::max(), "When 'do_sample' is false, top_k must be max of size_t, but got ", top_k);
        // OPENVINO_ASSERT(top_p == 1.0f, "When 'do_sample' is false, top_p must be 1.0f, but got ", top_p);
        // OPENVINO_ASSERT(temperature == 1.0f, "When 'do_sample' is false, temperature must be a 1.0f, but got ", temperature);
    }

    if (is_beam_search()) {
        OPENVINO_ASSERT(num_beams % num_beam_groups == 0, "'num_beams' (", num_beams, ") should be divisible by 'num_beam_groups' (", num_beam_groups, ")");
        OPENVINO_ASSERT(num_beams >= num_return_sequences, "'num_beams' (", num_beams, ") must be greater equal than 'num_return_sequences' (", num_return_sequences, ")");

        OPENVINO_ASSERT(!do_sample,
                        "Beam search with sampling is not supported yet. "
                        "Please either set do_sample=false to use beam search "
                        "or set num_beams=1 if you with to use multinomial sampling.");

        OPENVINO_ASSERT(no_repeat_ngram_size > 0, "'no_repeat_ngram_size' must be positive");
        if (num_beam_groups > 1) {
            OPENVINO_ASSERT(diversity_penalty != 0.0f, "For grouped beam search 'diversity_penalty' should not be zero, otherwise it fallbacks to non-grouped beam search");
        } else {
            OPENVINO_ASSERT(diversity_penalty == 0.0f, "For beam search 'diversity_penalty' is applicable only when grouped beam search is used, but got 'num_beam_groups' == 1");
        }
    } else {
        // parameters requiring beam search
        // OPENVINO_ASSERT(num_beam_groups == 1, "'num_beam_groups' is supported by beam search only and should be 1 otherwise, but got ", num_beam_groups);
        // OPENVINO_ASSERT(no_repeat_ngram_size == std::numeric_limits<size_t>::max(), "'no_repeat_ngram_size' is supported only by beam search, otherwise should be set to max of size_t, but got ", no_repeat_ngram_size);
        // OPENVINO_ASSERT(diversity_penalty == 0.0f, "'diversity_penalty' is set to ", diversity_penalty, " (default is 0.0f), which is supported only by beam search sampling");
        // OPENVINO_ASSERT(length_penalty == 1.0f, "'length_penalty' is set to ", length_penalty, " (default is 1.0f), which is supported only by beam search sampling");
    }

    // tree search (EAGLE)
    if (is_tree_search()) {
        OPENVINO_ASSERT(!do_sample,
                        "Tree search (EAGLE) is incompatible with do_sample=true; "
                        "set tree_depth=0 or do_sample=false");
        OPENVINO_ASSERT(num_beams == 1,
                        "Tree search (EAGLE) is incompatible with beam search; "
                        "set tree_depth=0 or num_beams=1");
        OPENVINO_ASSERT(branching_factor > 0,
                        "'branching_factor' must be > 0 when tree search is enabled, but got ",
                        branching_factor);
        OPENVINO_ASSERT(
            num_assistant_tokens > 0,
            "'num_assistant_tokens' must be > 0 when tree search is enabled, but got ",
            num_assistant_tokens);
        OPENVINO_ASSERT(num_assistant_tokens >= tree_depth,
                        "'num_assistant_tokens' (",
                        num_assistant_tokens,
                        ") must be >= 'tree_depth' (",
                        tree_depth,
                        ") to allow at least one node per draft layer");
    }

    // assistant generation

    if (is_assisting_generation()) {
        OPENVINO_ASSERT(!is_beam_search() && num_return_sequences == 1, "Beam search and parallel sampling are not compatible with assistant generation");
        OPENVINO_ASSERT(assistant_confidence_threshold == 0.0f || num_assistant_tokens == 0, "Parameters `assistant_confidence_threshold` and `num_assistant_tokens` are mutually exclusive in `GenerationConfig`");
    }

    if (num_assistant_tokens == 0) {
        OPENVINO_ASSERT(max_ngram_size == 0, "'max_ngram_size' should be set to default value 0 when prompt lookup is disabled");
    }

    if(is_structured_output_generation()) {
        (*structured_output_config).validate();
    }
}

void StructuredOutputConfig::validate() const {
    auto& registry = StructuredOutputController::get_backend_registry();
    std::string backend_name = backend.has_value() ? *backend : StructuredOutputController::get_default_backend_name();
    std::string upper_name = backend_name;
    std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), [](unsigned char c){ return std::toupper(c); });

    OPENVINO_ASSERT(registry.find(backend_name) != registry.end(),
                    "Structured output backend '", backend_name, "' is not registered. "
                    "Please recompile with -DENABLE_" + upper_name + "=ON option to enable it.");

    OPENVINO_ASSERT(
        (json_schema.has_value() + regex.has_value() + grammar.has_value() + structural_tags_config.has_value() + compound_grammar.has_value()) == 1,
        "Only one of json, regex, grammar, structural_tags_config, or compound_grammar should be set in StructuredOutputConfig, but got: ",
        (json_schema.has_value() ? "json=" + *json_schema +", " : ""),
        (regex.has_value() ? "regex=" + *regex + ", " : ""),
        (grammar.has_value() ? "grammar=" + *grammar : ""),
        (structural_tags_config.has_value() ? "structural_tags_config=" + std::visit([](const auto& config) -> std::string {
            if constexpr (std::is_same_v<std::decay_t<decltype(config)>, StructuralTagsConfig>) {
                return config.to_string();
            } else {
                return StructuredOutputConfig::structural_tag_to_string(config);
            }
        }, *structural_tags_config) : ""),
        (compound_grammar.has_value() ? "compound_grammar=" + std::visit([](const auto& g) -> std::string {
            return StructuredOutputConfig::structural_tag_to_string(g);
        }, *compound_grammar) : "")
    );
}

void StructuredOutputConfig::validate(Tokenizer& tokenizer) const {
    validate();
    OPENVINO_ASSERT(tokenizer.m_pimpl != nullptr, "Tokenizer not initialized properly");
    tokenizer.m_pimpl->get_structured_output_controller()->validate_grammar(*this);
}


std::shared_ptr<ov::genai::StructuredOutputConfig::Concat>
operator+(const ov::genai::StructuredOutputConfig::StructuralTag& lhs,
          const ov::genai::StructuredOutputConfig::StructuralTag& rhs) {
    using SOC = ov::genai::StructuredOutputConfig;
    const auto lhs_concat = std::get_if<std::shared_ptr<SOC::Concat>>(&lhs);
    const auto rhs_concat = std::get_if<std::shared_ptr<SOC::Concat>>(&rhs);

    if (lhs_concat && *lhs_concat) {
        // lhs is a Concat
        if (rhs_concat && *rhs_concat) {
            // both are Concat: combine elements
            std::vector<SOC::StructuralTag> elems = (*lhs_concat)->elements;
            elems.insert(elems.end(), (*rhs_concat)->elements.begin(), (*rhs_concat)->elements.end());
            return std::make_shared<SOC::Concat>(elems);
        } else {
            // only lhs is Concat: append rhs
            std::vector<SOC::StructuralTag> elems = (*lhs_concat)->elements;
            elems.push_back(rhs);
            return std::make_shared<SOC::Concat>(elems);
        }
    } else if (rhs_concat && *rhs_concat) {
        // only rhs is Concat: prepend lhs
        std::vector<SOC::StructuralTag> elems;
        elems.push_back(lhs);
        elems.insert(elems.end(), (*rhs_concat)->elements.begin(), (*rhs_concat)->elements.end());
        return std::make_shared<SOC::Concat>(elems);
    } else {
        // neither is Concat: create binary Concat
        return std::make_shared<SOC::Concat>(lhs, rhs);
    }
}

std::shared_ptr<ov::genai::StructuredOutputConfig::Union>
operator|(const ov::genai::StructuredOutputConfig::StructuralTag& lhs,
          const ov::genai::StructuredOutputConfig::StructuralTag& rhs) {
    using SOC = ov::genai::StructuredOutputConfig;
    const auto lhs_union = std::get_if<std::shared_ptr<SOC::Union>>(&lhs);
    const auto rhs_union = std::get_if<std::shared_ptr<SOC::Union>>(&rhs);

    if (lhs_union && *lhs_union) {
        if (rhs_union && *rhs_union) {
            // both are Union: combine elements
            std::vector<SOC::StructuralTag> elems = (*lhs_union)->elements;
            elems.insert(elems.end(), (*rhs_union)->elements.begin(), (*rhs_union)->elements.end());
            return std::make_shared<SOC::Union>(elems);
        } else {
            // only lhs is Union: append rhs
            std::vector<SOC::StructuralTag> elems = (*lhs_union)->elements;
            elems.push_back(rhs);
            return std::make_shared<SOC::Union>(elems);
        }
    } else if (rhs_union && *rhs_union) {
        // only rhs is Union: prepend lhs
        std::vector<SOC::StructuralTag> elems;
        elems.push_back(lhs);
        elems.insert(elems.end(), (*rhs_union)->elements.begin(), (*rhs_union)->elements.end());
        return std::make_shared<SOC::Union>(elems);
    } else {
        // neither is Union: create binary Union
        return std::make_shared<SOC::Union>(lhs, rhs);
    }
}

}  // namespace genai
}  // namespace ov
