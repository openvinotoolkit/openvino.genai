// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/core/except.hpp"

#include <nlohmann/json.hpp>

#include "openvino/genai/json_container.hpp"
#include "json_utils.hpp"


namespace ov {
namespace genai {

enum class AccessMode { Read, Write };

class JsonContainer::JsonContainerImpl {
public:
    JsonContainerImpl() : m_json(nlohmann::ordered_json::object()) {}
    explicit JsonContainerImpl(nlohmann::ordered_json json) : m_json(std::move(json)) {}

    nlohmann::ordered_json* get_json_value_ptr(const std::string& path, AccessMode mode) {
        try {
            auto json_pointer = nlohmann::ordered_json::json_pointer(path);
            if (mode == AccessMode::Read && !m_json.contains(json_pointer)) {
                OPENVINO_THROW("Path '", path, "' does not exist in the JsonContainer.");
            }
            return &m_json[json_pointer];
        } catch (const nlohmann::json::exception& e) {
            OPENVINO_THROW("Invalid JSON path '", path, "': ", e.what());
        }
    }
    
    const nlohmann::ordered_json* get_json_value_ptr(const std::string& path, AccessMode mode) const {
        return const_cast<JsonContainerImpl*>(this)->get_json_value_ptr(path, mode);
    }

private:
    nlohmann::ordered_json m_json;
};

JsonContainer::JsonContainer() :
    m_impl(std::make_shared<JsonContainerImpl>()) {}

JsonContainer::JsonContainer(bool value) :
    m_impl(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json(value))) {}
JsonContainer::JsonContainer(int value) :
    m_impl(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json(value))) {}
JsonContainer::JsonContainer(int64_t value) :
    m_impl(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json(value))) {}
JsonContainer::JsonContainer(double value) :
    m_impl(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json(value))) {}
JsonContainer::JsonContainer(float value) :
    m_impl(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json(value))) {}
JsonContainer::JsonContainer(const std::string& value) :
    m_impl(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json(value))) {}
JsonContainer::JsonContainer(const char* value) :
    m_impl(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json(value))) {}
JsonContainer::JsonContainer(std::nullptr_t) :
    m_impl(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json(nullptr))) {}

JsonContainer::JsonContainer(std::initializer_list<std::pair<std::string, ov::Any>> init) :
    m_impl(std::make_shared<JsonContainerImpl>(ov::genai::utils::any_map_to_json(ov::AnyMap{init.begin(), init.end()}))) {}

JsonContainer::JsonContainer(const ov::AnyMap& data) :
    m_impl(std::make_shared<JsonContainerImpl>(ov::genai::utils::any_map_to_json(data))) {}

JsonContainer::JsonContainer(ov::AnyMap&& data) :
    m_impl(std::make_shared<JsonContainerImpl>(ov::genai::utils::any_map_to_json(std::move(data)))) {}

JsonContainer::JsonContainer(std::shared_ptr<JsonContainerImpl> impl, const std::string& path) :
    m_impl(std::move(impl)),
    m_path(path) {}

JsonContainer::JsonContainer(const JsonContainer& other) :
    m_impl(std::make_shared<JsonContainerImpl>(*other.m_impl->get_json_value_ptr(other.m_path, AccessMode::Read))),
    m_path(other.m_path) {}

JsonContainer::JsonContainer(JsonContainer&& other) noexcept :
    m_impl(std::move(other.m_impl)),
    m_path(std::move(other.m_path)) {}

JsonContainer::~JsonContainer() = default;

JsonContainer JsonContainer::share() const {
    return JsonContainer(m_impl, m_path);
}
    
JsonContainer JsonContainer::copy() const {
    return JsonContainer(*this);
}

JsonContainer JsonContainer::from_json_string(const std::string& json_str) {
    try {
        return JsonContainer(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json::parse(json_str)));
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to construct JsonContainer from JSON string: ", e.what());
    }
}

JsonContainer JsonContainer::object() {
    return JsonContainer(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json::object()));
}

JsonContainer JsonContainer::array() {
    return JsonContainer(std::make_shared<JsonContainerImpl>(nlohmann::ordered_json::array()));
}

#define JSON_CONTAINER_PRIMITIVE_ASSIGNMENT(type) \
    JsonContainer& JsonContainer::operator=(type value) { \
        auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Write); \
        *json_value_ptr = nlohmann::ordered_json(value); \
        return *this; \
    }

JSON_CONTAINER_PRIMITIVE_ASSIGNMENT(bool)
JSON_CONTAINER_PRIMITIVE_ASSIGNMENT(int)
JSON_CONTAINER_PRIMITIVE_ASSIGNMENT(int64_t)
JSON_CONTAINER_PRIMITIVE_ASSIGNMENT(double)
JSON_CONTAINER_PRIMITIVE_ASSIGNMENT(float)
JSON_CONTAINER_PRIMITIVE_ASSIGNMENT(const std::string&)
JSON_CONTAINER_PRIMITIVE_ASSIGNMENT(const char*)

#undef JSON_CONTAINER_PRIMITIVE_ASSIGNMENT

JsonContainer& JsonContainer::operator=(std::nullptr_t) {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Write);
    *json_value_ptr = nlohmann::ordered_json(nullptr);
    return *this;
}

JsonContainer& JsonContainer::operator=(const JsonContainer& other) {
    if (this != &other) {
        auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Write);
        *json_value_ptr = *other.m_impl->get_json_value_ptr(other.m_path, AccessMode::Read);
    }
    return *this;
}

JsonContainer& JsonContainer::operator=(JsonContainer&& other) noexcept {
    if (this != &other) {
        auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Write);
        auto other_json_value_ptr = other.m_impl->get_json_value_ptr(other.m_path, AccessMode::Read);
        if (m_impl == other.m_impl) {
            *json_value_ptr = std::move(*other_json_value_ptr);
        } else {
            *json_value_ptr = *other_json_value_ptr;
        }
    }
    return *this;
}

bool JsonContainer::operator==(const JsonContainer& other) const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    auto other_json_value_ptr = other.m_impl->get_json_value_ptr(other.m_path, AccessMode::Read);
    return *json_value_ptr == *other_json_value_ptr;
}

bool JsonContainer::operator!=(const JsonContainer& other) const {
    return !(*this == other);
}

JsonContainer JsonContainer::operator[](const std::string& key) const {
    return JsonContainer(m_impl, m_path + "/" + key);
}

JsonContainer JsonContainer::operator[](const char* key) const {
    return operator[](std::string(key));
}

JsonContainer JsonContainer::operator[](size_t index) const {
    return JsonContainer(m_impl, m_path + "/" + std::to_string(index));
}

JsonContainer JsonContainer::operator[](int index) const {
    return operator[](size_t(index));
}

std::string JsonContainer::to_json_string(int indent) const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read)->dump(indent);
}

bool JsonContainer::is_null() const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read)->is_null();
}

bool JsonContainer::is_boolean() const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read)->is_boolean();
}

bool JsonContainer::is_number() const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read)->is_number();
}

bool JsonContainer::is_number_integer() const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read)->is_number_integer();
}

bool JsonContainer::is_number_float() const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read)->is_number_float();
}

bool JsonContainer::is_string() const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read)->is_string();
}

bool JsonContainer::is_array() const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read)->is_array();
}

bool JsonContainer::is_object() const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read)->is_object();
}

std::optional<bool> JsonContainer::as_bool() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    return json_value_ptr->is_boolean() ? std::make_optional(json_value_ptr->get<bool>()) : std::nullopt;
}

std::optional<int64_t> JsonContainer::as_int() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    return json_value_ptr->is_number_integer() ? std::make_optional(json_value_ptr->get<int64_t>()) : std::nullopt;
}

std::optional<double> JsonContainer::as_double() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    return json_value_ptr->is_number() ? std::make_optional(json_value_ptr->get<double>()) : std::nullopt;
}

std::optional<std::string> JsonContainer::as_string() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    return json_value_ptr->is_string() ? std::make_optional(json_value_ptr->get<std::string>()) : std::nullopt;
}

bool JsonContainer::get_bool() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    if (!json_value_ptr->is_boolean()) {
        OPENVINO_THROW("JsonContainer expected boolean at path '", m_path, "' but found ",
            json_value_ptr->type_name(), " with value: ", json_value_ptr->dump());
    }
    try {
        return json_value_ptr->get<bool>();
    } catch (const nlohmann::json::exception& e) {
        OPENVINO_THROW("Failed to get boolean value at path '", m_path, "': ", e.what());
    }
}

int64_t JsonContainer::get_int() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    if (!json_value_ptr->is_number_integer()) {
        OPENVINO_THROW("JsonContainer expected integer number at path '", m_path, "' but found ",
            json_value_ptr->type_name(), " with value: ", json_value_ptr->dump());
    }
    try {
        return json_value_ptr->get<int64_t>();
    } catch (const nlohmann::json::exception& e) {
        OPENVINO_THROW("Failed to get integer value at path '", m_path, "': ", e.what());
    }
}

double JsonContainer::get_double() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    if (!json_value_ptr->is_number_float()) {
        OPENVINO_THROW("JsonContainer expected floating-point number at path '", m_path, "' but found ",
            json_value_ptr->type_name(), " with value: ", json_value_ptr->dump());
    }
    try {
        return json_value_ptr->get<double>();
    } catch (const nlohmann::json::exception& e) {
        OPENVINO_THROW("Failed to get double value at path '", m_path, "': ", e.what());
    }
}

std::string JsonContainer::get_string() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    if (!json_value_ptr->is_string()) {
        OPENVINO_THROW("JsonContainer expected string at path '", m_path, "' but found ",
            json_value_ptr->type_name(), " with value: ", json_value_ptr->dump());
    }
    try {
        return json_value_ptr->get<std::string>();
    } catch (const nlohmann::json::exception& e) {
        OPENVINO_THROW("Failed to get string value at path '", m_path, "': ", e.what());
    }
}

JsonContainer& JsonContainer::to_empty_object() {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Write);
    *json_value_ptr = nlohmann::ordered_json::object();
    return *this;
}

JsonContainer& JsonContainer::to_empty_array() {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Write);
    *json_value_ptr = nlohmann::ordered_json::array();
    return *this;
}

JsonContainer& JsonContainer::push_back(const JsonContainer& item) {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Write);
    if (!json_value_ptr->is_array()) {
        *json_value_ptr = nlohmann::ordered_json::array();
    }
    json_value_ptr->push_back(*item.m_impl->get_json_value_ptr(item.m_path, AccessMode::Read));
    return *this;
}

#define JSON_CONTAINER_PRIMITIVE_PUSH_BACK(type) \
    JsonContainer& JsonContainer::push_back(type value) { \
        auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Write); \
        if (!json_value_ptr->is_array()) { \
            *json_value_ptr = nlohmann::ordered_json::array(); \
        } \
        json_value_ptr->push_back(nlohmann::ordered_json(value)); \
        return *this; \
    }

JSON_CONTAINER_PRIMITIVE_PUSH_BACK(bool)
JSON_CONTAINER_PRIMITIVE_PUSH_BACK(int)
JSON_CONTAINER_PRIMITIVE_PUSH_BACK(int64_t)
JSON_CONTAINER_PRIMITIVE_PUSH_BACK(double)
JSON_CONTAINER_PRIMITIVE_PUSH_BACK(float)
JSON_CONTAINER_PRIMITIVE_PUSH_BACK(const std::string&)
JSON_CONTAINER_PRIMITIVE_PUSH_BACK(const char*)

#undef JSON_CONTAINER_PRIMITIVE_PUSH_BACK

JsonContainer& JsonContainer::push_back(std::nullptr_t) {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Write);
    if (!json_value_ptr->is_array()) {
        *json_value_ptr = nlohmann::ordered_json::array();
    }
    json_value_ptr->push_back(nlohmann::ordered_json(nullptr));
    return *this;
}

bool JsonContainer::contains(const std::string& key) const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    if (!json_value_ptr->is_object()) {
        return false;
    }
    return json_value_ptr->contains(key) && !(*json_value_ptr)[key].is_null();
}

size_t JsonContainer::size() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    return json_value_ptr->size();
}

bool JsonContainer::empty() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    return json_value_ptr->empty();
}

void JsonContainer::erase(const std::string& key) const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    if (!json_value_ptr->is_object()) {
        OPENVINO_THROW("JsonContainer erase by key is only supported for objects, but found ",
            json_value_ptr->type_name(), " at path '", m_path, "'");
    }
    auto erased_count = json_value_ptr->erase(key);
    if (erased_count == 0) {
        OPENVINO_THROW("JsonContainer erase key '", key, "' does not exist in the object at path '", m_path, "'");
    }
}

void JsonContainer::erase(size_t index) const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    if (!json_value_ptr->is_array()) {
        OPENVINO_THROW("JsonContainer erase by index is only supported for arrays, but found ",
            json_value_ptr->type_name(), " at path '", m_path, "'");
    }
    if (index >= json_value_ptr->size()) {
        OPENVINO_THROW("JsonContainer erase index ", index, " is out of bounds for array of size ",
            json_value_ptr->size(), " at path '", m_path, "'");
    }
    json_value_ptr->erase(json_value_ptr->begin() + index);
}

void JsonContainer::clear() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    if (!json_value_ptr->is_structured()) {
        OPENVINO_THROW("JsonContainer clear is only supported for objects and arrays, but found ",
            json_value_ptr->type_name(), " at path '", m_path, "'");
    }
    json_value_ptr->clear();
}

std::string JsonContainer::type_name() const {
    auto json_value_ptr = m_impl->get_json_value_ptr(m_path, AccessMode::Read);
    if (json_value_ptr->is_null()) {
        return "null";
    } else if (json_value_ptr->is_boolean()) {
        return "boolean";
    } else if (json_value_ptr->is_number()) {
        return "number";
    } else if (json_value_ptr->is_string()) {
        return "string";
    } else if (json_value_ptr->is_array()) {
        return "array";
    } else if (json_value_ptr->is_object()) {
        return "object";
    } else {
        return "unknown";
    }
}

void* JsonContainer::_get_json_value_ptr() const {
    return m_impl->get_json_value_ptr(m_path, AccessMode::Read);
}

} // namespace genai
} // namespace ov
