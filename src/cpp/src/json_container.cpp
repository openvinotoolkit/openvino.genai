// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <regex>

#include "openvino/core/except.hpp"

#include "openvino/genai/json_container.hpp"
#include "json_utils.hpp"


namespace ov {
namespace genai {

JsonContainer::JsonContainer() :
    JsonContainer(nlohmann::ordered_json::object()) {}

JsonContainer::JsonContainer(std::initializer_list<std::pair<std::string, ov::Any>> init) :
    JsonContainer(ov::genai::utils::any_map_to_json(ov::AnyMap{init.begin(), init.end()})) {}

JsonContainer::JsonContainer(const ov::AnyMap& data) :
    JsonContainer(ov::genai::utils::any_map_to_json(data)) {}

JsonContainer::JsonContainer(ov::AnyMap&& data) :
    JsonContainer(ov::genai::utils::any_map_to_json(std::move(data))) {}

JsonContainer::JsonContainer(std::shared_ptr<nlohmann::ordered_json> json_ptr, const std::string& path) :
    m_json(json_ptr),
    m_path(path) {}

JsonContainer::JsonContainer(nlohmann::ordered_json json) :
    m_json(std::make_shared<nlohmann::ordered_json>(std::move(json))) {}

JsonContainer::JsonContainer(const JsonContainer& other) :
    m_json(std::make_shared<nlohmann::ordered_json>(*other.m_json)),
    m_path(other.m_path) {}

JsonContainer::JsonContainer(JsonContainer&& other) noexcept :
    m_json(std::move(other.m_json)),
    m_path(std::move(other.m_path)) {}

JsonContainer::~JsonContainer() = default;

JsonContainer JsonContainer::share() const {
    return JsonContainer(m_json, m_path);
}
    
JsonContainer JsonContainer::copy() const {
    return JsonContainer(*this);
}

JsonContainer JsonContainer::from_json_string(const std::string& json_str) {
    try {
        return JsonContainer(nlohmann::ordered_json::parse(json_str));
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to construct JsonContainer from JSON string: ", e.what());
    }
}

nlohmann::ordered_json* JsonContainer::get_json_value_ptr(AccessMode mode) const {
    auto json_pointer = nlohmann::ordered_json::json_pointer(m_path);
    if (mode == AccessMode::Read && !m_json->contains(json_pointer)) {
        OPENVINO_THROW("Path '", m_path, "' does not exist in the JsonContainer.");
    }
    return &(*m_json)[json_pointer];
}

JsonContainer& JsonContainer::operator=(const JsonContainer& other) {
    // if (this != &other) {
    //     m_json = other.m_json;
    //     m_path = other.m_path;
    // }
    // return *this;

    // Hybrid approach
    // if (this != &other) {
    //     if (!m_path.empty()) {
    //         // This is a path-based access - do value assignment
    //         nlohmann::ordered_json other_value = other.to_json();
    //         auto json_value_ptr = get_json_value_ptr(AccessMode::Write);
    //         *json_value_ptr = other_value;
    //     } else {
    //         // This is a root container - do standard assignment
    //         m_json = other.m_json;
    //         m_path = other.m_path;
    //     }
    // }
    if (this != &other) {
        auto json_value_ptr = get_json_value_ptr(AccessMode::Write);
        *json_value_ptr = other.to_json();
    }
    return *this;
}

JsonContainer& JsonContainer::operator=(JsonContainer&& other) noexcept {
    if (this != &other) {
        auto json_value_ptr = get_json_value_ptr(AccessMode::Write);
        if (m_json == other.m_json) {
            *json_value_ptr = std::move(*other.get_json_value_ptr(AccessMode::Read));
        } else {
            *json_value_ptr = other.to_json();
        }
    }
    return *this;
}

bool JsonContainer::operator==(const JsonContainer& other) const {
    return to_json() == other.to_json();
}

bool JsonContainer::operator!=(const JsonContainer& other) const {
    return !(*this == other);
}

JsonContainer JsonContainer::operator[](const std::string& key) const {
    return JsonContainer(m_json, m_path + "/" + key);
}

JsonContainer JsonContainer::operator[](const char* key) const {
    return operator[](std::string(key));
}

JsonContainer JsonContainer::operator[](size_t index) const {
    return JsonContainer(m_json, m_path + "/" + std::to_string(index));
}

JsonContainer JsonContainer::operator[](int index) const {
    return operator[](size_t(index));
}

nlohmann::ordered_json JsonContainer::to_json() const {
    return *get_json_value_ptr(AccessMode::Read);
}

std::string JsonContainer::to_json_string(int indent) const {
    return to_json().dump(indent);
}

bool JsonContainer::is_null() const {
    return get_json_value_ptr(AccessMode::Read)->is_null();
}

bool JsonContainer::is_boolean() const {
    return get_json_value_ptr(AccessMode::Read)->is_boolean();
}

bool JsonContainer::is_number() const {
    return get_json_value_ptr(AccessMode::Read)->is_number();
}

bool JsonContainer::is_number_integer() const {
    return get_json_value_ptr(AccessMode::Read)->is_number_integer();
}

bool JsonContainer::is_number_float() const {
    return get_json_value_ptr(AccessMode::Read)->is_number_float();
}

bool JsonContainer::is_string() const {
    return get_json_value_ptr(AccessMode::Read)->is_string();
}

bool JsonContainer::is_array() const {
    return get_json_value_ptr(AccessMode::Read)->is_array();
}

bool JsonContainer::is_object() const {
    return get_json_value_ptr(AccessMode::Read)->is_object();
}

std::optional<bool> JsonContainer::as_bool() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    return json_value_ptr->is_boolean() ? std::make_optional(json_value_ptr->get<bool>()) : std::nullopt;
}

std::optional<int64_t> JsonContainer::as_int() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    return json_value_ptr->is_number_integer() ? std::make_optional(json_value_ptr->get<int64_t>()) : std::nullopt;
}

std::optional<double> JsonContainer::as_double() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    return json_value_ptr->is_number() ? std::make_optional(json_value_ptr->get<double>()) : std::nullopt;
}

std::optional<std::string> JsonContainer::as_string() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    return json_value_ptr->is_string() ? std::make_optional(json_value_ptr->get<std::string>()) : std::nullopt;
}

bool JsonContainer::get_bool() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    if (!json_value_ptr->is_boolean()) {
        OPENVINO_THROW("JsonContainer expected boolean at path '", m_path, "' but found ",
            json_value_ptr->type_name(), " with value: ", json_value_ptr->dump());
    }
    return json_value_ptr->get<bool>();
}

int64_t JsonContainer::get_int() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    if (!json_value_ptr->is_number_integer()) {
        OPENVINO_THROW("JsonContainer expected integer number at path '", m_path, "' but found ",
            json_value_ptr->type_name(), " with value: ", json_value_ptr->dump());
    }
    return json_value_ptr->get<int64_t>();
}

double JsonContainer::get_double() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    if (!json_value_ptr->is_number_float()) {
        OPENVINO_THROW("JsonContainer expected floating-point number at path '", m_path, "' but found ",
            json_value_ptr->type_name(), " with value: ", json_value_ptr->dump());
    }
    return json_value_ptr->get<double>();
}

std::string JsonContainer::get_string() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    if (!json_value_ptr->is_string()) {
        OPENVINO_THROW("JsonContainer expected string at path '", m_path, "' but found ",
            json_value_ptr->type_name(), " with value: ", json_value_ptr->dump());
    }
    return json_value_ptr->get<std::string>();
}

JsonContainer& JsonContainer::set_object() {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Write);
    *json_value_ptr = nlohmann::ordered_json::object();
    return *this;
}

JsonContainer& JsonContainer::set_array() {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Write);
    *json_value_ptr = nlohmann::ordered_json::array();
    return *this;
}

JsonContainer& JsonContainer::push_back(const JsonContainer& item) {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Write);
    if (!json_value_ptr->is_array()) {
        *json_value_ptr = nlohmann::ordered_json::array();
    }
    json_value_ptr->push_back(item.to_json());
    return *this;
}

bool JsonContainer::contains(const std::string& key) const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    if (!json_value_ptr->is_object()) {
        return false;
    }
    return json_value_ptr->contains(key) && !(*json_value_ptr)[key].is_null();
}

size_t JsonContainer::size() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    return json_value_ptr->size();
}

bool JsonContainer::empty() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    return json_value_ptr->empty();
}

void JsonContainer::erase(const std::string& key) const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
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
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
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
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
    if (!json_value_ptr->is_structured()) {
        OPENVINO_THROW("JsonContainer clear is only supported for objects and arrays, but found ",
            json_value_ptr->type_name(), " at path '", m_path, "'");
    }
    json_value_ptr->clear();
}

std::string JsonContainer::type_name() const {
    auto json_value_ptr = get_json_value_ptr(AccessMode::Read);
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

} // namespace genai
} // namespace ov
