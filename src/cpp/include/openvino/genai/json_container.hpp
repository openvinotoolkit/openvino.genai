// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <initializer_list>

#include <nlohmann/json.hpp>

#include "openvino/core/any.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS JsonContainer {
private:
    template<typename T>
    struct is_json_primitive {
        using type = typename std::decay<T>::type;
        static constexpr bool value = 
            std::is_same<type, bool>::value ||
            std::is_same<type, int64_t>::value ||
            std::is_same<type, int>::value ||
            std::is_same<type, double>::value ||
            std::is_same<type, float>::value ||
            std::is_same<type, std::string>::value ||
            std::is_same<type, const char*>::value ||
            std::is_same<type, std::nullptr_t>::value;
    };

public:
    /**
     * @brief Default constructor creates an empty JSON object.
     */
    JsonContainer();

    /**
     * @brief Construct from JSON primitive types (bool, int64_t, double, string, etc.).
     */
    template<typename T>
    JsonContainer(T&& value, typename std::enable_if<is_json_primitive<T>::value, int>::type = 0) :
        JsonContainer(nlohmann::ordered_json(std::forward<T>(value))) {}

    /**
     * @brief Construct from initializer list of key-value pairs.
     * 
     * Example:
     * JsonContainer({{"role", "user"}, {"content", "hello"}})
     */
    JsonContainer(std::initializer_list<std::pair<std::string, ov::Any>> init);

    /**
     * @brief Construct from AnyMap.
     */
    explicit JsonContainer(const ov::AnyMap& data);

    /**
     * @brief Construct from AnyMap (move version).
     */
    explicit JsonContainer(ov::AnyMap&& data);

    /**
     * @brief Copy constructor.
     */
    JsonContainer(const JsonContainer& other);

    /**
     * @brief Move constructor.
     */
    JsonContainer(JsonContainer&& other) noexcept;

    ~JsonContainer();

    /**
     * @brief Create a shared copy of this JsonContainer.
     */
    JsonContainer share() const;

    /**
     * @brief Create a deep copy of this JsonContainer.
     */
    JsonContainer copy() const;

    /**
     * @brief Create JsonContainer from JSON string.
     * @param json_str Valid JSON string
     * @throw ov::Exception if parsing fails
     */
    static JsonContainer from_json_string(const std::string& json_str);

    /**
     * @brief Assignment operator for JSON primitive types (bool, int64_t, double, string, etc.).
     */
    template<typename T>
    typename std::enable_if<is_json_primitive<T>::value, JsonContainer&>::type
    operator=(T&& value) {
        auto json_value_ptr = get_json_value_ptr(AccessMode::Write);
        *json_value_ptr = nlohmann::ordered_json(std::forward<T>(value));
        return *this;
    }

    /**
     * @brief Copy assignment operator.
     */
    JsonContainer& operator=(const JsonContainer& other);

    /**
     * @brief Move assignment operator.
     */
    JsonContainer& operator=(JsonContainer&& other) noexcept;

    bool operator==(const JsonContainer& other) const;
    bool operator!=(const JsonContainer& other) const;

    JsonContainer operator[](const std::string& key) const;
    JsonContainer operator[](const char* key) const;
    JsonContainer operator[](size_t index) const;
    JsonContainer operator[](int index) const;

    bool is_null() const;
    bool is_boolean() const;
    bool is_number() const;
    bool is_number_integer() const;
    bool is_number_float() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;

    std::optional<bool> as_bool() const;
    std::optional<int64_t> as_int() const;
    std::optional<double> as_double() const;
    std::optional<std::string> as_string() const;

    bool get_bool() const;
    int64_t get_int() const;
    double get_double() const;
    std::string get_string() const;

    /**
     * @brief Add JsonContainer to end of array.
     * If this container is not an array, it will be converted to an array.
     * @param value JsonContainer to append
     * @return Reference to this container for chaining
     */
    JsonContainer& push_back(const JsonContainer& value);

    /**
     * @brief Add JSON primitive to end of array.
     * If this container is not an array, it will be converted to an array.
     * @param value JSON primitive to append (bool, int64_t, double, string, etc.)
     * @return Reference to this container for chaining
     */
    template<typename T>
    typename std::enable_if<is_json_primitive<T>::value, JsonContainer&>::type
    push_back(T&& value) {
        auto json_value_ptr = get_json_value_ptr(AccessMode::Write);
        if (!json_value_ptr->is_array()) {
            *json_value_ptr = nlohmann::ordered_json::array();
        }
        json_value_ptr->push_back(nlohmann::ordered_json(std::forward<T>(value)));
        return *this;
    }

    /**
     * @brief Convert this container to an empty object.
     * @return Reference to this container for chaining
     */
    JsonContainer& set_object();

    /**
     * @brief Convert this container to an empty array.
     * @return Reference to this container for chaining
     */
    JsonContainer& set_array();

    /**
     * @brief Check if object contains a key.
     * Returns false if container is not an object.
     */
    bool contains(const std::string& key) const;

    /**
     * @brief Get container size.
     * @return Number of elements (object members or array elements), 1 for primitives, 0 for null
     */
    size_t size() const;

    /**
     * @brief Check if container is empty.
     * @return true if container has no elements (or is null), false for primitives
     */
    bool empty() const;

    /**
     * @brief Erase a key from an object.
     * @throw ov::Exception if container is not an object or key does not exist.
     */
    void erase(const std::string& key) const;

    /**
     * @brief Erase an element from an array by index with shifting.
     * @throw ov::Exception if container is not an array or index is out of bounds.
     */
    void erase(size_t index) const;

    /**
     * @brief Clear all contents of the container (array or object).
     * @throw ov::Exception if container is not an array or object.
     */
    void clear() const;

    /**
     * @brief Convert to JSON string.
     * @param indent Indentation level (-1 for compact output)
     * @return JSON string representation
     */
    std::string to_json_string(int indent = -1) const;

    /**
     * @brief Convert to nlohmann::ordered_json for internal use.
     * @return nlohmann::ordered_json representation
     */
    nlohmann::ordered_json to_json() const;

    /**
     * @brief Get string representation of the JSON type.
     * @return Type name: "null", "boolean", "number", "string", "array", "object" or "unknown"
     */
    std::string type_name() const;

private:
    JsonContainer(std::shared_ptr<nlohmann::ordered_json> json_ptr, const std::string& path);
    JsonContainer(nlohmann::ordered_json json);

    std::shared_ptr<nlohmann::ordered_json> m_json;

    std::string m_path = "";

    enum class AccessMode { Read, Write };
    
    nlohmann::ordered_json* get_json_value_ptr(AccessMode mode) const;
};

} // namespace genai
} // namespace ov
