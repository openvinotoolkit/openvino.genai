// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/json_container.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS ChatHistory {
public:
    ChatHistory();

    explicit ChatHistory(const JsonContainer& messages);

    explicit ChatHistory(const std::vector<ov::AnyMap>& messages);

    /**
     * @brief Construct from initializer list for convenient inline creation.
     * 
     * Example:
     * ChatHistory history({
     *     {{"role", "system"}, {"content", "You are helpful assistant."}},
     *     {{"role", "user"}, {"content", "Hello"}}
     * });
     */
    ChatHistory(std::initializer_list<std::initializer_list<std::pair<std::string, ov::Any>>> messages);

    ~ChatHistory();

    ChatHistory& push_back(const JsonContainer& message);
    ChatHistory& push_back(const ov::AnyMap& message);
    ChatHistory& push_back(std::initializer_list<std::pair<std::string, ov::Any>> message);

    void pop_back();
    
    const JsonContainer& get_messages() const;
    JsonContainer& get_messages();

    JsonContainer operator[](size_t index) const;
    JsonContainer operator[](int index) const;

    JsonContainer first() const;
    JsonContainer last() const;

    void clear();

    size_t size() const;
    bool empty() const;

private:
    JsonContainer m_messages;
};

} // namespace genai
} // namespace ov
