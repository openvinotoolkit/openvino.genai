// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/chat_history.hpp"

namespace ov {
namespace genai {

ChatHistory::ChatHistory() = default;

ChatHistory::ChatHistory(const JsonContainer& messages) : m_messages(messages) {
    if (!m_messages.is_array()) {
        OPENVINO_THROW("Chat history must be initialized with a JSON array.");
    }
}
ChatHistory::ChatHistory(const std::vector<ov::AnyMap>& messages) :
    m_messages(JsonContainer::array()) {
    for (const auto& message : messages) {
        m_messages.push_back(JsonContainer(message));
    }
}

ChatHistory::ChatHistory(std::initializer_list<std::initializer_list<std::pair<std::string, ov::Any>>> messages) :
    m_messages(JsonContainer::array()) {
    for (const auto& message : messages) {
        m_messages.push_back(JsonContainer(message));
    }
}

ChatHistory::~ChatHistory() = default;

ChatHistory& ChatHistory::push_back(const JsonContainer& message) {
    m_messages.push_back(message);
    return *this;
}

ChatHistory& ChatHistory::push_back(const ov::AnyMap& message) {
    m_messages.push_back(JsonContainer(message));
    return *this;
}

ChatHistory& ChatHistory::push_back(std::initializer_list<std::pair<std::string, ov::Any>> message) {
    m_messages.push_back(JsonContainer(message));
    return *this;
}

void ChatHistory::pop_back() {
    if (m_messages.empty()) {
        OPENVINO_THROW("Cannot pop_back from an empty chat history.");
    }
    m_messages.erase(m_messages.size() - 1);
}

const JsonContainer& ChatHistory::get_messages() const {
    return m_messages;
}

JsonContainer& ChatHistory::get_messages() {
    return m_messages;
}

JsonContainer ChatHistory::operator[](size_t index) const {
    if (index >= m_messages.size()) {
        OPENVINO_THROW("Index ", index, " is out of bounds for chat history of size ", m_messages.size());
    }
    return m_messages[index];
}

JsonContainer ChatHistory::operator[](int index) const {
    return operator[](size_t(index));
}

JsonContainer ChatHistory::first() const {
    if (m_messages.empty()) {
        OPENVINO_THROW("Cannot access first message of an empty chat history.");
    }
    return m_messages[0];
}

JsonContainer ChatHistory::last() const {
    if (m_messages.empty()) {
        OPENVINO_THROW("Cannot access last message of an empty chat history.");
    }
    return m_messages[m_messages.size() - 1];
}

void ChatHistory::clear() {
    m_messages.clear();
}

size_t ChatHistory::size() const {
    return m_messages.size();
}

bool ChatHistory::empty() const {
    return m_messages.empty();
}

ChatHistory& ChatHistory::set_tools(const JsonContainer& tools) {
    if (!tools.is_array()) {
        OPENVINO_THROW("Tools must be an array-like JsonContainer.");
    }
    m_tools = tools;
    return *this;
}

const JsonContainer& ChatHistory::get_tools() const {
    return m_tools;
}

ChatHistory& ChatHistory::set_extra_context(const JsonContainer& extra_context) {
    if (!extra_context.is_object()) {
        OPENVINO_THROW("Extra context must be an object-like JsonContainer.");
    }
    m_extra_context = extra_context;
    return *this;
}

const JsonContainer& ChatHistory::get_extra_context() const {
    return m_extra_context;
}

} // namespace genai
} // namespace ov
