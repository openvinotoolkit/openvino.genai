// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/chat_history_state.hpp"

namespace ov::genai {

ChatHistoryInternalStateClass::ChatHistoryInternalStateClass(std::shared_ptr<VisionRegistry> vision_registry)
    : m_vision_registry(vision_registry) {}

ChatHistoryInternalStateClass::~ChatHistoryInternalStateClass() {
    release_all_references();
}

void ChatHistoryInternalStateClass::set_vision_registry(std::shared_ptr<VisionRegistry> vision_registry) {
    m_vision_registry = vision_registry;
}

std::shared_ptr<VisionRegistry> ChatHistoryInternalStateClass::get_vision_registry() const {
    return m_vision_registry.lock();
}

void ChatHistoryInternalStateClass::add_message_metadata(MessageMetadata metadata) {
    m_messages_metadata.push_back(std::move(metadata));
}

size_t ChatHistoryInternalStateClass::find_valid_prefix(const ChatHistory& history) const {
    size_t valid = 0;
    const size_t check_count = std::min(m_messages_metadata.size(), history.size());
    
    for (size_t i = 0; i < check_count; ++i) {
        std::string current_json = history[i].to_json_string();
        // TODO Consider passing JsonContainer
        if (!m_messages_metadata[i].matches(current_json)) {
            break;
        }
        valid = i + 1;
    }
    
    return std::min(valid, history.size());
}

void ChatHistoryInternalStateClass::truncate_to(size_t size) {
    if (size >= m_messages_metadata.size()) {
        return;
    }
    
    // Release vision references for removed messages
    if (auto vision_registry = m_vision_registry.lock()) {
        for (size_t i = size; i < m_messages_metadata.size(); ++i) {
            // vision_registry->release_refs(m_messages_metadata[i].provided_image_ids);
            // vision_registry->release_refs(m_messages_metadata[i].provided_video_ids);
            for (const auto& id : m_messages_metadata[i].provided_image_ids) {
                vision_registry->release_ref(id);
            }
            for (const auto& id : m_messages_metadata[i].provided_video_ids) {
                vision_registry->release_ref(id);
            }
        }
    }
    
    m_messages_metadata.resize(size);
    m_kv_cache_valid_messages = 0;
}

void ChatHistoryInternalStateClass::reset() {
    release_all_references();
    m_messages_metadata.clear();
    m_kv_cache_valid_messages = 0;
}

std::vector<VisionID> ChatHistoryInternalStateClass::build_full_image_sequence() const {
    std::vector<VisionID> sequence;
    for (const auto& metadata : m_messages_metadata) {
        sequence.insert(sequence.end(),
                       metadata.image_sequence.begin(),
                       metadata.image_sequence.end());
    }
    return sequence;
}

std::vector<VisionID> ChatHistoryInternalStateClass::build_full_video_sequence() const {
    std::vector<VisionID> sequence;
    for (const auto& metadata : m_messages_metadata) {
        sequence.insert(sequence.end(),
                       metadata.video_sequence.begin(),
                       metadata.video_sequence.end());
    }
    return sequence;
}

std::vector<std::pair<size_t, size_t>> ChatHistoryInternalStateClass::build_vision_counts() const {
    std::vector<std::pair<size_t, size_t>> vision_counts;
    vision_counts.reserve(m_messages_metadata.size());
    for (const auto& metadata : m_messages_metadata) {
        vision_counts.push_back(metadata.get_vision_count());
    }
    return vision_counts;
}

ChatHistory ChatHistoryInternalStateClass::build_normalized_history(
    const ChatHistory& history
) const {
    OPENVINO_ASSERT(m_messages_metadata.size() == history.size(),
                    "Internal state size (", m_messages_metadata.size(),
                    ") doesn't match history size (", history.size(), ")");
    
    ChatHistory normalized_history;
    normalized_history.set_tools(history.get_tools());
    normalized_history.set_extra_context(history.get_extra_context());
    for (size_t i = 0; i < history.size(); ++i) {
        // Start with copy of original message to preserve all fields
        JsonContainer normalized_msg = history[i].copy();

        // std::cout << "normalized_msg:\n" << normalized_msg.to_json_string(2) << std::endl;

        if (normalized_msg["role"].get_string() == "user") {
            // Replace content with normalized version
            normalized_msg["content"] = m_messages_metadata[i].normalized_content;
        }
        
        normalized_history.push_back(std::move(normalized_msg));
    }
    return normalized_history;
}

std::shared_ptr<ChatHistoryInternalStateClass> ChatHistoryInternalStateClass::get_or_create(
    ChatHistory& history,
    std::shared_ptr<VisionRegistry> vision_registry
) {
    auto state = std::dynamic_pointer_cast<ChatHistoryInternalStateClass>(
        history.get_internal_state_class()
    );
    
    if (!state) {
        state = std::make_shared<ChatHistoryInternalStateClass>(vision_registry);
        history.set_internal_state_class(state);
    } else if (vision_registry && !state->get_vision_registry()) {
        state->set_vision_registry(vision_registry);
    }
    return state;
}

void ChatHistoryInternalStateClass::release_all_references() {
    if (auto vision_registry = m_vision_registry.lock()) {
        for (const auto& metadata : m_messages_metadata) {
            // vision_registry->release_refs(metadata.provided_image_ids);
            // vision_registry->release_refs(metadata.provided_video_ids);
            for (const auto& id : metadata.image_sequence) {
                vision_registry->release_ref(id);
            }
            for (const auto& id : metadata.video_sequence) {
                vision_registry->release_ref(id);
            }
        }
    }
}

} // namespace ov::genai
