// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include "visual_language/vision_encoder.hpp" // TODO Remove after migrating to class
#include "visual_language/vision_registry.hpp"
#include "openvino/genai/chat_history.hpp"

namespace ov::genai {

struct ChatHistoryInternalState {
    size_t processed_history_size = 0;
    
    std::vector<ov::genai::EncodedImage> encoded_images;
    std::vector<ov::genai::EncodedVideo> encoded_videos;
    
    size_t image_id = 0;
    size_t video_id = 0;

    std::vector<size_t> image_sequence;
    std::vector<size_t> video_sequence;

    std::vector<std::pair<std::size_t, std::size_t>> vision_count;  // pair<video count, image count>
    
    // TODO Detect new chat and calculate messages diff e.g. via hash
    bool is_continuation(size_t history_size) const {
        if (processed_history_size == 0) {
            return false;
        }
        return history_size == processed_history_size + 2; // assistant response and last user messages are added to history manually
    }

    static std::shared_ptr<ChatHistoryInternalState> get_or_create(ChatHistory& history) {
        auto state = history.get_internal_state();
        if (!state) {
            state = std::make_shared<ChatHistoryInternalState>();
            history.set_internal_state(state);
        }
        return state;
    }
};

struct MessageMetadata {
    std::string original_message_json;
    
    // TODO Consider using full serialized message json
    std::string normalized_content;
    
    // VisionIDs provided WITH this message (for reference counting)
    std::vector<VisionID> provided_image_ids;
    std::vector<VisionID> provided_video_ids;
    
    // VisionIDs appearing in normalized content, in order of appearance
    std::vector<VisionID> image_sequence;
    std::vector<VisionID> video_sequence;
    
    // TODO Check if needed
    size_t compute_hash() const {
        return std::hash<std::string>{}(original_message_json);
    }
    
    bool matches(const std::string& message_json) const {
        return original_message_json == message_json;
    }
    
    std::pair<size_t, size_t> get_vision_count() const {
        return {video_sequence.size(), image_sequence.size()};
    }
};

class ChatHistoryInternalStateClass {
public:
    // TODO Check constructors and destructors
    ChatHistoryInternalStateClass() = default;
    explicit ChatHistoryInternalStateClass(std::shared_ptr<VisionRegistry> registry);
    ~ChatHistoryInternalStateClass();

    ChatHistoryInternalStateClass(const ChatHistoryInternalStateClass&) = delete;
    ChatHistoryInternalStateClass& operator=(const ChatHistoryInternalStateClass&) = delete;

    ChatHistoryInternalStateClass(ChatHistoryInternalStateClass&&) = default;
    ChatHistoryInternalStateClass& operator=(ChatHistoryInternalStateClass&&) = default;

    void set_vision_registry(std::shared_ptr<VisionRegistry> vision_registry);
    
    std::shared_ptr<VisionRegistry> get_vision_registry() const;

    const std::vector<MessageMetadata>& get_messages_metadata() const { return m_messages_metadata; }
    std::vector<MessageMetadata>& get_messages_metadata() { return m_messages_metadata; }

    size_t messages_metadata_size() const { return m_messages_metadata.size(); }
    
    // TODO Check if index getter needed
    const MessageMetadata& get_message_metadata(size_t index) const { return m_messages_metadata.at(index); }
    MessageMetadata& get_message_metadata(size_t index) { return m_messages_metadata.at(index); }
    
    void add_message_metadata(MessageMetadata metadata);

    size_t find_valid_prefix(const ChatHistory& history) const;

    void truncate_to(size_t size);

    // TODO Check if needed
    void reset();

    std::vector<VisionID> build_full_image_sequence() const;
    std::vector<VisionID> build_full_video_sequence() const;
    std::vector<std::pair<size_t, size_t>> build_vision_counts() const;

    ChatHistory build_normalized_history(const ChatHistory& history) const;

    size_t get_kv_cache_valid_messages() const { return m_kv_cache_valid_messages; }
    void set_kv_cache_valid_messages(size_t count) { m_kv_cache_valid_messages = count; }

    static std::shared_ptr<ChatHistoryInternalStateClass> get_or_create(
        ChatHistory& history,
        std::shared_ptr<VisionRegistry> vision_registry = nullptr
    );

private:
    std::vector<MessageMetadata> m_messages_metadata;
    std::weak_ptr<VisionRegistry> m_vision_registry;
    size_t m_kv_cache_valid_messages = 0;

    void release_all_references();
};

} // namespace ov::genai
