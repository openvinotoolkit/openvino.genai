// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/vision_registry.hpp"
#include "openvino/genai/chat_history.hpp"

namespace ov::genai {

/**
 * @brief Format of chat history user messages.
 * 
 * STRING_CONTENT: User message content is a string, e.g.:
 *  ```
 *  {"role": "user", "content": "What's in this image?"}
 *  ```
 * 
 * MULTIPART_CONTENT: User message content is an array of typed objects (OpenAI-like), e.g.:
 *  ```
 *  {"role": "user", "content": [
 *    {"type": "text", "text": "Describe this:"},
 *    {"type": "image"},
 *    {"type": "video"}
 *  ]}
 *  ```
 */
enum ChatHistoryFormat {
    UNKNOWN,
    STRING_CONTENT,
    MULTIPART_CONTENT,
};

struct MessageMetadata {
    // Original message for change detection
    JsonContainer original_message;
    
    // Contains vision placeholders after prompt normalization.
    // Empty for system/assistant messages.
    std::string normalized_content;
    
    // Global indices provided with corresponding message (input order)
    std::vector<size_t> provided_image_indices;
    std::vector<size_t> provided_video_indices;

    // Global indices in order of appearance in normalized content
    std::vector<size_t> image_sequence;
    std::vector<size_t> video_sequence;
    
    std::pair<size_t, size_t> get_vision_count() const {
        return {video_sequence.size(), image_sequence.size()};
    }
};

class ChatHistoryInternalState {
public:
    struct ResolvedVisions {
        std::vector<EncodedImage> encoded_images;
        std::vector<EncodedVideo> encoded_videos;
        std::vector<size_t> image_sequence;  // Indices into encoded_images
        std::vector<size_t> video_sequence;  // Indices into encoded_videos
    };

    ChatHistoryInternalState() = default;
    explicit ChatHistoryInternalState(const std::shared_ptr<VisionRegistry>& registry);
    ~ChatHistoryInternalState();

    ChatHistoryInternalState(const ChatHistoryInternalState&) = delete;
    ChatHistoryInternalState& operator=(const ChatHistoryInternalState&) = delete;

    ChatHistoryInternalState(ChatHistoryInternalState&&) = default;
    ChatHistoryInternalState& operator=(ChatHistoryInternalState&&) = default;

    const std::vector<MessageMetadata>& get_messages_metadata() const { return m_messages_metadata; }
    std::vector<MessageMetadata>& get_messages_metadata() { return m_messages_metadata; }
    
    void add_message_metadata(MessageMetadata metadata);

    std::vector<size_t> register_images(const std::vector<ov::Tensor>& images);
    std::vector<size_t> register_videos(const std::vector<ov::Tensor>& videos);

    VisionID get_image_vision_id(size_t index) const { return m_image_index_to_id.at(index); }
    VisionID get_video_vision_id(size_t index) const { return m_video_index_to_id.at(index); }

    size_t get_base_image_index() const { return m_image_index_to_id.size(); }
    size_t get_base_video_index() const { return m_video_index_to_id.size(); }

    std::vector<EncodedImage> get_encoded_images(const std::vector<size_t>& indices) const;
    std::vector<EncodedVideo> get_encoded_videos(const std::vector<size_t>& indices) const;

    ResolvedVisions resolve_visions_with_sequence(
        std::optional<const std::vector<size_t>> image_sequence = std::nullopt,
        std::optional<const std::vector<size_t>> video_sequence = std::nullopt
    ) const;

    std::vector<size_t> build_full_image_sequence() const;
    std::vector<size_t> build_full_video_sequence() const;
    std::vector<std::pair<size_t, size_t>> build_vision_counts() const;

    ChatHistory build_normalized_history(const ChatHistory& history) const;

    const size_t find_matching_history_length(const ChatHistory& history) const;

    void truncate_to(size_t size);

    void reset();

    ChatHistoryFormat get_chat_history_format() { return m_chat_history_format; }

    size_t get_last_user_message_index() const { return m_last_user_message_index; }

    static std::shared_ptr<ChatHistoryInternalState> get_or_create(
        const ov::genai::ChatHistory& history,
        const std::shared_ptr<VisionRegistry>& vision_registry = nullptr
    );

private:
    std::weak_ptr<VisionRegistry> m_vision_registry;

    ChatHistoryFormat m_chat_history_format = ChatHistoryFormat::UNKNOWN;

    // Global index to VisionID mapping
    std::vector<VisionID> m_image_index_to_id;
    std::vector<VisionID> m_video_index_to_id;

    std::vector<MessageMetadata> m_messages_metadata;

    size_t m_last_user_message_index;

    void set_vision_registry(const std::shared_ptr<VisionRegistry>& vision_registry);
    std::shared_ptr<VisionRegistry> get_vision_registry() const;

    void release_refs_from(size_t image_index, size_t video_index);

    void detect_chat_history_format(const ov::genai::ChatHistory& history);

    static size_t find_last_user_message_index(const ov::genai::ChatHistory& history);
};

} // namespace ov::genai
