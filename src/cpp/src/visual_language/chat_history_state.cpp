// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/chat_history_state.hpp"

namespace ov::genai {

ChatHistoryInternalState::ChatHistoryInternalState(std::shared_ptr<VisionRegistry> vision_registry)
    : m_vision_registry(vision_registry) {}

ChatHistoryInternalState::~ChatHistoryInternalState() {
    reset();
}

void ChatHistoryInternalState::set_vision_registry(std::shared_ptr<VisionRegistry> vision_registry) {
    m_vision_registry = vision_registry;
}

std::shared_ptr<VisionRegistry> ChatHistoryInternalState::get_vision_registry() const {
    auto vision_registry = m_vision_registry.lock();
    OPENVINO_ASSERT(vision_registry, "VisionRegistry is not available");
    return vision_registry;
}

void ChatHistoryInternalState::add_message_metadata(MessageMetadata metadata) {
    m_messages_metadata.push_back(std::move(metadata));
}

std::vector<size_t> ChatHistoryInternalState::register_images(const std::vector<ov::Tensor>& images) {
    auto vision_registry = get_vision_registry();
    
    std::vector<size_t> indices;
    indices.reserve(images.size());
    
    for (const auto& image : images) {
        VisionID id = vision_registry->register_image(image);
        size_t global_idx = m_image_index_to_id.size();
        m_image_index_to_id.push_back(id);
        indices.push_back(global_idx);
    }
    return indices;
}

std::vector<size_t> ChatHistoryInternalState::register_videos(const std::vector<ov::Tensor>& videos) {
    auto vision_registry = get_vision_registry();
    
    std::vector<size_t> indices;
    indices.reserve(videos.size());
    
    for (const auto& video : videos) {
        VisionID id = vision_registry->register_video(video);
        size_t global_idx = m_video_index_to_id.size();
        m_video_index_to_id.push_back(id);
        indices.push_back(global_idx);
    }
    return indices;
}

std::vector<EncodedImage> ChatHistoryInternalState::get_encoded_images(const std::vector<size_t>& indices) const {
    auto vision_registry = get_vision_registry();
    
    std::vector<EncodedImage> result;
    result.reserve(indices.size());
    for (size_t idx : indices) {
        VisionID id = m_image_index_to_id.at(idx);
        result.push_back(vision_registry->get_encoded_image(id));
    }
    return result;
}

std::vector<EncodedVideo> ChatHistoryInternalState::get_encoded_videos(const std::vector<size_t>& indices) const {
    auto vision_registry = get_vision_registry();
    
    std::vector<EncodedVideo> result;
    result.reserve(indices.size());
    for (size_t idx : indices) {
        VisionID id = m_video_index_to_id.at(idx);
        result.push_back(vision_registry->get_encoded_video(id));
    }
    return result;
}

ChatHistoryInternalState::ResolvedVisions ChatHistoryInternalState::resolve_visions_with_sequence(
    std::optional<const std::vector<size_t>> image_sequence,
    std::optional<const std::vector<size_t>> video_sequence
) const {
    auto vision_registry = get_vision_registry();
    
    ResolvedVisions result;
    
    std::vector<size_t> _image_sequence = image_sequence.value_or(build_full_image_sequence());
    std::vector<size_t> _video_sequence = video_sequence.value_or(build_full_video_sequence());
    
    std::unordered_map<VisionID, size_t> image_id_to_local_index;
    for (size_t global_idx : _image_sequence) {
        VisionID id = m_image_index_to_id.at(global_idx);
        
        auto it = image_id_to_local_index.find(id);
        if (it == image_id_to_local_index.end()) {
            size_t local_idx = result.encoded_images.size();
            image_id_to_local_index[id] = local_idx;
            result.encoded_images.push_back(vision_registry->get_encoded_image(id));
            result.image_sequence.push_back(local_idx);
        } else {
            result.image_sequence.push_back(it->second);
        }
    }
    
    std::unordered_map<VisionID, size_t> video_id_to_local_index;
    for (size_t global_idx : _video_sequence) {
        VisionID id = m_video_index_to_id.at(global_idx);
        
        auto it = video_id_to_local_index.find(id);
        if (it == video_id_to_local_index.end()) {
            size_t local_idx = result.encoded_videos.size();
            video_id_to_local_index[id] = local_idx;
            result.encoded_videos.push_back(vision_registry->get_encoded_video(id));
            result.video_sequence.push_back(local_idx);
        } else {
            result.video_sequence.push_back(it->second);
        }
    }
    
    return result;
}

std::vector<size_t> ChatHistoryInternalState::build_full_image_sequence() const {
    std::vector<size_t> sequence;
    for (const auto& metadata : m_messages_metadata) {
        sequence.insert(sequence.end(),
                        metadata.image_sequence.begin(),
                        metadata.image_sequence.end());
    }
    return sequence;
}

std::vector<size_t> ChatHistoryInternalState::build_full_video_sequence() const {
    std::vector<size_t> sequence;
    for (const auto& metadata : m_messages_metadata) {
        sequence.insert(sequence.end(),
                        metadata.video_sequence.begin(),
                        metadata.video_sequence.end());
    }
    return sequence;
}

std::vector<std::pair<size_t, size_t>> ChatHistoryInternalState::build_vision_counts() const {
    std::vector<std::pair<size_t, size_t>> vision_counts;
    vision_counts.reserve(m_messages_metadata.size());
    for (const auto& metadata : m_messages_metadata) {
        vision_counts.push_back(metadata.get_vision_count());
    }
    return vision_counts;
}

ChatHistory ChatHistoryInternalState::build_normalized_history(const ChatHistory& history) const {
    OPENVINO_ASSERT(m_messages_metadata.size() == history.size(),
                    "Internal state size (", m_messages_metadata.size(),
                    ") doesn't match history size (", history.size(), ")");
    
    ChatHistory normalized_history;

    normalized_history.set_tools(history.get_tools());
    normalized_history.set_extra_context(history.get_extra_context());

    for (size_t i = 0; i < history.size(); ++i) {
        JsonContainer normalized_msg = history[i].copy();

        if (normalized_msg["role"].get_string() == "user") {
            normalized_msg["content"] = m_messages_metadata[i].normalized_content;
        }
        
        normalized_history.push_back(std::move(normalized_msg));
    }
    return normalized_history;
}

std::shared_ptr<ChatHistoryInternalState> ChatHistoryInternalState::get_or_create(
    ChatHistory& history,
    std::shared_ptr<VisionRegistry> vision_registry
) {
    auto state = std::dynamic_pointer_cast<ChatHistoryInternalState>(
        history.get_internal_state()
    );
    
    if (!state) {
        state = std::make_shared<ChatHistoryInternalState>(vision_registry);
        history.set_internal_state(state);
    } else if (vision_registry && !state->get_vision_registry()) {
        state->set_vision_registry(vision_registry);
    }
    return state;
}

const size_t ChatHistoryInternalState::find_matching_history_length(const ChatHistory& history) const {
    size_t matching_history_length = 0;
    
    for (size_t i = 0; i < std::min(m_messages_metadata.size(), history.size()); ++i) {
        if (m_messages_metadata[i].original_message_json != history[i].to_json_string()) {
            break;
        }
        matching_history_length = i + 1;
    }
    
    return std::min(matching_history_length, history.size());
}

void ChatHistoryInternalState::truncate_to(size_t size) {
    if (size >= m_messages_metadata.size()) {
        return;
    }

    size_t new_image_base_index = 0;
    size_t new_video_base_index = 0;
    for (size_t i = 0; i < size; ++i) {
        new_image_base_index += m_messages_metadata[i].provided_image_indices.size();
        new_video_base_index += m_messages_metadata[i].provided_video_indices.size();
    }

    release_refs_from(new_image_base_index, new_video_base_index);

    m_image_index_to_id.resize(new_image_base_index);
    m_video_index_to_id.resize(new_video_base_index);
    
    m_messages_metadata.resize(size);
}

void ChatHistoryInternalState::reset() {
    release_refs_from(0, 0);
    m_image_index_to_id.clear();
    m_video_index_to_id.clear();
    m_messages_metadata.clear();
}

void ChatHistoryInternalState::release_refs_from(size_t image_index, size_t video_index) {
    auto vision_registry = m_vision_registry.lock();
    if (!vision_registry)
        return;
    
    for (size_t i = image_index; i < m_image_index_to_id.size(); ++i) {
        vision_registry->release_ref(m_image_index_to_id[i]);
    }
    for (size_t i = video_index; i < m_video_index_to_id.size(); ++i) {
        vision_registry->release_ref(m_video_index_to_id[i]);
    }
}

} // namespace ov::genai
