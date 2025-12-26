// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vlm_chat_context.hpp"

namespace ov::genai {

VLMChatContext::VLMChatContext(
    ChatHistory& history,
    std::shared_ptr<VisionRegistry> vision_registry,
    InputsEmbedder& embedder
) : m_history(history),
    m_vision_registry(vision_registry),
    m_embedder(embedder)
{
    m_history_state = ChatHistoryInternalStateClass::get_or_create(history, vision_registry);
    m_checkpoint_message_count = m_history_state->messages_metadata_size(); // TODO Consider removing field
}

VLMChatContext::ProcessedChatData VLMChatContext::process(
    const std::vector<ov::Tensor>& new_images,
    const std::vector<ov::Tensor>& new_videos
) {
    OPENVINO_ASSERT(!m_history.empty(), "Chat history cannot be empty");
    OPENVINO_ASSERT(m_history.last()["role"].get_string() == "user",
                    "Last message must be from user");
    
    ProcessedChatData result;
    
    // Step 1: Sync state with history
    size_t valid_prefix = sync_with_history();
    // TODO Check if state_changed in result needed, consider replacing with local var
    result.state_changed = (valid_prefix < m_checkpoint_message_count);
    
    // Step 2: Register new visions
    std::vector<VisionID> new_image_ids = m_vision_registry->register_images(new_images);
    std::vector<VisionID> new_video_ids = m_vision_registry->register_videos(new_videos);
    
    // Step 3: Encode new visions
    ensure_visions_encoded(new_image_ids, new_video_ids);
    
    // Step 4: Process new messages
    process_new_messages(valid_prefix, new_image_ids, new_video_ids);
    
    // Step 5: Build normalized history
    result.normalized_history = m_history_state->build_normalized_history(m_history);
    
    // Step 6: Resolve sequences
    // auto [encoded_images, encoded_videos, image_sequence, video_sequence] = resolve_vision_sequences();
    auto vision_sequence_data = resolve_vision_sequences();
    result.encoded_images = std::move(vision_sequence_data.encoded_images);
    result.encoded_videos = std::move(vision_sequence_data.encoded_videos);
    result.image_sequence = std::move(vision_sequence_data.image_sequence);
    result.video_sequence = std::move(vision_sequence_data.video_sequence);

    // Step 7: Calculate new_*_indices using the mappings
    result.new_image_indices.reserve(new_image_ids.size());
    for (const auto& id : new_image_ids) {
        auto it = vision_sequence_data.image_id_to_index.find(id);
        OPENVINO_ASSERT(it != vision_sequence_data.image_id_to_index.end(),
                        "New image ID not found in mapping");
        result.new_image_indices.push_back(it->second);
    }
    
    result.new_video_indices.reserve(new_video_ids.size());
    for (const auto& id : new_video_ids) {
        auto it = vision_sequence_data.video_id_to_index.find(id);
        OPENVINO_ASSERT(it != vision_sequence_data.video_id_to_index.end(),
                        "New video ID not found in mapping");
        result.new_video_indices.push_back(it->second);
    }
    
    // Step 7: Build vision counts
    result.vision_counts = m_history_state->build_vision_counts();
    
    // Step 8: Determine KV cache reset need
    m_needs_kv_reset = result.state_changed ||
                       (m_history_state->get_kv_cache_valid_messages() == 0) ||
                       (m_history.size() < m_history_state->get_kv_cache_valid_messages());
    
    return result;
}

void VLMChatContext::finalize() {
    m_history_state->set_kv_cache_valid_messages(m_history.size());
}

void VLMChatContext::rollback(size_t history_size) {
    m_history_state->truncate_to(history_size);
}

size_t VLMChatContext::message_count() const {
    return m_history_state->messages_metadata_size();
}

size_t VLMChatContext::sync_with_history() {
    size_t valid_prefix = m_history_state->find_valid_prefix(m_history);
    
    if (valid_prefix < m_history_state->messages_metadata_size()) {
        m_history_state->truncate_to(valid_prefix);
    }
    
    return valid_prefix;
}

// TODO consider renaming
void VLMChatContext::ensure_visions_encoded(
    const std::vector<VisionID>& image_ids,
    const std::vector<VisionID>& video_ids
) {
    // Encode images/videos that don't have encoded data yet
    for (const auto& id : image_ids) {
        if (!m_vision_registry->has_encoded_image(id)) {
            const ov::Tensor& original = m_vision_registry->get_original(id);
            auto encoded = m_embedder.encode_images({original});
            m_vision_registry->set_encoded_image(id, std::move(encoded[0]));
        }
    }
    for (const auto& id : video_ids) {
        if (!m_vision_registry->has_encoded_video(id)) {
            const ov::Tensor& original = m_vision_registry->get_original(id);
            auto encoded = m_embedder.encode_videos({original});
            m_vision_registry->set_encoded_video(id, std::move(encoded[0]));
        }
    }
}

// Process new messages metadata
void VLMChatContext::process_new_messages(
    size_t start_index,
    const std::vector<VisionID>& new_image_ids,
    const std::vector<VisionID>& new_video_ids
) {
    const size_t last_index = m_history.size() - 1;
    
    // Resize messages vector
    m_history_state->get_messages_metadata().resize(m_history.size());
    
    for (size_t i = start_index; i < m_history.size(); ++i) {
        const auto& message = m_history[i];
        auto& metadata = m_history_state->get_message_metadata(i);
        
        // Store original
        metadata.original_message_json = message.to_json_string();

        const auto original_content = message["content"].get_string();
        
        std::string role = message["role"].get_string();

        if (role != "user") {
            // Assistant/system - no normalization needed
            metadata.normalized_content = original_content;
            // TODO Check if other fields should be cleared - check where metadata is created
            metadata.image_sequence.clear();
            metadata.video_sequence.clear();
            continue;
        }
        
        // Associate new visions with last user message
        if (i == last_index) {
            metadata.provided_image_ids = new_image_ids;
            metadata.provided_video_ids = new_video_ids;
        }
        
        // Get encoded visions for normalization
        std::vector<EncodedImage> msg_images;
        std::vector<EncodedVideo> msg_videos;
        
        for (const auto& id : metadata.provided_image_ids) {
            msg_images.push_back(m_vision_registry->get_encoded_image(id));
        }
        for (const auto& id : metadata.provided_video_ids) {
            msg_videos.push_back(m_vision_registry->get_encoded_video(id));
        }
        
        // Get text content from message
        // std::string last_user_message_text = m_embedder.get_last_user_message_text(m_history);
        // std::cout << "Last user message text: " << last_user_message_text << std::endl;
        // std::cout << "original_content: " << original_content << std::endl;
        
        // std::cout << "msg_images size: " << msg_images.size() << std::endl;

        // Normalize prompt
        auto normalized = m_embedder.normalize_prompt(
            original_content, 0, 0, msg_images, msg_videos
        );
        
        metadata.normalized_content = normalized.unified_prompt;
        
        // Build VisionID sequences from index sequences
        // normalized.images_sequence contains indices into msg_images
        metadata.image_sequence.clear();
        for (size_t idx : normalized.images_sequence) {
            if (idx < metadata.provided_image_ids.size()) {
                metadata.image_sequence.push_back(metadata.provided_image_ids[idx]);
            }
        }
        
        metadata.video_sequence.clear();
        for (size_t idx : normalized.videos_sequence) {
            if (idx < metadata.provided_video_ids.size()) {
                metadata.video_sequence.push_back(metadata.provided_video_ids[idx]);
            }
        }
    }
}

VLMChatContext::VisionSequenceData VLMChatContext::resolve_vision_sequences() {
    VisionSequenceData result;

    // Collect all VisionIDs
    std::vector<VisionID> all_image_ids = m_history_state->build_full_image_sequence();
    std::vector<VisionID> all_video_ids = m_history_state->build_full_video_sequence();
    
    // Deduplicate and build mappings
    // std::unordered_map<VisionID, size_t> image_id_to_index;
    // std::unordered_map<VisionID, size_t> video_id_to_index;
    
    // std::vector<EncodedImage> encoded_images;
    // std::vector<EncodedVideo> encoded_videos;
    
    for (const auto& id : all_image_ids) {
        if (result.image_id_to_index.find(id) == result.image_id_to_index.end()) {
            size_t index = result.encoded_images.size();
            result.image_id_to_index[id] = index;
            result.encoded_images.push_back(m_vision_registry->get_encoded_image(id));
        }
    }
    
    for (const auto& id : all_video_ids) {
        if (result.video_id_to_index.find(id) == result.video_id_to_index.end()) {
            size_t index = result.encoded_videos.size();
            result.video_id_to_index[id] = index;
            result.encoded_videos.push_back(m_vision_registry->get_encoded_video(id));
        }
    }
    
    // Build index sequences
    // std::vector<size_t> image_sequence;
    result.image_sequence.reserve(all_image_ids.size());
    for (const auto& id : all_image_ids) {
        result.image_sequence.push_back(result.image_id_to_index[id]);
    }
    
    // std::vector<size_t> video_sequence;
    result.video_sequence.reserve(all_video_ids.size());
    for (const auto& id : all_video_ids) {
        result.video_sequence.push_back(result.video_id_to_index[id]);
    }
    
    // return {
    //     std::move(encoded_images),
    //     std::move(encoded_videos),
    //     std::move(image_sequence),
    //     std::move(video_sequence)
    // };
    return result;
}

} // namespace ov::genai
