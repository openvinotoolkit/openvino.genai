// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vlm_chat_context.hpp"

namespace ov::genai {

VLMChatContext::VLMChatContext(
    ChatHistory& history,
    std::shared_ptr<VisionRegistry> vision_registry,
    InputsEmbedder& embedder
) : m_history(history),
    m_vision_registry(vision_registry),
    m_inputs_embedder(embedder)
{
    m_history_state = ChatHistoryInternalState::get_or_create(history, vision_registry);
    m_initial_messages_metadata_count = m_history_state->get_messages_metadata().size();
}

VLMChatContext::ProcessedChatData VLMChatContext::process(
    const std::vector<ov::Tensor>& new_images,
    const std::vector<ov::Tensor>& new_videos
) {
    OPENVINO_ASSERT(!m_history.empty(), "Chat history cannot be empty");
    OPENVINO_ASSERT(m_history.last()["role"].get_string() == "user",
                    "Last message must be from user");
    
    ProcessedChatData result;
    
    const size_t matching_history_length = m_history_state->find_matching_history_length(m_history);
    const bool history_modified = matching_history_length < m_initial_messages_metadata_count;

    if (history_modified) {
        m_history_state->truncate_to(matching_history_length);
    }
    
    std::vector<size_t> new_image_indices = m_history_state->register_images(new_images);
    std::vector<size_t> new_video_indices = m_history_state->register_videos(new_videos);
    
    encode_visions_if_needed(new_image_indices, new_video_indices);
    
    fill_messages_metadata(matching_history_length, new_image_indices, new_video_indices);
    
    result.normalized_history = m_history_state->build_normalized_history(m_history);
    
    auto resolved_visions = m_history_state->resolve_visions_with_sequence();
    result.encoded_images = std::move(resolved_visions.encoded_images);
    result.encoded_videos = std::move(resolved_visions.encoded_videos);
    result.image_sequence = std::move(resolved_visions.image_sequence);
    result.video_sequence = std::move(resolved_visions.video_sequence);

    const auto& last_message_metadata = m_history_state->get_messages_metadata().back();

    auto resolved_new_visions = m_history_state->resolve_visions_with_sequence(
        last_message_metadata.image_sequence,
        last_message_metadata.video_sequence
    );

    result.new_encoded_images = std::move(resolved_new_visions.encoded_images);
    result.new_encoded_videos = std::move(resolved_new_visions.encoded_videos);
    result.new_image_sequence = std::move(resolved_new_visions.image_sequence);
    result.new_video_sequence = std::move(resolved_new_visions.video_sequence);
    
    result.vision_counts = m_history_state->build_vision_counts();
    
    result.needs_kv_cache_reset = m_initial_messages_metadata_count == 0 || history_modified;
    
    return result;
}

void VLMChatContext::rollback() {
     m_history_state->truncate_to(m_initial_messages_metadata_count);
}

void VLMChatContext::encode_visions_if_needed(
    const std::vector<size_t>& image_indices,
    const std::vector<size_t>& video_indices
) {
    for (size_t idx : image_indices) {
        VisionID id = m_history_state->get_image_vision_id(idx);
        if (!m_vision_registry->has_encoded_image(id)) {
            const ov::Tensor& original = m_vision_registry->get_original(id);
            auto encoded = m_inputs_embedder.encode_images({original});
            m_vision_registry->set_encoded_image(id, std::move(encoded[0]));
        }
    }

    for (size_t idx : video_indices) {
        VisionID id = m_history_state->get_video_vision_id(idx);
        if (!m_vision_registry->has_encoded_video(id)) {
            const ov::Tensor& original = m_vision_registry->get_original(id);
            auto encoded = m_inputs_embedder.encode_videos({original});
            m_vision_registry->set_encoded_video(id, std::move(encoded[0]));
        }
    }
}

void VLMChatContext::fill_messages_metadata(
    size_t start_index,
    const std::vector<size_t>& new_image_indices,
    const std::vector<size_t>& new_video_indices
) {
    size_t base_image_index = 0;
    size_t base_video_index = 0;
    for (size_t i = 0; i < start_index; ++i) {
        base_image_index += m_history_state->get_messages_metadata().at(i).provided_image_indices.size();
        base_video_index += m_history_state->get_messages_metadata().at(i).provided_video_indices.size();
    }

    for (size_t i = start_index; i < m_history.size(); ++i) {
        const auto& message = m_history[i];
        
        MessageMetadata metadata;
        metadata.original_message_json = message.to_json_string();
        
        std::string role = message["role"].get_string();
        std::string original_content = message["content"].get_string();
        
        if (role != "user") {
            // Non user messages do not require normalization and vision metadata
            metadata.normalized_content = original_content;
            m_history_state->add_message_metadata(std::move(metadata));
            continue;
        }
        
        if (i == m_history.size() - 1) {
            metadata.provided_image_indices = new_image_indices;
            metadata.provided_video_indices = new_video_indices;
        }
        
        std::vector<EncodedImage> encoded_images = m_history_state->get_encoded_images(metadata.provided_image_indices);
        std::vector<EncodedVideo> encoded_videos = m_history_state->get_encoded_videos(metadata.provided_video_indices);
        
        auto normalized = m_inputs_embedder.normalize_prompt(
            original_content, base_image_index, base_video_index, encoded_images, encoded_videos
        );
        
        metadata.normalized_content = normalized.unified_prompt;
        
        metadata.image_sequence = normalized.images_sequence;
        metadata.video_sequence = normalized.videos_sequence;

        // image_sequence can be empty after prompt normalization (phi3_vision and phi4mm) - use provided indices (input order)
        if (metadata.image_sequence.empty() && !metadata.provided_image_indices.empty()) {
            metadata.image_sequence = metadata.provided_image_indices;
        }
        if (metadata.video_sequence.empty() && !metadata.provided_video_indices.empty()) {
            metadata.video_sequence = metadata.provided_video_indices;
        }
        
        base_image_index += metadata.provided_image_indices.size();
        base_video_index += metadata.provided_video_indices.size();
        
        m_history_state->add_message_metadata(std::move(metadata));
    }
}

} // namespace ov::genai
