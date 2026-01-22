// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/chat_history_state.hpp"

namespace ov::genai {

/**
 * @brief Processes chat history with vision inputs for a single generation call and
 * prepares `ProcessedChatData` for further model inputs processing.
 * 
 * Handles:
 * - Vision registration/encoding via VisionRegistry
 * - Chat history synchronization and incremental updates
 * - Chat history content normalization
 * - Vision sequence resolution
 */
class VLMChatContext  {
public:
    struct ProcessedChatData {
        ChatHistory normalized_history;
        
        std::vector<EncodedImage> encoded_images;
        std::vector<EncodedVideo> encoded_videos;
        std::vector<size_t> image_sequence;
        std::vector<size_t> video_sequence;
        
        std::vector<EncodedImage> new_encoded_images;
        std::vector<EncodedVideo> new_encoded_videos;
        std::vector<size_t> new_image_sequence;
        std::vector<size_t> new_video_sequence;
        
        std::vector<std::pair<size_t, size_t>> vision_counts;

        bool needs_kv_cache_reset = false;
    };

    VLMChatContext(
        const ChatHistory& history,
        const std::shared_ptr<VisionRegistry>& vision_registry,
        InputsEmbedder& embedder
    );

    ProcessedChatData process(
        const std::vector<ov::Tensor>& new_images,
        const std::vector<ov::Tensor>& new_videos = {}
    );

    void rollback();

private:
    const ChatHistory& m_history;
    const std::shared_ptr<VisionRegistry>& m_vision_registry;
    InputsEmbedder& m_inputs_embedder;
    const std::shared_ptr<ChatHistoryInternalState> m_history_state;
    
    size_t m_initial_messages_metadata_count = 0;
    size_t m_initial_base_image_index = 0;
    size_t m_initial_base_video_index = 0;

    void encode_visions_if_needed(
        const std::vector<size_t>& image_indices,
        const std::vector<size_t>& video_indices
    );
                
    void fill_messages_metadata(
        size_t start_index,
        const std::vector<size_t>& new_image_indices,
        const std::vector<size_t>& new_video_indices
    );

    std::string multipart_message_to_string(
        const JsonContainer& message,
        size_t base_image_index,
        size_t base_video_index
    ) const;
};

} // namespace ov::genai
