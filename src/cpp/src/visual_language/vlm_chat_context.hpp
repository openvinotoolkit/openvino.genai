// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/chat_history_state.hpp"

namespace ov::genai {

class VLMChatContext  {
public:
    struct ProcessedChatData {
        ChatHistory normalized_history;
        
        std::vector<EncodedImage> encoded_images;
        std::vector<EncodedVideo> encoded_videos;
        std::vector<size_t> image_sequence;
        std::vector<size_t> video_sequence;
        
        // TODO Consider using indices to encoded_images/video instead of copying
        std::vector<EncodedImage> new_encoded_images;
        std::vector<EncodedVideo> new_encoded_videos;
        std::vector<size_t> new_image_sequence;
        std::vector<size_t> new_video_sequence;
        
        std::vector<std::pair<size_t, size_t>> vision_counts;
    };

    VLMChatContext(
        ChatHistory& history,
        std::shared_ptr<VisionRegistry> vision_registry,
        InputsEmbedder& embedder
    );

    ProcessedChatData process(
        const std::vector<ov::Tensor>& new_images,
        const std::vector<ov::Tensor>& new_videos = {}
    );

    void finalize();

    void rollback();

    bool needs_kv_cache_reset() const { return m_needs_kv_reset; }

private:
    ChatHistory& m_history;
    std::shared_ptr<VisionRegistry> m_vision_registry;
    InputsEmbedder& m_inputs_embedder;
    std::shared_ptr<ChatHistoryInternalState> m_history_state;
    
    // TODO Check if needed
    size_t m_checkpoint_message_count = 0;
    // TODO Consider renaming
    bool m_needs_kv_reset = false;

    void encode_visions_if_needed(const std::vector<size_t>& image_indices,
                                  const std::vector<size_t>& video_indices);
                
    void fill_messages_metadata(
        size_t start_index,
        const std::vector<size_t>& new_image_indices,
        const std::vector<size_t>& new_video_indices
    );
};

} // namespace ov::genai
