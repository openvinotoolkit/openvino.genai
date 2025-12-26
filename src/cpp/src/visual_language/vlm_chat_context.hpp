// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/chat_history_state.hpp"

namespace ov::genai {

class VLMChatContext  {
public:
    struct VisionSequenceData {
        std::vector<EncodedImage> encoded_images;
        std::vector<EncodedVideo> encoded_videos;
        std::vector<size_t> image_sequence;
        std::vector<size_t> video_sequence;
        std::unordered_map<VisionID, size_t> image_id_to_index;
        std::unordered_map<VisionID, size_t> video_id_to_index;
    };
    
    struct ProcessedChatData {
        ChatHistory normalized_history;
        
        std::vector<EncodedImage> encoded_images;
        std::vector<EncodedVideo> encoded_videos;

        // Full index sequences (indices into encoded_* vectors, in order of appearance)
        std::vector<size_t> image_sequence;
        std::vector<size_t> video_sequence;

        std::vector<size_t> new_image_indices;
        std::vector<size_t> new_video_indices;

        std::vector<std::pair<size_t, size_t>> vision_counts;
        
        bool state_changed = false;

        std::vector<EncodedImage> get_new_images() const {
            std::vector<EncodedImage> result;
            result.reserve(new_image_indices.size());
            for (size_t idx : new_image_indices) {
                result.push_back(encoded_images[idx]);
            }
            return result;
        }
        
        std::vector<EncodedVideo> get_new_videos() const {
            std::vector<EncodedVideo> result;
            result.reserve(new_video_indices.size());
            for (size_t idx : new_video_indices) {
                result.push_back(encoded_videos[idx]);
            }
            return result;
        }
    };

    VLMChatContext(
        ChatHistory& history,
        std::shared_ptr<VisionRegistry> vision_registry,
        InputsEmbedder& embedder // TODO Should be shared_ptr?
    );

    ProcessedChatData process(
        const std::vector<ov::Tensor>& new_images,
        const std::vector<ov::Tensor>& new_videos
    );

    void finalize();

    void rollback(size_t history_size);

    // TODO Check if needed
    size_t message_count() const;

    std::shared_ptr<ChatHistoryInternalStateClass> history_state() const { return m_history_state; }

    bool needs_kv_cache_reset() const { return m_needs_kv_reset; }

private:
    ChatHistory& m_history;
    std::shared_ptr<VisionRegistry> m_vision_registry;
    InputsEmbedder& m_embedder;
    std::shared_ptr<ChatHistoryInternalStateClass> m_history_state;
    bool m_needs_kv_reset = false;
    // TODO Check if needed
    size_t m_checkpoint_message_count = 0;

    size_t sync_with_history();

    void process_new_messages(
        size_t start_index,
        const std::vector<VisionID>& new_image_ids,
        const std::vector<VisionID>& new_video_ids
    );

    void ensure_visions_encoded(const std::vector<VisionID>& image_ids,
                                const std::vector<VisionID>& video_ids);

    // TODO Consider using a struct or type for reuse
    VisionSequenceData resolve_vision_sequences();
};

} // namespace ov::genai
