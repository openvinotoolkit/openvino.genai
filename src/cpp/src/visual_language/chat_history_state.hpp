// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/vision_encoder.hpp"

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
    
    void reset() {
        processed_history_size = 0;
        encoded_images.clear();
        encoded_videos.clear();
        image_id = 0;
        video_id = 0;
        image_sequence.clear();
        video_sequence.clear();
        vision_count.clear();
    }
    
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

} // namespace ov::genai
