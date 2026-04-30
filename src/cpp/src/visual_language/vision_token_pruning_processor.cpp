// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vision_token_pruning_processor.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>

#include "logger.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/core.hpp"
#include "utils.hpp"
#include "visual_language/vision_encoder.hpp"

namespace ov::genai {

VisionTokenPruningProcessor::VisionTokenPruningProcessor(const std::string& device) : m_config() {
    m_config.device = device;
}

ov::Tensor VisionTokenPruningProcessor::process(const std::vector<ov::Tensor>& visual_features,
                                                const ov::Tensor& text_features) {
    if (!m_pruner) {
        return ov::Tensor();
    }

    // Delegate to CDPruner for processing
    return m_pruner->apply_pruning(visual_features, text_features);
}

void VisionTokenPruningProcessor::set_config(const cdpruner::Config& config) {
    std::string device = std::move(m_config.device);
    m_config = config;
    m_config.device = std::move(device);

    if (!m_pruner) {
        m_pruner = std::make_unique<cdpruner::CDPruner>(m_config);
    } else {
        // Update existing pruner configuration
        if (!m_pruner->update_config(m_config)) {
            m_pruner = std::make_unique<cdpruner::CDPruner>(m_config);
        }
    }
}

cdpruner::Config VisionTokenPruningProcessor::get_config() const {
    // If pruner exists, return its configuration as the single source of truth
    // Otherwise, return the stored configuration that will be used when pruner is created
    if (m_pruner) {
        return m_pruner->get_config();
    }
    return m_config;
}

std::optional<cdpruner::PruningStatistics> VisionTokenPruningProcessor::get_last_statistics() const {
    if (!m_pruner) {
        return std::nullopt;
    }

    try {
        return m_pruner->get_last_pruning_statistics();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get pruning statistics: " << e.what() << std::endl;
        return std::nullopt;
    }
}

std::vector<std::vector<size_t>> VisionTokenPruningProcessor::get_last_selected_tokens() const {
    if (!m_pruner) {
        return {};
    }

    try {
        return m_pruner->get_last_selected_tokens();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get selected token indices: " << e.what() << std::endl;
        return {};
    }
}

std::string VisionTokenPruningProcessor::get_last_pruned_prompt(const std::string& original_prompt,
                                                                const std::string& vision_start_token,
                                                                const std::string& vision_end_token,
                                                                const std::string& image_pad_token,
                                                                const std::string& video_pad_token) const {
    if (m_last_keep_flags.empty()) {
        return original_prompt;
    }

    std::string result;
    result.reserve(original_prompt.size());

    size_t pos = 0;
    bool inside_vision_region = false;
    size_t region_idx = 0;
    size_t pad_idx = 0;
    const size_t region_count = m_last_keep_flags.size();
    size_t total_pads_processed = 0;
    size_t total_pads_kept = 0;

    while (pos < original_prompt.size()) {
        // Find the next nearest special token position
        size_t next_vision_start = original_prompt.find(vision_start_token, pos);
        size_t next_vision_end = original_prompt.find(vision_end_token, pos);
        size_t next_image_pad = inside_vision_region ? original_prompt.find(image_pad_token, pos) : std::string::npos;
        size_t next_video_pad = inside_vision_region ? original_prompt.find(video_pad_token, pos) : std::string::npos;

        // Determine which token comes first
        size_t next_token_pos = std::min({next_vision_start, next_vision_end, next_image_pad, next_video_pad});

        // If no special tokens found, copy remaining text and exit
        if (next_token_pos == std::string::npos) {
            result.append(original_prompt, pos, std::string::npos);
            break;
        }

        // Copy regular text before the next special token
        if (next_token_pos > pos) {
            result.append(original_prompt, pos, next_token_pos - pos);
            pos = next_token_pos;
        }

        // Process the special token found at current position
        if (next_token_pos == next_vision_start) {
            result.append(vision_start_token);
            pos += vision_start_token.size();
            inside_vision_region = true;
            pad_idx = 0;
        } else if (next_token_pos == next_vision_end) {
            result.append(vision_end_token);
            pos += vision_end_token.size();
            inside_vision_region = false;
            region_idx++;
        } else if (next_token_pos == next_image_pad || next_token_pos == next_video_pad) {
            const std::string& pad_token = (next_token_pos == next_image_pad) ? image_pad_token : video_pad_token;

            if (region_idx < region_count && pad_idx < m_last_keep_flags[region_idx].size()) {
                total_pads_processed++;
                if (m_last_keep_flags[region_idx][pad_idx]) {
                    result.append(pad_token);
                    total_pads_kept++;
                }
                pad_idx++;
            }
            pos += pad_token.size();
        }
    }

    GENAI_DEBUG("Prompt update (len=%zu, regions=%zu): processed %zu pad tokens, kept %zu",
                original_prompt.size(),
                region_count,
                total_pads_processed,
                total_pads_kept);
    return result;
}

// Extract text features by averaging instruction token embeddings
ov::Tensor VisionTokenPruningProcessor::extract_text_features(const ov::Tensor& text_embeds,
                                                              const ov::Tensor& input_ids,
                                                              int64_t image_pad_token_id,
                                                              int64_t vision_start_token_id,
                                                              int64_t vision_end_token_id,
                                                              int64_t video_pad_token_id) const {
    // Find instruction token positions (skip vision regions and pad tokens)
    std::vector<size_t> instruction_indices;
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    size_t seq_len = input_ids.get_shape()[1];  // [batch_size, seq_len]
    bool inside_vision_region = false;

    for (size_t i = 0; i < seq_len; ++i) {
        int64_t current_token = input_ids_data[i];

        if (current_token == vision_start_token_id) {
            inside_vision_region = true;
            continue;
        }
        if (current_token == vision_end_token_id) {
            inside_vision_region = false;
            continue;
        }
        // Skip vision region tokens and vision pad tokens
        if (inside_vision_region || current_token == image_pad_token_id ||
            (video_pad_token_id >= 0 && current_token == video_pad_token_id)) {
            continue;
        }
        instruction_indices.push_back(i);
    }

    // Handle empty instruction case
    if (instruction_indices.empty()) {
        ov::Tensor zero_embedding(ov::element::f32, {1, text_embeds.get_shape().back()});
        std::memset(zero_embedding.data<float>(), 0, zero_embedding.get_byte_size());
        return zero_embedding;
    }

    // Extract and average instruction token embeddings from text_embeds
    size_t hidden_size = text_embeds.get_shape().back();
    ov::Tensor avg_embedding(ov::element::f32, {1, hidden_size});
    float* avg_data = avg_embedding.data<float>();
    const float* text_data = text_embeds.data<const float>();

    std::memset(avg_data, 0, avg_embedding.get_byte_size());

    // Sum embeddings at instruction positions
    for (size_t idx : instruction_indices) {
        const float* token_embed = text_data + idx * hidden_size;
        for (size_t dim = 0; dim < hidden_size; ++dim) {
            avg_data[dim] += token_embed[dim];
        }
    }

    // Calculate average
    float num_tokens = static_cast<float>(instruction_indices.size());
    for (size_t dim = 0; dim < hidden_size; ++dim) {
        avg_data[dim] /= num_tokens;
    }

    return avg_embedding;
}

std::vector<ov::Tensor> VisionTokenPruningProcessor::convert_visual_features(
    const ov::Tensor& vision_embeds,
    size_t chunk_count,
    const std::vector<size_t>& tokens_per_image) const {
    // Convert from [num_patches, embedding_dim] to chunk_count * [1, num_patches_i, embedding_dim]
    ov::Shape original_shape = vision_embeds.get_shape();
    size_t total_tokens = original_shape[0];
    size_t embedding_dim = original_shape[1];
    const float* src_data = vision_embeds.data<const float>();

    std::vector<ov::Tensor> visual_features;

    // When chunk_count = 1 (frame chunking disabled), treat all tokens as a single batch
    if (chunk_count == 1) {
        ov::Shape batch_shape = {1, total_tokens, embedding_dim};
        ov::Tensor features(vision_embeds.get_element_type(), batch_shape);
        float* dst_data = features.data<float>();
        std::memcpy(dst_data, src_data, total_tokens * embedding_dim * sizeof(float));
        visual_features.push_back(features);
        return visual_features;
    }

    // Frame chunking enabled: split by individual images
    OPENVINO_ASSERT(tokens_per_image.size() >= chunk_count,
                    "Insufficient tokens_per_image entries. Got " + std::to_string(tokens_per_image.size()) +
                        ", need " + std::to_string(chunk_count));

    size_t current_offset = 0;

    for (size_t i = 0; i < chunk_count; i++) {
        size_t image_tokens = tokens_per_image[i];

        // Boundary check
        OPENVINO_ASSERT(current_offset + image_tokens <= total_tokens,
                        "Image boundary exceeds embeddings size. Image " + std::to_string(i) +
                            ": offset=" + std::to_string(current_offset) + ", tokens=" + std::to_string(image_tokens) +
                            ", total=" + std::to_string(total_tokens));

        // Create tensor for current image [1, tokens_i, D]
        ov::Shape image_shape = {1, image_tokens, embedding_dim};
        ov::Tensor features(vision_embeds.get_element_type(), image_shape);
        float* dst_data = features.data<float>();

        // Copy data
        size_t elements_to_copy = image_tokens * embedding_dim;
        std::memcpy(dst_data, src_data + current_offset * embedding_dim, elements_to_copy * sizeof(float));

        visual_features.push_back(features);
        current_offset += image_tokens;
    }

    // Verify all tokens processed
    OPENVINO_ASSERT(current_offset == total_tokens,
                    "Not all tokens were processed. Expected: " + std::to_string(total_tokens) +
                        ", processed: " + std::to_string(current_offset));

    return visual_features;
}

ov::Tensor VisionTokenPruningProcessor::generate_pruned_input_ids(
    const ov::Tensor& input_ids,
    const std::vector<std::vector<bool>>& keep_flags_per_region,
    int64_t image_pad_token_id,
    int64_t vision_start_token_id,
    int64_t vision_end_token_id,
    int64_t video_pad_token_id) const {
    size_t original_seq_len = input_ids.get_shape().at(1);

    // Calculate total tokens to remove
    size_t tokens_to_remove = 0;
    for (const auto& mask : keep_flags_per_region) {
        tokens_to_remove += static_cast<size_t>(std::count(mask.begin(), mask.end(), false));
    }

    size_t new_sequence_length = original_seq_len - tokens_to_remove;
    ov::Tensor pruned_input_ids(ov::element::i64, {1, new_sequence_length});

    const int64_t* input_data = input_ids.data<const int64_t>();
    int64_t* pruned_data = pruned_input_ids.data<int64_t>();

    size_t write_idx = 0;
    bool inside_vision_region = false;
    size_t region_idx = 0;
    size_t pad_index = 0;
    size_t region_count = keep_flags_per_region.size();

    for (size_t seq_idx = 0; seq_idx < original_seq_len; ++seq_idx) {
        int64_t token_id = input_data[seq_idx];

        if (token_id == vision_start_token_id) {
            OPENVINO_ASSERT(region_idx < region_count,
                            "Encountered more vision regions than metadata entries while pruning input ids");
            inside_vision_region = true;
            pad_index = 0;
        }

        bool is_vision_pad =
            (token_id == image_pad_token_id) || (video_pad_token_id >= 0 && token_id == video_pad_token_id);
        if (inside_vision_region && is_vision_pad) {
            OPENVINO_ASSERT(region_idx < region_count,
                            "Vision region index exceeds metadata size while pruning input ids");
            const auto& keep_mask = keep_flags_per_region.at(region_idx);
            OPENVINO_ASSERT(pad_index < keep_mask.size(),
                            "Visual token index exceeds region token count while pruning input ids");
            if (keep_mask[pad_index]) {
                OPENVINO_ASSERT(write_idx < new_sequence_length,
                                "Pruned input ids index exceeds expected sequence length");
                pruned_data[write_idx++] = token_id;
            }
            ++pad_index;
            continue;
        }

        OPENVINO_ASSERT(write_idx < new_sequence_length, "Pruned input ids index exceeds expected sequence length");
        pruned_data[write_idx++] = token_id;

        if (inside_vision_region && token_id == vision_end_token_id) {
            const auto& keep_mask = keep_flags_per_region.at(region_idx);
            OPENVINO_ASSERT(pad_index == keep_mask.size(),
                            "Mismatch between consumed visual tokens and region metadata while pruning input ids");
            inside_vision_region = false;
            ++region_idx;
        }
    }

    OPENVINO_ASSERT(!inside_vision_region, "Unexpected end of sequence inside a vision region while pruning input ids");
    OPENVINO_ASSERT(region_idx == region_count, "Not all vision regions processed while generating pruned input ids");
    OPENVINO_ASSERT(write_idx == new_sequence_length, "Pruned input ids length mismatch after visual token pruning");

    return pruned_input_ids;
}

ov::Tensor VisionTokenPruningProcessor::generate_pruned_text_embeds(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    int64_t image_pad_token_id,
    int64_t vision_start_token_id,
    int64_t vision_end_token_id,
    const std::vector<std::vector<bool>>& keep_flags_per_region,
    int64_t video_pad_token_id) const {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t original_seq_length = text_embeds_shape.at(1);
    size_t hidden_size = text_embeds_shape.at(2);

    // Calculate new sequence length after removing filtered tokens
    size_t total_original_visual_tokens = 0;
    size_t total_kept_visual_tokens = 0;
    for (const auto& mask : keep_flags_per_region) {
        total_original_visual_tokens += mask.size();
        total_kept_visual_tokens += static_cast<size_t>(std::count(mask.begin(), mask.end(), true));
    }
    size_t tokens_removed = total_original_visual_tokens - total_kept_visual_tokens;
    size_t new_seq_length = original_seq_length - tokens_removed;

    ov::Tensor pruned_text_embeds(text_embeds.get_element_type(), {batch_size, new_seq_length, hidden_size});

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const float* text_embeds_data = text_embeds.data<const float>();
    float* pruned_data = pruned_text_embeds.data<float>();

    const size_t region_count = keep_flags_per_region.size();

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t write_idx = 0;
        size_t region_idx = 0;
        size_t pad_index = 0;
        bool inside_vision_region = false;

        for (size_t seq_idx = 0; seq_idx < original_seq_length; ++seq_idx) {
            size_t input_flat_idx = batch_idx * original_seq_length + seq_idx;
            int64_t token_id = input_ids_data[input_flat_idx];

            if (token_id == vision_start_token_id) {
                inside_vision_region = true;
                pad_index = 0;
            }

            // Skip filtered vision pad tokens (both image_pad and video_pad)
            bool is_vision_pad =
                (token_id == image_pad_token_id) || (video_pad_token_id >= 0 && token_id == video_pad_token_id);
            if (inside_vision_region && is_vision_pad) {
                const auto& keep_mask = keep_flags_per_region.at(region_idx);
                if (!keep_mask[pad_index]) {
                    ++pad_index;
                    continue;  // Skip this token
                }
                ++pad_index;
            }

            // Copy the embedding
            size_t output_flat_idx = batch_idx * new_seq_length + write_idx;
            std::copy_n(text_embeds_data + input_flat_idx * hidden_size,
                        hidden_size,
                        pruned_data + output_flat_idx * hidden_size);
            ++write_idx;

            if (inside_vision_region && token_id == vision_end_token_id) {
                inside_vision_region = false;
                ++region_idx;
            }
        }

        OPENVINO_ASSERT(write_idx == new_seq_length,
                        "Pruned text embeddings length mismatch. Expected: " + std::to_string(new_seq_length) +
                            ", Got: " + std::to_string(write_idx));
    }

    return pruned_text_embeds;
}

void VisionTokenPruningProcessor::adjust_position_ids(ov::Tensor& position_ids,
                                                      const ov::Tensor& input_ids,
                                                      const std::vector<std::array<size_t, 3>>& images_grid_thw,
                                                      const std::vector<size_t>& images_sequence,
                                                      int64_t image_pad_token_id,
                                                      int64_t vision_start_token_id,
                                                      size_t spatial_merge_size,
                                                      std::vector<std::vector<bool>>& keep_flags_per_region_out,
                                                      int64_t video_pad_token_id) const {
    auto kept_indices_per_image = get_last_selected_tokens();
    OPENVINO_ASSERT(!images_sequence.empty(), "Vision region sequence must not be empty when pruning visual tokens");
    OPENVINO_ASSERT(!kept_indices_per_image.empty(), "Kept token indices are missing after pruning");

    // Reorder images according to sequence
    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;
    reordered_images_grid_thw.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        OPENVINO_ASSERT(new_image_id < images_grid_thw.size(), "Vision region sequence index is out of range");
        reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id));
    }

    // Detect position encoding type from shape
    const ov::Shape& pos_shape = position_ids.get_shape();
    bool is_3d_encoding = (pos_shape.size() == 3 && pos_shape[0] == 3);

    if (is_3d_encoding) {
        // 3D RoPE position encoding (Qwen2VL style)
        position_ids = update_position_ids_3d(position_ids,
                                              input_ids,
                                              vision_start_token_id,
                                              image_pad_token_id,
                                              reordered_images_grid_thw,
                                              kept_indices_per_image,
                                              spatial_merge_size,
                                              keep_flags_per_region_out,
                                              video_pad_token_id);
    } else {
        // 1D position encoding (LLaVA, MiniCPM, etc.)
        position_ids = update_position_ids_1d(position_ids,
                                              input_ids,
                                              vision_start_token_id,
                                              image_pad_token_id,
                                              reordered_images_grid_thw,
                                              kept_indices_per_image,
                                              keep_flags_per_region_out,
                                              video_pad_token_id);
    }
}

// 3D position IDs update for Qwen2VL-style models
ov::Tensor VisionTokenPruningProcessor::update_position_ids_3d(
    const ov::Tensor& original_position_ids,
    const ov::Tensor& input_ids,
    int64_t vision_start_token_id,
    int64_t image_pad_token_id,
    const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw,
    const std::vector<std::vector<size_t>>& kept_indices_per_image,
    size_t spatial_merge_size,
    std::vector<std::vector<bool>>& keep_flags_out,
    int64_t video_pad_token_id) const {
    const ov::Shape& pos_shape = original_position_ids.get_shape();
    OPENVINO_ASSERT(pos_shape.size() == 3 && pos_shape[0] == 3, "Position ids must be [3, batch, seq_len]");

    const size_t batch_size = pos_shape[1];
    const size_t seq_len = pos_shape[2];
    const size_t region_count = reordered_images_grid_thw.size();

    // Build region metadata
    struct RegionInfo {
        size_t tokens, grid_width, spatial_area, offset;
    };

    std::vector<RegionInfo> regions;
    regions.reserve(region_count);
    size_t cumulative_offset = 0;

    for (size_t idx = 0; idx < reordered_images_grid_thw.size(); ++idx) {
        const auto& [grid_t, grid_h, grid_w] = reordered_images_grid_thw[idx];
        OPENVINO_ASSERT(grid_h % spatial_merge_size == 0 && grid_w % spatial_merge_size == 0,
                        "Grid dimensions must be divisible by spatial merge size");

        size_t llm_grid_h = grid_h / spatial_merge_size;
        size_t llm_grid_w = grid_w / spatial_merge_size;
        size_t spatial_area = llm_grid_h * llm_grid_w;
        OPENVINO_ASSERT(spatial_area > 0, "Vision region must contain at least one spatial token");

        size_t t = std::max<size_t>(1, grid_t);
        size_t total_tokens = spatial_area * t;

        regions.push_back({total_tokens, std::max<size_t>(1, llm_grid_w), spatial_area, cumulative_offset});
        cumulative_offset += total_tokens;
    }

    // Normalize kept indices: convert aggregated indices to per-region format if needed
    auto normalize_kept_indices = [&]() -> std::vector<std::vector<size_t>> {
        if (kept_indices_per_image.empty())
            return std::vector<std::vector<size_t>>(region_count);

        if (kept_indices_per_image.size() == region_count) {
            return kept_indices_per_image;
        }

        // Handle single aggregated vector case
        OPENVINO_ASSERT(kept_indices_per_image.size() == 1 && region_count > 1,
                        "Kept token indices layout does not match vision regions. Got " +
                            std::to_string(kept_indices_per_image.size()) + " vectors, expected 1 or " +
                            std::to_string(region_count));

        std::vector<std::vector<size_t>> normalized(region_count);
        for (size_t kept_idx : kept_indices_per_image[0]) {
            OPENVINO_ASSERT(kept_idx < cumulative_offset,
                            "Aggregated kept index " + std::to_string(kept_idx) + " out of range [0, " +
                                std::to_string(cumulative_offset) + ")");
            for (size_t img_idx = 0; img_idx < region_count; ++img_idx) {
                if (kept_idx >= regions[img_idx].offset &&
                    kept_idx < regions[img_idx].offset + regions[img_idx].tokens) {
                    size_t local_idx = kept_idx - regions[img_idx].offset;
                    normalized[img_idx].push_back(local_idx);
                    break;
                }
            }
        }
        return normalized;
    };

    auto normalized_indices = normalize_kept_indices();

    // Sort and deduplicate each region's indices
    for (auto& indices : normalized_indices) {
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    }

    // Build keep flags and calculate new sequence length
    keep_flags_out.clear();
    keep_flags_out.reserve(region_count);
    size_t total_removed = 0;

    for (size_t idx = 0; idx < region_count; ++idx) {
        std::vector<bool> flags(regions[idx].tokens, false);
        for (size_t kept_idx : normalized_indices[idx]) {
            OPENVINO_ASSERT(kept_idx < regions[idx].tokens, "Kept token index out of range");
            flags[kept_idx] = true;
        }
        size_t kept_count = std::count(flags.begin(), flags.end(), true);
        OPENVINO_ASSERT(kept_count <= regions[idx].tokens, "Kept tokens exceed region size");
        total_removed += regions[idx].tokens - kept_count;
        keep_flags_out.push_back(std::move(flags));
    }

    OPENVINO_ASSERT(seq_len >= total_removed, "Sequence length underflow after pruning");
    size_t new_seq_len = seq_len - total_removed;

    // Allocate new position IDs tensor
    ov::Tensor new_position_ids(original_position_ids.get_element_type(), {3, batch_size, new_seq_len});
    int64_t* pos_data[3] = {new_position_ids.data<int64_t>(),
                            new_position_ids.data<int64_t>() + batch_size * new_seq_len,
                            new_position_ids.data<int64_t>() + 2 * batch_size * new_seq_len};

    const int64_t* input_ids_data = input_ids.data<const int64_t>();

    // Process each batch
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t write_idx = 0, image_idx = 0, visual_idx = 0;
        bool inside_vision = false;
        int64_t next_pos = 0, grid_base = 0;
        int64_t max_pos[3] = {-1, -1, -1};  // temporal, height, width
        size_t batch_offset = batch_idx * seq_len;
        size_t out_offset = batch_idx * new_seq_len;

        for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            int64_t token_id = input_ids_data[batch_offset + seq_idx];

            // Handle vision pad tokens (image_pad or video_pad) inside vision region
            bool is_vision_pad =
                (token_id == image_pad_token_id) || (video_pad_token_id >= 0 && token_id == video_pad_token_id);
            if (inside_vision && is_vision_pad) {
                OPENVINO_ASSERT(image_idx < region_count, "Vision region index out of bounds");
                if (keep_flags_out[image_idx][visual_idx]) {
                    const auto& region = regions[image_idx];
                    size_t local_idx = visual_idx % region.spatial_area;
                    size_t temporal_idx = visual_idx / region.spatial_area;
                    size_t row = local_idx / region.grid_width;
                    size_t col = local_idx % region.grid_width;

                    int64_t pos_vals[3] = {grid_base + static_cast<int64_t>(temporal_idx),
                                           grid_base + static_cast<int64_t>(row),
                                           grid_base + static_cast<int64_t>(col)};

                    for (int dim = 0; dim < 3; ++dim) {
                        pos_data[dim][out_offset + write_idx] = pos_vals[dim];
                        max_pos[dim] = std::max(max_pos[dim], pos_vals[dim]);
                    }
                    ++write_idx;
                }
                ++visual_idx;
                continue;
            }

            // Handle end of vision region
            if (inside_vision) {
                inside_vision = false;
                next_pos = std::max({max_pos[0], max_pos[1], max_pos[2]}) + 1;
                ++image_idx;
                visual_idx = 0;
                std::fill(max_pos, max_pos + 3, next_pos - 1);
            }

            // Write text token position
            for (int dim = 0; dim < 3; ++dim) {
                pos_data[dim][out_offset + write_idx] = next_pos;
            }
            ++write_idx;
            ++next_pos;

            // Handle start of vision region
            if (token_id == vision_start_token_id) {
                inside_vision = true;
                visual_idx = 0;
                grid_base = next_pos;
            }
        }

        OPENVINO_ASSERT(!inside_vision, "Unexpected end of sequence inside vision region");
        OPENVINO_ASSERT(image_idx == region_count, "Not all vision regions processed");
        OPENVINO_ASSERT(write_idx == new_seq_len, "Output sequence length mismatch");
    }

    return new_position_ids;
}

// 1D position IDs update for LLaVA-style models
ov::Tensor VisionTokenPruningProcessor::update_position_ids_1d(
    const ov::Tensor& original_position_ids,
    const ov::Tensor& input_ids,
    int64_t vision_start_token_id,
    int64_t image_pad_token_id,
    const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw,
    const std::vector<std::vector<size_t>>& kept_indices_per_image,
    std::vector<std::vector<bool>>& keep_flags_out,
    int64_t video_pad_token_id) const {
    const ov::Shape& pos_shape = original_position_ids.get_shape();
    OPENVINO_ASSERT(pos_shape.size() == 2, "1D position ids must be [batch, seq_len]");

    const size_t batch_size = pos_shape[0];
    const size_t seq_len = pos_shape[1];
    const size_t region_count = reordered_images_grid_thw.size();

    // Calculate tokens per image
    std::vector<size_t> tokens_per_image;
    tokens_per_image.reserve(region_count);
    size_t cumulative_offset = 0;

    for (const auto& [grid_t, grid_h, grid_w] : reordered_images_grid_thw) {
        size_t tokens = grid_t * grid_h * grid_w;
        tokens_per_image.push_back(tokens);
        cumulative_offset += tokens;
    }

    // Normalize kept indices
    auto normalize_kept_indices = [&]() -> std::vector<std::vector<size_t>> {
        if (kept_indices_per_image.empty())
            return std::vector<std::vector<size_t>>(region_count);

        if (kept_indices_per_image.size() == region_count) {
            return kept_indices_per_image;
        }

        // Handle aggregated case
        OPENVINO_ASSERT(kept_indices_per_image.size() == 1 && region_count > 1, "Kept token indices layout mismatch");

        std::vector<std::vector<size_t>> normalized(region_count);
        size_t offset = 0;
        for (size_t kept_idx : kept_indices_per_image[0]) {
            for (size_t img_idx = 0; img_idx < region_count; ++img_idx) {
                if (kept_idx >= offset && kept_idx < offset + tokens_per_image[img_idx]) {
                    normalized[img_idx].push_back(kept_idx - offset);
                    break;
                }
                offset += tokens_per_image[img_idx];
            }
        }
        return normalized;
    };

    auto normalized_indices = normalize_kept_indices();

    // Build keep flags
    keep_flags_out.clear();
    keep_flags_out.reserve(region_count);
    size_t total_removed = 0;

    for (size_t idx = 0; idx < region_count; ++idx) {
        std::vector<bool> flags(tokens_per_image[idx], false);
        for (size_t kept_idx : normalized_indices[idx]) {
            flags[kept_idx] = true;
        }
        size_t kept_count = std::count(flags.begin(), flags.end(), true);
        total_removed += tokens_per_image[idx] - kept_count;
        keep_flags_out.push_back(std::move(flags));
    }

    size_t new_seq_len = seq_len - total_removed;

    // Allocate new position IDs
    ov::Tensor new_position_ids(original_position_ids.get_element_type(), {batch_size, new_seq_len});
    int64_t* pos_data = new_position_ids.data<int64_t>();
    const int64_t* input_ids_data = input_ids.data<const int64_t>();

    // Process each batch
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t write_idx = 0, image_idx = 0, visual_idx = 0;
        bool inside_vision = false;
        int64_t next_pos = 0;
        size_t batch_offset = batch_idx * seq_len;
        size_t out_offset = batch_idx * new_seq_len;

        for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            int64_t token_id = input_ids_data[batch_offset + seq_idx];

            bool is_vision_pad_1d =
                (token_id == image_pad_token_id) || (video_pad_token_id >= 0 && token_id == video_pad_token_id);
            if (inside_vision && is_vision_pad_1d) {
                if (keep_flags_out[image_idx][visual_idx]) {
                    pos_data[out_offset + write_idx] = next_pos;
                    ++write_idx;
                    ++next_pos;
                }
                ++visual_idx;
                continue;
            }

            if (inside_vision && !is_vision_pad_1d) {
                inside_vision = false;
                ++image_idx;
                visual_idx = 0;
            }

            pos_data[out_offset + write_idx] = next_pos;
            ++write_idx;
            ++next_pos;

            if (token_id == vision_start_token_id) {
                inside_vision = true;
                visual_idx = 0;
            }
        }
    }

    return new_position_ids;
}

std::optional<VisionTokenPruningProcessor::PruningResult> VisionTokenPruningProcessor::execute(
    const PruningContext& context,
    ov::Tensor& position_ids,
    utils::CacheState& cache_state,
    size_t& prev_hist_length) {
    auto pruning_start = std::chrono::high_resolution_clock::now();

    // ---- Validate inputs ----
    // Placeholder fields may be default-constructed ov::Tensor (no _impl);
    // use operator bool() before calling any accessors.
    const bool has_images = static_cast<bool>(context.image_embeddings);
    const bool has_videos = static_cast<bool>(context.video_embeddings);
    const size_t image_token_count = has_images ? context.image_embeddings.get_shape()[0] : 0;
    const size_t video_token_count = has_videos ? context.video_embeddings.get_shape()[0] : 0;
    const size_t total_tokens = image_token_count + video_token_count;
    if (total_tokens == 0) {
        // No vision tokens to prune (caller may invoke this with empty merged embeddings
        // when the prompt contains no pad-token runs). Skip pruning entirely.
        return std::nullopt;
    }
    OPENVINO_ASSERT(image_token_count == 0 || context.image_pad_token_id != -1,
                    "image_embeddings provided without image_pad_token_id");
    OPENVINO_ASSERT(video_token_count == 0 || context.video_pad_token_id != -1,
                    "video_embeddings provided without video_pad_token_id");
    OPENVINO_ASSERT(image_token_count == 0 || !context.image_grids.empty(),
                    "image_embeddings provided without image_grids");
    OPENVINO_ASSERT(video_token_count == 0 || !context.video_grids.empty(),
                    "video_embeddings provided without video_grids");

    // ---- Determine hidden dim / element type from the non-empty modality ----
    const size_t hidden_dim =
        (video_token_count > 0) ? context.video_embeddings.get_shape()[1] : context.image_embeddings.get_shape()[1];
    const auto embed_element_type = (video_token_count > 0) ? context.video_embeddings.get_element_type()
                                                            : context.image_embeddings.get_element_type();
    if (image_token_count > 0 && video_token_count > 0) {
        OPENVINO_ASSERT(context.image_embeddings.get_shape()[1] == context.video_embeddings.get_shape()[1],
                        "Image and video embedding widths must match");
    }
    const size_t row_bytes = hidden_dim * embed_element_type.size();

    // ---- Single scan over input_ids: build prompt_to_merger and combined_grid_thw ----
    // Each contiguous run of identical pad tokens is one region. The merger layout is
    // [video_tokens; image_tokens]: video occupies indices [0, video_token_count),
    // image occupies indices [video_token_count, total_tokens).
    OPENVINO_ASSERT(context.input_ids.get_shape().at(0) == 1,
                    "Vision token pruning pipeline only supports batch_size == 1, got: ",
                    context.input_ids.get_shape().at(0));
    const int64_t* ids = context.input_ids.data<const int64_t>();
    const size_t seq_len = context.input_ids.get_shape().at(1);

    std::vector<std::array<size_t, 3>> combined_grid_thw;
    combined_grid_thw.reserve(context.video_grids.size() + context.image_grids.size());
    std::vector<size_t> prompt_to_merger;
    prompt_to_merger.reserve(total_tokens);

    size_t v_region = 0;
    size_t i_region = 0;
    size_t v_idx = 0;
    size_t i_idx = 0;
    int64_t prev_pad = -1;
    for (size_t pos = 0; pos < seq_len; ++pos) {
        const int64_t tok = ids[pos];
        const bool is_video = (context.video_pad_token_id != -1 && tok == context.video_pad_token_id);
        const bool is_image = (context.image_pad_token_id != -1 && tok == context.image_pad_token_id);
        if (!is_video && !is_image) {
            prev_pad = -1;
            continue;
        }
        if (tok != prev_pad) {
            if (is_video) {
                OPENVINO_ASSERT(v_region < context.video_grids.size(),
                                "More video regions in prompt than entries in video_grids");
                combined_grid_thw.push_back(context.video_grids[v_region++]);
            } else {
                OPENVINO_ASSERT(i_region < context.image_grids.size(),
                                "More image regions in prompt than entries in image_grids");
                combined_grid_thw.push_back(context.image_grids[i_region++]);
            }
            prev_pad = tok;
        }
        prompt_to_merger.push_back(is_video ? v_idx++ : video_token_count + i_idx++);
    }
    OPENVINO_ASSERT(prompt_to_merger.size() == total_tokens,
                    "prompt-order vision tokens (",
                    prompt_to_merger.size(),
                    ") != merged vision tokens (",
                    total_tokens,
                    ")");
    OPENVINO_ASSERT(v_idx == video_token_count, "Video pad tokens in prompt do not cover all video embeddings");
    OPENVINO_ASSERT(i_idx == image_token_count, "Image pad tokens in prompt do not cover all image embeddings");
    OPENVINO_ASSERT(v_region == context.video_grids.size(),
                    "Fewer video regions in prompt than entries in video_grids (",
                    v_region, " vs ", context.video_grids.size(), ")");
    OPENVINO_ASSERT(i_region == context.image_grids.size(),
                    "Fewer image regions in prompt than entries in image_grids (",
                    i_region, " vs ", context.image_grids.size(), ")");

    const size_t total_regions = combined_grid_thw.size();
    const size_t spatial_merge_size = std::max<size_t>(1, context.spatial_merge_size);
    const size_t merge_length = spatial_merge_size * spatial_merge_size;

    std::vector<size_t> tokens_per_vision;
    tokens_per_vision.reserve(total_regions);
    for (const auto& [gt, gh, gw] : combined_grid_thw) {
        const size_t region_elements = gt * gh * gw;
        OPENVINO_ASSERT(region_elements % merge_length == 0,
                        "Vision grid elements (", region_elements,
                        ") must be divisible by merge_length (", merge_length, ")");
        tokens_per_vision.push_back(region_elements / merge_length);
    }
    const size_t derived_total_tokens = std::accumulate(tokens_per_vision.begin(), tokens_per_vision.end(), size_t{0});
    OPENVINO_ASSERT(derived_total_tokens == total_tokens,
                    "Derived per-vision token count (", derived_total_tokens,
                    ") != merged vision tokens (", total_tokens, ")");

    std::vector<size_t> combined_sequence(total_regions);
    std::iota(combined_sequence.begin(), combined_sequence.end(), 0);

    // ---- Gather per-modality embeddings into prompt order ----
    ov::Tensor combined_embeds(embed_element_type, {total_tokens, hidden_dim});
    {
        auto* dst = static_cast<uint8_t*>(combined_embeds.data());
        const auto* vid_src =
            video_token_count > 0 ? static_cast<const uint8_t*>(context.video_embeddings.data()) : nullptr;
        const auto* img_src =
            image_token_count > 0 ? static_cast<const uint8_t*>(context.image_embeddings.data()) : nullptr;
        for (size_t k = 0; k < total_tokens; ++k) {
            const size_t src_idx = prompt_to_merger[k];
            const auto* src = (src_idx < video_token_count) ? vid_src + src_idx * row_bytes
                                                            : img_src + (src_idx - video_token_count) * row_bytes;
            std::memcpy(dst + k * row_bytes, src, row_bytes);
        }
    }

    PruningResult result;
    result.original_visual_tokens = total_tokens;

    // ---- Step 1: Extract text features for relevance calculation ----
    auto current_pruning_config = get_config();
    ov::Tensor text_features = extract_text_features(context.text_embeds,
                                                     context.input_ids,
                                                     context.image_pad_token_id,
                                                     context.vision_start_token_id,
                                                     context.vision_end_token_id,
                                                     context.video_pad_token_id);

    // ---- Step 2: Convert visual features to CDPruner format ----
    const size_t chunk_count = current_pruning_config.enable_frame_chunking ? total_regions : 1;
    OPENVINO_ASSERT(!current_pruning_config.enable_frame_chunking || chunk_count > 0,
                    "Frame chunking requires at least one vision region");
    std::vector<ov::Tensor> visual_features = convert_visual_features(combined_embeds, chunk_count, tokens_per_vision);

    // ---- Step 3: Apply token selection (DPP) ----
    ov::Tensor pruned_visual_features = process(visual_features, text_features);

    // ---- Step 4: Reshape to 2D and copy into a packed tensor ----
    OPENVINO_ASSERT(pruned_visual_features.get_element_type() == ov::element::f32,
                    "CDPruner pruned visual features must be f32, got: ",
                    pruned_visual_features.get_element_type());
    const ov::Shape pruned_shape = pruned_visual_features.get_shape();
    OPENVINO_ASSERT(pruned_shape.size() == 3,
                    "CDPruner pruned visual features expected 3D shape [1, tokens, hidden], got rank: ",
                    pruned_shape.size());
    result.pruned_visual_tokens = pruned_shape[1];
    const size_t out_hidden_size = pruned_shape[2];
    ov::Tensor pruned_2d_tensor(pruned_visual_features.get_element_type(),
                                {result.pruned_visual_tokens, out_hidden_size});
    std::memcpy(pruned_2d_tensor.data(),
                pruned_visual_features.data<const float>(),
                result.pruned_visual_tokens * out_hidden_size * sizeof(float));

    if (result.original_visual_tokens == result.pruned_visual_tokens) {
        GENAI_INFO("Original visual tokens and pruned visual tokens are the same!");
        return std::nullopt;
    }

    // ---- Step 5: Adjust position_ids ----
    adjust_position_ids(position_ids,
                        context.input_ids,
                        combined_grid_thw,
                        combined_sequence,
                        context.image_pad_token_id,
                        context.vision_start_token_id,
                        spatial_merge_size,
                        result.keep_flags_per_region,
                        context.video_pad_token_id);

    // ---- Step 6: Validate metadata ----
    OPENVINO_ASSERT(!result.keep_flags_per_region.empty(), "keep_flags_per_region is empty after pruning");
    size_t total_original = 0;
    size_t total_kept = 0;
    for (const auto& mask : result.keep_flags_per_region) {
        total_original += mask.size();
        total_kept += static_cast<size_t>(std::count(mask.begin(), mask.end(), true));
    }
    OPENVINO_ASSERT(total_original == result.original_visual_tokens,
                    "Original visual token metadata mismatch. Expected: ",
                    result.original_visual_tokens,
                    ", got: ",
                    total_original);
    OPENVINO_ASSERT(total_kept == result.pruned_visual_tokens,
                    "Pruned visual token metadata mismatch. Expected: ",
                    result.pruned_visual_tokens,
                    ", got: ",
                    total_kept);
    OPENVINO_ASSERT(result.keep_flags_per_region.size() == total_regions,
                    "Region count mismatch in keep_flags_per_region");

    // ---- Step 7: Cache keep_flags ----
    m_last_keep_flags = result.keep_flags_per_region;

    // ---- Step 8: Generate pruned input_ids and text embeddings ----
    result.pruned_input_ids = generate_pruned_input_ids(context.input_ids,
                                                        result.keep_flags_per_region,
                                                        context.image_pad_token_id,
                                                        context.vision_start_token_id,
                                                        context.vision_end_token_id,
                                                        context.video_pad_token_id);
    result.pruned_text_embeds = generate_pruned_text_embeds(context.input_ids,
                                                            context.text_embeds,
                                                            context.image_pad_token_id,
                                                            context.vision_start_token_id,
                                                            context.vision_end_token_id,
                                                            result.keep_flags_per_region,
                                                            context.video_pad_token_id);

    // ---- Step 9: Update rope_delta ----
    {
        const int64_t* pos_data = position_ids.data<const int64_t>();
        const int64_t max_pos = *std::max_element(pos_data, pos_data + position_ids.get_size());
        const size_t pos_seq_len = position_ids.get_shape().back();
        result.updated_rope_delta = max_pos + 1 - static_cast<int64_t>(pos_seq_len);
    }

    // ---- Step 10: Update cache state ----
    {
        auto& cache_history = cache_state.get_state();
        OPENVINO_ASSERT(cache_history.size() >= context.input_ids.get_size(),
                        "Cache history does not contain expected original prompt length");
        cache_history.resize(prev_hist_length);
        cache_state.add_inputs(result.pruned_input_ids);
        prev_hist_length = cache_state.get_state().size();
    }

    // ---- Step 11: Scatter pruned embeddings back into per-modality tensors ----
    std::vector<size_t> kept_merger_indices;
    kept_merger_indices.reserve(result.pruned_visual_tokens);
    size_t kept_video = 0;
    size_t kept_image = 0;
    {
        size_t prompt_pos = 0;
        for (const auto& region_flags : result.keep_flags_per_region) {
            for (bool keep : region_flags) {
                if (keep) {
                    const size_t merger_idx = prompt_to_merger[prompt_pos];
                    kept_merger_indices.push_back(merger_idx);
                    if (merger_idx < video_token_count)
                        ++kept_video;
                    else
                        ++kept_image;
                }
                ++prompt_pos;
            }
        }
    }

    const auto pruned_elem_type = pruned_2d_tensor.get_element_type();
    result.pruned_video_embeddings = ov::Tensor(pruned_elem_type, {kept_video, out_hidden_size});
    result.pruned_image_embeddings = ov::Tensor(pruned_elem_type, {kept_image, out_hidden_size});
    {
        const size_t out_row_bytes = out_hidden_size * pruned_elem_type.size();
        const auto* src = static_cast<const uint8_t*>(pruned_2d_tensor.data());
        auto* vid_dst = kept_video > 0 ? static_cast<uint8_t*>(result.pruned_video_embeddings.data()) : nullptr;
        auto* img_dst = kept_image > 0 ? static_cast<uint8_t*>(result.pruned_image_embeddings.data()) : nullptr;
        size_t dst_v = 0;
        size_t dst_i = 0;
        for (size_t k = 0; k < kept_merger_indices.size(); ++k) {
            const size_t merger_idx = kept_merger_indices[k];
            auto* dst = (merger_idx < video_token_count) ? vid_dst + dst_v++ * out_row_bytes
                                                         : img_dst + dst_i++ * out_row_bytes;
            std::memcpy(dst, src + k * out_row_bytes, out_row_bytes);
        }
    }

    // ---- Step 12: Optionally prune deepstack tensor in place ----
    if (context.deepstack_visual_embeds && *context.deepstack_visual_embeds &&
        context.deepstack_visual_embeds->get_size() > 0) {
        const auto deepstack_shape = context.deepstack_visual_embeds->get_shape();
        if (deepstack_shape.size() == 3 && result.pruned_visual_tokens < deepstack_shape[1]) {
            const size_t num_layers = deepstack_shape[0];
            const size_t original_tokens = deepstack_shape[1];
            const size_t ds_hidden_size = deepstack_shape[2];
            const auto ds_elem_type = context.deepstack_visual_embeds->get_element_type();
            const size_t ds_token_bytes = ds_hidden_size * ds_elem_type.size();

            OPENVINO_ASSERT(original_tokens == total_tokens,
                            "Deepstack token count (",
                            original_tokens,
                            ") != merged vision tokens (",
                            total_tokens,
                            ")");

            std::vector<size_t> sorted_indices = kept_merger_indices;
            std::sort(sorted_indices.begin(), sorted_indices.end());

            const size_t kept_count = sorted_indices.size();
            ov::Tensor pruned_deepstack(ds_elem_type, {num_layers, kept_count, ds_hidden_size});
            const auto* src = static_cast<const uint8_t*>(context.deepstack_visual_embeds->data());
            auto* dst = static_cast<uint8_t*>(pruned_deepstack.data());

            for (size_t layer = 0; layer < num_layers; ++layer) {
                const size_t layer_src_offset = layer * original_tokens * ds_token_bytes;
                const size_t layer_dst_offset = layer * kept_count * ds_token_bytes;
                size_t dst_idx = 0;
                while (dst_idx < kept_count) {
                    const size_t run_src_start = sorted_indices[dst_idx];
                    size_t run_length = 1;
                    while (dst_idx + run_length < kept_count &&
                           sorted_indices[dst_idx + run_length] == run_src_start + run_length) {
                        ++run_length;
                    }
                    std::memcpy(dst + layer_dst_offset + dst_idx * ds_token_bytes,
                                src + layer_src_offset + run_src_start * ds_token_bytes,
                                run_length * ds_token_bytes);
                    dst_idx += run_length;
                }
            }
            *context.deepstack_visual_embeds = std::move(pruned_deepstack);
        }
    }

    auto pruning_end = std::chrono::high_resolution_clock::now();
    auto pruning_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pruning_end - pruning_start).count();

    GENAI_INFO("CDPruner Summary:");
    GENAI_INFO("\tConfiguration:");
    GENAI_INFO("\t  Pruning Ratio: %d%%", current_pruning_config.pruning_ratio);
    GENAI_INFO("\t  Relevance Weight: %.1f", current_pruning_config.relevance_weight);
    GENAI_INFO("\t  Use CL Kernel: %s", current_pruning_config.use_cl_kernel ? "true" : "false");
    GENAI_INFO("\t  Enable Frame Chunking: %s", current_pruning_config.enable_frame_chunking ? "true" : "false");
    GENAI_INFO("\t  Use Negative Relevance: %s", current_pruning_config.use_negative_relevance ? "true" : "false");
    const bool exceeds_split_threshold = (current_pruning_config.split_threshold > 0) &&
                                         (result.original_visual_tokens > current_pruning_config.split_threshold);
    GENAI_INFO("\t  Exceeds Split Threshold: %s", exceeds_split_threshold ? "true" : "false");
    GENAI_INFO("\tResults:");
    GENAI_INFO("\t  Original Visual Tokens: %zu", result.original_visual_tokens);
    GENAI_INFO("\t  Removed Visual Tokens: %zu", result.original_visual_tokens - result.pruned_visual_tokens);
    GENAI_INFO("\t  Actual Pruning Ratio: %.2f%%",
               static_cast<float>(result.original_visual_tokens - result.pruned_visual_tokens) /
                   result.original_visual_tokens * 100.0f);
    GENAI_INFO("\tTotal Pruning Time: %ld ms", pruning_duration);

    return result;
}

}  // namespace ov::genai
