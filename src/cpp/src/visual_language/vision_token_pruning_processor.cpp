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

// Extract text features by averaging instruction token embeddings
ov::Tensor VisionTokenPruningProcessor::extract_text_features(const ov::Tensor& text_embeds,
                                                              const ov::Tensor& input_ids,
                                                              int64_t image_pad_token_id,
                                                              int64_t vision_start_token_id,
                                                              int64_t vision_end_token_id) const {
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
        if (inside_vision_region || current_token == image_pad_token_id) {
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
    int64_t vision_end_token_id) const {
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

        if (inside_vision_region && token_id == image_pad_token_id) {
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
    const std::vector<std::vector<bool>>& keep_flags_per_region) const {
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

            // Skip filtered image_pad tokens
            if (inside_vision_region && token_id == image_pad_token_id) {
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

void VisionTokenPruningProcessor::adjust_position_ids(ov::Tensor& position_ids_inout,
                                                      const ov::Tensor& input_ids,
                                                      const std::vector<std::array<size_t, 3>>& images_grid_thw,
                                                      const std::vector<size_t>& images_sequence,
                                                      int64_t image_pad_token_id,
                                                      int64_t vision_start_token_id,
                                                      size_t spatial_merge_size,
                                                      std::vector<std::vector<bool>>& keep_flags_per_region_out) const {
    auto kept_indices_per_image = get_last_selected_tokens();
    OPENVINO_ASSERT(!images_sequence.empty(), "Image sequence must not be empty when pruning visual tokens");
    OPENVINO_ASSERT(!kept_indices_per_image.empty(), "Kept token indices are missing after pruning");

    // Reorder images according to sequence
    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;
    reordered_images_grid_thw.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        OPENVINO_ASSERT(new_image_id < images_grid_thw.size(), "Image sequence index is out of range");
        reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id));
    }

    // Detect position encoding type from shape
    const ov::Shape& pos_shape = position_ids_inout.get_shape();
    bool is_3d_encoding = (pos_shape.size() == 3 && pos_shape[0] == 3);

    if (is_3d_encoding) {
        // 3D RoPE position encoding (Qwen2VL style)
        position_ids_inout = update_position_ids_3d(position_ids_inout,
                                                    input_ids,
                                                    vision_start_token_id,
                                                    image_pad_token_id,
                                                    reordered_images_grid_thw,
                                                    kept_indices_per_image,
                                                    spatial_merge_size,
                                                    keep_flags_per_region_out);
    } else {
        // 1D position encoding (LLaVA, MiniCPM, etc.)
        position_ids_inout = update_position_ids_1d(position_ids_inout,
                                                    input_ids,
                                                    vision_start_token_id,
                                                    image_pad_token_id,
                                                    reordered_images_grid_thw,
                                                    kept_indices_per_image,
                                                    keep_flags_per_region_out);
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
    std::vector<std::vector<bool>>& keep_flags_out) const {
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

            // Handle image pad tokens inside vision region
            if (inside_vision && token_id == image_pad_token_id) {
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
    std::vector<std::vector<bool>>& keep_flags_out) const {
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

            if (inside_vision && token_id == image_pad_token_id) {
                if (keep_flags_out[image_idx][visual_idx]) {
                    pos_data[out_offset + write_idx] = next_pos;
                    ++write_idx;
                    ++next_pos;
                }
                ++visual_idx;
                continue;
            }

            if (inside_vision && token_id != image_pad_token_id) {
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
    utils::KVCacheState& kv_cache_state,
    size_t prev_hist_length) {
    auto pruning_start = std::chrono::high_resolution_clock::now();

    PruningResult result;
    result.original_visual_tokens = context.merged_visual_embeddings.get_shape()[0];

    // Retrieve current pruning configuration
    auto current_pruning_config = get_config();

    // Step 1: Extract text features for relevance calculation
    ov::Tensor text_features = extract_text_features(context.text_embeds,
                                                     context.input_ids,
                                                     context.vision_pad_token_id,
                                                     context.vision_start_token_id,
                                                     context.vision_end_token_id);

    // Step 2: Convert visual features to CDPruner format
    OPENVINO_ASSERT(!current_pruning_config.enable_frame_chunking || context.vision_count > 0,
                    "vision_count must be non-zero when frame chunking is enabled");
    size_t chunk_count = current_pruning_config.enable_frame_chunking ? context.vision_count : 1;
    std::vector<ov::Tensor> visual_features =
        convert_visual_features(context.merged_visual_embeddings, chunk_count, context.tokens_per_vision);

    // Step 3: Apply vision token processing (pruning)
    ov::Tensor pruned_visual_features = process(visual_features, text_features);

    // Step 4: Reshape from 3D [1, num_tokens, hidden_size] to 2D [num_tokens, hidden_size]
    ov::Shape pruned_shape = pruned_visual_features.get_shape();
    result.pruned_visual_tokens = pruned_shape[1];
    size_t hidden_size = pruned_shape[2];

    ov::Tensor pruned_2d_tensor(pruned_visual_features.get_element_type(), {result.pruned_visual_tokens, hidden_size});
    const float* src_data = pruned_visual_features.data<const float>();
    float* dst_data = pruned_2d_tensor.data<float>();
    std::memcpy(dst_data, src_data, result.pruned_visual_tokens * hidden_size * sizeof(float));

    result.pruned_embeddings = pruned_2d_tensor;

    if (result.original_visual_tokens == result.pruned_visual_tokens) {
        GENAI_INFO("Original visual tokens and pruned visual tokens are the same!");
        return std::nullopt;
    }

    // Step 5: Adjust position_ids to account for removed visual tokens
    adjust_position_ids(position_ids,
                        context.input_ids,
                        context.visions_grid_thw,
                        context.visions_sequence,
                        context.vision_pad_token_id,
                        context.vision_start_token_id,
                        context.spatial_merge_size,
                        result.keep_flags_per_region);

    // Step 6: Validate pruning metadata consistency
    OPENVINO_ASSERT(result.keep_flags_per_region.size() > 0, "Vision region metadata not available for pruning");

    size_t total_original_tokens = 0;
    size_t total_pruned_tokens = 0;
    for (size_t i = 0; i < result.keep_flags_per_region.size(); ++i) {
        const auto& mask = result.keep_flags_per_region[i];
        size_t region_original = mask.size();
        size_t region_kept = static_cast<size_t>(std::count(mask.begin(), mask.end(), true));
        total_original_tokens += region_original;
        total_pruned_tokens += region_kept;
    }

    OPENVINO_ASSERT(total_original_tokens == result.original_visual_tokens,
                    "Original visual token metadata mismatch after pruning. Expected: " +
                        std::to_string(result.original_visual_tokens) +
                        ", Got: " + std::to_string(total_original_tokens));
    OPENVINO_ASSERT(total_pruned_tokens == result.pruned_visual_tokens,
                    "Pruned visual token metadata mismatch after pruning. Expected: " +
                        std::to_string(result.pruned_visual_tokens) + ", Got: " + std::to_string(total_pruned_tokens));
    OPENVINO_ASSERT(result.keep_flags_per_region.size() == context.visions_sequence.size(),
                    "Kept visual token mask count mismatch with vision regions");

    // Step 7: Generate pruned input_ids with visual tokens removed
    result.pruned_input_ids = generate_pruned_input_ids(context.input_ids,
                                                        result.keep_flags_per_region,
                                                        context.vision_pad_token_id,
                                                        context.vision_start_token_id,
                                                        context.vision_end_token_id);

    // Step 8: Generate pruned text embeddings
    result.pruned_text_embeds = generate_pruned_text_embeds(context.input_ids,
                                                            context.text_embeds,
                                                            context.vision_pad_token_id,
                                                            context.vision_start_token_id,
                                                            context.vision_end_token_id,
                                                            result.keep_flags_per_region);

    // Step 9: Update rope_delta to maintain position continuity in generation phase
    const int64_t* pos_data = position_ids.data<const int64_t>();
    int64_t max_pos = *std::max_element(pos_data, pos_data + position_ids.get_size());
    size_t seq_len = position_ids.get_shape().back();
    result.updated_rope_delta = max_pos + 1 - static_cast<int64_t>(seq_len);

    // Step 10: Update KV cache with pruned input_ids
    auto& kv_history = kv_cache_state.get_state();
    OPENVINO_ASSERT(kv_history.size() >= context.input_ids.get_size(),
                    "KV cache history does not contain expected original prompt length");
    OPENVINO_ASSERT(kv_history.size() >= prev_hist_length,
                    "KV cache history is shorter than recorded previous history length");
    kv_history.resize(prev_hist_length);
    kv_cache_state.add_inputs(result.pruned_input_ids);

    auto pruning_end = std::chrono::high_resolution_clock::now();
    auto pruning_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pruning_end - pruning_start).count();

    // CDPruner Summary
    GENAI_INFO("CDPruner Summary:");
    GENAI_INFO("\tConfiguration:");
    GENAI_INFO("\t  Pruning Ratio: %d%%", current_pruning_config.pruning_ratio);
    GENAI_INFO("\t  Relevance Weight: %.1f", current_pruning_config.relevance_weight);
    GENAI_INFO("\t  Use CL Kernel: %s", current_pruning_config.use_cl_kernel ? "true" : "false");
    GENAI_INFO("\t  Enable Frame Chunking: %s", current_pruning_config.enable_frame_chunking ? "true" : "false");
    GENAI_INFO("\t  Use Negative Relevance: %s", current_pruning_config.use_negative_relevance ? "true" : "false");
    bool exceeds_split_threshold = (current_pruning_config.split_threshold > 0) &&
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
