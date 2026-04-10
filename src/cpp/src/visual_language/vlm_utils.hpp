// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <regex>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai::vlm_utils {

std::vector<std::variant<ov::Tensor, size_t>> split_tokenize(const std::string& text,
                                                             ov::genai::Tokenizer& tokenizer,
                                                             const std::regex& native_pattern);

ov::Tensor insert_image_placeholders(const std::vector<std::variant<ov::Tensor, size_t>>& chunks,
                                     const std::vector<size_t>& tokens_per_images);

std::vector<std::variant<ov::Tensor, size_t>> drop_image_placeholders(const ov::Tensor& tokens);

// Build final inputs_embeds by interleaving inferred text embeddings and precomputed visual embeddings.
// chunks contains either text token tensors or visual ids that index visual_embeds with base_id offset.
//
// Assumptions:
// - Both text and visual embedding tensors must have:
//     - element type: f32
//     - shape: [1, length, hidden_size] (rank 3, batch size 1, last dim == hidden_size)
// These are checked explicitly before copying data.
template <typename InferTextEmbeddings>
ov::Tensor build_inputs_embeds_from_text_and_visual_chunks(
    const std::vector<std::variant<ov::Tensor, size_t>>& chunks,
    InferTextEmbeddings&& infer_text_embeddings,
    const std::vector<ov::Tensor>& visual_embeds,
    size_t base_id,
    size_t sequence_length,
    size_t hidden_size) {
    OPENVINO_ASSERT(hidden_size > 0, "hidden_size must be greater than 0.");
    ov::Tensor inputs_embeds{ov::element::f32, {1, sequence_length, hidden_size}};
    float* inputs_embeds_ptr = inputs_embeds.data<float>();
    size_t offset = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        offset += std::visit(
            [&](const auto& chunk_value) {
                using ChunkType = std::decay_t<decltype(chunk_value)>;
                if constexpr (std::is_same_v<ChunkType, ov::Tensor>) {
                    const ov::Tensor text_embeds = infer_text_embeddings(chunk_value);
                    const auto& text_shape = text_embeds.get_shape();
                    OPENVINO_ASSERT(text_shape.size() == 3, "text_embeds must have rank 3, got ", text_shape.size(), ".");
                    OPENVINO_ASSERT(text_shape[0] == 1, "text_embeds batch size must be 1, got ", text_shape[0], ".");
                    OPENVINO_ASSERT(text_shape[2] == hidden_size, "text_embeds last dim must match hidden_size (", hidden_size, "), got ", text_shape[2], ".");
                    OPENVINO_ASSERT(text_embeds.get_element_type() == ov::element::f32, "text_embeds must be f32.");
                    const size_t text_length = text_shape[1];
                    OPENVINO_ASSERT(offset + text_length <= sequence_length,
                                    "text chunk exceeds sequence_length. offset=",
                                    offset,
                                    ", text_length=",
                                    text_length,
                                    ", sequence_length=",
                                    sequence_length,
                                    ".");
                    std::copy_n(text_embeds.data<float>(),
                                text_embeds.get_size(),
                                inputs_embeds_ptr + offset * hidden_size);
                    return text_length;
                } else {
                    OPENVINO_ASSERT(chunk_value >= base_id,
                                    "visual_id must be greater than or equal to base_id. Got visual_id=",
                                    chunk_value,
                                    ", base_id=",
                                    base_id,
                                    ".");
                    const ov::Tensor& visual_embed = visual_embeds.at(chunk_value - base_id);
                    const auto& visual_shape = visual_embed.get_shape();
                    OPENVINO_ASSERT(visual_shape.size() == 3, "visual_embed must have rank 3, got ", visual_shape.size(), ".");
                    OPENVINO_ASSERT(visual_shape[0] == 1, "visual_embed batch size must be 1, got ", visual_shape[0], ".");
                    OPENVINO_ASSERT(visual_shape[2] == hidden_size, "visual_embed last dim must match hidden_size (", hidden_size, "), got ", visual_shape[2], ".");
                    OPENVINO_ASSERT(visual_embed.get_element_type() == ov::element::f32, "visual_embed must be f32.");
                    const size_t visual_length = visual_shape[1];
                    OPENVINO_ASSERT(offset + visual_length <= sequence_length,
                                    "visual chunk exceeds sequence_length. offset=",
                                    offset,
                                    ", visual_length=",
                                    visual_length,
                                    ", sequence_length=",
                                    sequence_length,
                                    ".");
                    std::copy_n(visual_embed.data<float>(),
                                visual_embed.get_size(),
                                inputs_embeds_ptr + offset * hidden_size);
                    return visual_length;
                }
            },
            chunk);
    }

    OPENVINO_ASSERT(offset == sequence_length,
                    "Merged inputs length mismatch. Expected ",
                    sequence_length,
                    ", got ",
                    offset,
                    ".");
    return inputs_embeds;
}

}  // namespace ov::genai::vlm_utils
