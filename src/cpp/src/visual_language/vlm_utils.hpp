// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <regex>
#include <string>
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
template <typename InferTextEmbeddings>
ov::Tensor build_inputs_embeds_from_text_and_visual_chunks(
    const std::vector<std::variant<ov::Tensor, size_t>>& chunks,
    InferTextEmbeddings&& infer_text_embeddings,
    const std::vector<ov::Tensor>& visual_embeds,
    size_t base_id,
    size_t hidden_size) {
    OPENVINO_ASSERT(hidden_size > 0, "hidden_size must be greater than 0.");

    size_t merged_length = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        if (std::holds_alternative<ov::Tensor>(chunk)) {
            merged_length += std::get<ov::Tensor>(chunk).get_shape().at(1);
            continue;
        }

        const size_t visual_id = std::get<size_t>(chunk);
        OPENVINO_ASSERT(visual_id >= base_id,
                        "visual_id must be greater than or equal to base_id. Got visual_id=",
                        visual_id,
                        ", base_id=",
                        base_id,
                        ".");
        merged_length += visual_embeds.at(visual_id - base_id).get_shape().at(1);
    }

    ov::Tensor inputs_embeds{ov::element::f32, {1, merged_length, hidden_size}};
    float* inputs_embeds_ptr = inputs_embeds.data<float>();
    size_t offset = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        if (std::holds_alternative<ov::Tensor>(chunk)) {
            const ov::Tensor& token_chunk = std::get<ov::Tensor>(chunk);
            const ov::Tensor text_embeds = infer_text_embeddings(token_chunk);
            const size_t text_length = text_embeds.get_shape().at(1);
            std::copy_n(text_embeds.data<float>(),
                        text_embeds.get_size(),
                        inputs_embeds_ptr + offset * hidden_size);
            offset += text_length;
            continue;
        }

        const size_t visual_id = std::get<size_t>(chunk);
        OPENVINO_ASSERT(visual_id >= base_id,
                        "visual_id must be greater than or equal to base_id. Got visual_id=",
                        visual_id,
                        ", base_id=",
                        base_id,
                        ".");
        const ov::Tensor& visual_embed = visual_embeds.at(visual_id - base_id);
        const size_t visual_length = visual_embed.get_shape().at(1);
        std::copy_n(visual_embed.data<float>(),
                    visual_embed.get_size(),
                    inputs_embeds_ptr + offset * hidden_size);
        offset += visual_length;
    }

    OPENVINO_ASSERT(offset == merged_length,
                    "Merged inputs length mismatch. Expected ",
                    merged_length,
                    ", got ",
                    offset,
                    ".");
    return inputs_embeds;
}

}  // namespace ov::genai::vlm_utils
