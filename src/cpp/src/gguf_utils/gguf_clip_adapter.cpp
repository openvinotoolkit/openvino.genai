// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_clip_adapter.hpp"

#include <cstring>

namespace ov::genai {
ov::Tensor merge_gguf_llava_embeddings(const std::vector<bool>& image_slots,
                                       const ov::Tensor& text_embeddings,
                                       const std::vector<ov::Tensor>& image_embeddings) {
    const auto& shape = text_embeddings.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && image_slots.size() >= shape[1],
                    "GGUF LLaVA image slots must cover the language embedding suffix");
    const size_t prefix = image_slots.size() - shape[1];
    const size_t row_bytes = shape[2] * text_embeddings.get_element_type().size();
    ov::Tensor merged(text_embeddings.get_element_type(), shape);
    text_embeddings.copy_to(merged);
    auto* destination = static_cast<uint8_t*>(merged.data());
    size_t token = shape[1];
    for (auto image = image_embeddings.rbegin(); image != image_embeddings.rend(); ++image) {
        const auto& image_shape = image->get_shape();
        OPENVINO_ASSERT(image->get_element_type() == text_embeddings.get_element_type() && image_shape.size() == 3 &&
                            image_shape[0] == 1 && image_shape[2] == shape[2],
                        "GGUF LLaVA image embeddings must match the language embedding width and type");
        const auto* source = static_cast<const uint8_t*>(image->data());
        for (size_t row = image_shape[1]; row > 0; --row) {
            while (token && !image_slots[prefix + token - 1])
                --token;
            if (!token)
                return merged;
            --token;
            std::memcpy(destination + token * row_bytes, source + (row - 1) * row_bytes, row_bytes);
        }
    }
    return merged;
}

void validate_gguf_clip_language(const std::shared_ptr<ov::Model>& model) {
    const auto positions = model->input("position_ids");
    OPENVINO_ASSERT(positions.get_element_type() == ov::element::i64 && positions.get_partial_shape().rank() == 2,
                    "GGUF LLaVA and InternVL require rank-2 int64 language position_ids [batch, tokens]");
    for (const auto& input : model->inputs()) {
        const auto& names = input.get_names();
        OPENVINO_ASSERT(names.count("inputs_embeds") || names.count("attention_mask") || names.count("position_ids") ||
                            names.count("beam_idx"),
                        "GGUF LLaVA and InternVL do not support additional language input ",
                        input.get_any_name());
    }
}

}  // namespace ov::genai
