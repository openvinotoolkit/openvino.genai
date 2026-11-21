// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_clip_adapter.hpp"

namespace ov::genai {
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
