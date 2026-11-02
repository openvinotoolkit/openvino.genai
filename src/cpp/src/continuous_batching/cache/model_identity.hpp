// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

#include "content_hash.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"

namespace ov::genai {

/**
 * @brief Digest of a model's weights, as 32 hex characters.
 *
 * Block hashes of the prefix cache depend only on tokens, so a store that outlives the pipeline needs
 * something that tells two models apart. The digest covers every constant's element type, shape and bytes in
 * graph order, so the same weights give the same value across processes and builds.
 */
inline std::string compute_model_identity(const std::shared_ptr<const ov::Model>& model) {
    ContentHash state{0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL};
    const auto mix = [&state](const void* data, size_t size) {
        // Seeding each step with the previous digest chains the pieces in order.
        state = compute_content_hash(data, size, state.lo ^ (state.hi << 1));
    };

    for (const auto& op : model->get_ordered_ops()) {
        const auto constant = std::dynamic_pointer_cast<const ov::op::v0::Constant>(op);
        if (!constant) {
            continue;
        }
        std::ostringstream description;
        description << constant->get_element_type() << ' ' << constant->get_shape();
        const std::string text = description.str();
        mix(text.data(), text.size());
        if (constant->get_element_type() != ov::element::string) {
            mix(constant->get_data_ptr(), constant->get_byte_size());
        }
    }

    char hex[33];
    std::snprintf(hex, sizeof(hex), "%016llx%016llx", static_cast<unsigned long long>(state.hi),
                  static_cast<unsigned long long>(state.lo));
    return hex;
}

}  // namespace ov::genai
