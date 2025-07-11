// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/compiled_model.hpp"

namespace ov {
namespace genai {

namespace utils {

/** Applies transformations to the ov::Model to transform SDPA to VLSDPA.
 * @param model Pointer to the ov::Model representing one of the supported VLM architectures.
 */
void apply_vl_sdpa_transformations(std::shared_ptr<ov::Model> model);

bool check_vl_sdpa_transformations(ov::CompiledModel& compiled_model);

}  // namespace utils
}  // namespace genai
}  // namespace ov
