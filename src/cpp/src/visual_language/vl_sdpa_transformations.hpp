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

/** Requests transforming SDPA ov::Model to VLSDPA. It's up to a plugin to apply the transformation.
 * @param model Pointer to the ov::Model representing one of the supported VLM architectures.
 */
void request_vl_sdpa_transformations(std::shared_ptr<ov::Model> model);

bool check_vl_sdpa_transformations(const ov::CompiledModel& compiled_model);

}  // namespace utils
}  // namespace genai
}  // namespace ov
