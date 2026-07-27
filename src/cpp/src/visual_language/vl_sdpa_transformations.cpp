// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vl_sdpa_transformations.hpp"

#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {

void request_vl_sdpa_transformations(std::shared_ptr<ov::Model> model) {
    model->set_rt_info("QWenVL", "model_type_hint");
}

bool has_vl_sdpa_input(const ov::CompiledModel& compiled_model, const std::string& input_name) {
    for (const auto& input : compiled_model.inputs()) {
        const auto& names = input.get_names();
        if (names.find(input_name) != names.end()) {
            return true;
        }
    }
    return false;
}

bool check_vl_sdpa_transformations(const ov::CompiledModel& compiled_model) {
    return has_vl_sdpa_input(compiled_model, "cu_seq_lens");
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
