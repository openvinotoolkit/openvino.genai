// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vl_sdpa_transformations.hpp"

#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {

void request_vl_sdpa_transformations(std::shared_ptr<ov::Model> model) {
    model->set_rt_info("QWenVL", "model_type_hint");
}

bool check_vl_sdpa_transformations(const ov::CompiledModel& compiled_model) {
    const std::vector<std::string> target_names {"cu_seq_lens", "cu_window_seqlens"};

    bool exists = false;
    for (auto &input : compiled_model.inputs()) {
        const auto& names = input.get_names();

        for (const auto& target : target_names) {
            exists |= (names.find(target) != names.end());
        }
    }

    return exists;
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
