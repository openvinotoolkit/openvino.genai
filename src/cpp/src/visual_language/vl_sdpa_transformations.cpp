// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/paged_attention_transformations.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_vlsdpa.hpp"

namespace ov {
namespace genai {
namespace utils {

void apply_vl_sdpa_transformations(std::shared_ptr<ov::Model> model) {
    const ov::op::util::VariableVector& variables = model->get_variables();
    OPENVINO_ASSERT(variables.empty(), "Model is supposed to be stateless");

    ov::pass::SDPAToVLSDPA().run_on_model(model);

    model->validate_nodes_and_infer_types();
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
