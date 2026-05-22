// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace genai {

// Slice before last MatMul transformations.
// Each matcher handles a different graph tail pattern.
// All insert a Slice on MatMul's first input to keep only the last token.
// @param pa_based_model  When true, tokens are in dim 0 (PagedAttention layout).

class SliceLastMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::genai::SliceLastMatmul");
    explicit SliceLastMatmul(bool pa_based_model = false);
};

class SliceLastMatmulAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::genai::SliceLastMatmulAdd");
    explicit SliceLastMatmulAdd(bool pa_based_model = false);
};

class SliceLastMatmulTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::genai::SliceLastMatmulTranspose");
    explicit SliceLastMatmulTranspose(bool pa_based_model = false);
};

class SliceLastMatmulMultiply : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::genai::SliceLastMatmulMultiply");
    explicit SliceLastMatmulMultiply(bool pa_based_model = false);
};

// Gather before last MatMul transformations.
// Each matcher handles a different graph tail pattern.
// All insert a Gather on MatMul's first input and add a new Parameter
// ("sampled_tokens_indices") to the model.
// @param model           The model to add the new Parameter to.
// @param pa_based_model  When true, tokens are in dim 0 (PagedAttention layout).

class GatherLastMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::genai::GatherLastMatmul");
    explicit GatherLastMatmul(std::shared_ptr<ov::Model> model, bool pa_based_model = false);
};

class GatherLastMatmulAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::genai::GatherLastMatmulAdd");
    explicit GatherLastMatmulAdd(std::shared_ptr<ov::Model> model, bool pa_based_model = false);
};

class GatherLastMatmulTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::genai::GatherLastMatmulTranspose");
    explicit GatherLastMatmulTranspose(std::shared_ptr<ov::Model> model, bool pa_based_model = false);
};

class GatherLastMatmulMultiply : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::genai::GatherLastMatmulMultiply");
    explicit GatherLastMatmulMultiply(std::shared_ptr<ov::Model> model, bool pa_based_model = false);
};

}  // namespace genai
}  // namespace ov
