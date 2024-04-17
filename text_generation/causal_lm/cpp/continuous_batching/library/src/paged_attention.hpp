// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/op.hpp"

class PagedAttention : public ov::op::Op {
public:
    OPENVINO_OP("PagedAttentionExtension");

    PagedAttention() = default;

    PagedAttention(const ov::OutputVector& inputs);

    PagedAttention(const ov::Output<ov::Node>& query,
                   const ov::Output<ov::Node>& key,
                   const ov::Output<ov::Node>& value,
                   const ov::Output<ov::Node>& key_cache,
                   const ov::Output<ov::Node>& value_cache,
                   // start of arguments from InputMetadata
                   const ov::Output<ov::Node>& is_prompt,
                   const ov::Output<ov::Node>& slot_mapping,
                //    const ov::Output<ov::Node>& prompt_lens,
                //    const ov::Output<ov::Node>& max_seq_len,
                //    const ov::Output<ov::Node>& start_loc,
                   const ov::Output<ov::Node>& max_context_len,
                   const ov::Output<ov::Node>& context_lens,
                   const ov::Output<ov::Node>& block_tables,
                //    const ov::Output<ov::Node>& use_cuda_graph,
                //    const ov::Output<ov::Node>& attn_bias
                   // end of arguments from InputMetadata
                   const ov::Output<ov::Node>& scale,
                   const ov::Output<ov::Node>& alibi_slopes,
                   const ov::Output<ov::Node>& sliding_window);

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
