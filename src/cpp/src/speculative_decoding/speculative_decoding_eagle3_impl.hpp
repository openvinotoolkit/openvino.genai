// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "speculative_decoding_impl.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"

namespace ov::genai {
class ContinuousBatchingPipeline::Eagle3DecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
public:
    template<class Impl>
    friend std::vector<EncodedGenerationResult> generate_common(
        Impl*,
        const std::vector<ov::Tensor>&,
        const std::vector<GenerationConfig>&,
        const StreamerVariant&,
        std::optional<std::vector<ov::Tensor>>,
        GenerateStrategy&);
    Eagle3DecodingImpl(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc, const std::vector<int>& hidden_layers_to_abstract);

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<ov::Tensor>>& token_type_ids = std::nullopt,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;
protected:
    void update_eagle_pipeline_params(std::shared_ptr<ov::op::v0::Constant>& d2t_tensor);
    ov::Tensor create_draft_input_ids(const ov::Tensor& original_input_ids);
    std::vector<int> m_hidden_layers_to_abstract;
};

using NodePtr = std::shared_ptr<ov::Node>;
using namespace ov::op;

class EagleBaseTransform : public ov::pass::MatcherPass {
public:
    using NodePtr = std::shared_ptr<ov::Node>;
    OPENVINO_MATCHER_PASS_RTTI("EagleBaseTransform");
    EagleBaseTransform(std::vector<std::shared_ptr<ov::op::v0::Result>>& results);

    ~EagleBaseTransform() = default;

private:
    bool apply(NodePtr node, std::vector<std::shared_ptr<ov::op::v0::Result>>& results);
    size_t applied = 0;
    std::shared_ptr<ov::Node> find_last_residual_node(const std::shared_ptr<ov::Node>& start_node);
    std::shared_ptr<ov::Node> find_last_residual_node(const std::shared_ptr<ov::Node>& start_node, 
                                                               std::set<ov::Node*>& visited_nodes);
};
class EagleInputTransform : public ov::pass::MatcherPass { // eagle3 specific for draft model
public:
    using NodePtr = std::shared_ptr<ov::Node>;
    OPENVINO_MATCHER_PASS_RTTI("EagleInputTransform");
    EagleInputTransform(std::vector<std::shared_ptr<ov::op::v0::Parameter>>& params);

    ~EagleInputTransform() = default;

private:
    bool apply(NodePtr node, std::vector<std::shared_ptr<ov::op::v0::Parameter>>& params);
    size_t applied = 0;
};
class Eagle3Transform : public ov::pass::MatcherPass {
public:
    using NodePtr = std::shared_ptr<ov::Node>;
    OPENVINO_MATCHER_PASS_RTTI("Eagle3Transform");
    Eagle3Transform(const std::vector<int>& layers, std::vector<Output<Node>>& hidden_state_outputs);

    ~Eagle3Transform() = default;

private:
    std::vector<int> m_layers; // layers to be abstracted
};

class EagleModelTransform : public ov::pass::ModelPass {
public:
    EagleModelTransform(const std::vector<int>& layer_ids);
    bool run_on_model(const std::shared_ptr<Model>& model) override;

private:
    const std::vector<int> m_layer_ids;
    std::vector<std::shared_ptr<ov::op::v0::Result>> m_new_results;
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> m_new_parameters;
    std::vector<Output<Node>> m_hidden_layer_outputs;
};
}
