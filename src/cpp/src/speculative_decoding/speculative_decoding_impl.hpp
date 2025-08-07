// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "speculative_decoding/continuous_batching_for_speculative_decoding_impl.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"

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

struct ModelDesc {
    std::string device;
    ov::genai::SchedulerConfig scheduler_config;
    ov::AnyMap properties;
    ov::genai::GenerationConfig generation_config;
    std::shared_ptr<ov::Model> model = nullptr;
    ov::genai::Tokenizer tokenizer;

    ModelDesc(const std::shared_ptr<ov::Model>& model,
              const ov::genai::Tokenizer& tokenizer,
              const std::string& device = {},
              const ov::AnyMap& properties = {},
              const ov::genai::SchedulerConfig& scheduler_config = {},
              const ov::genai::GenerationConfig& generation_config = {}) :
        model(model),
        tokenizer(tokenizer),
        device(device),
        properties(properties),
        scheduler_config(scheduler_config),
        generation_config(generation_config) {}
    
    ModelDesc() = default;
};

class ContinuousBatchingPipeline::SpeculativeDecodingImpl : public ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    std::shared_ptr<ContinuousBatchingForSpeculativeDecodingImpl> m_main_pipeline, m_draft_pipeline;
    // Metrics
    SpeculativeDecodingMetrics m_sd_metrics;
    ov::genai::SDPerModelsPerfMetrics m_perf_metrics;

    // Mutex protecting access to m_draft_generations, so add_request and step methods can be called from different threads
    std::mutex m_draft_generations_mutex;
    std::map<uint64_t, GenerationHandle> m_draft_generations;

    void drop_requests();
    bool is_requests_empty();
    std::vector<SequenceGroup::Ptr> get_awaiting_requests();
    
public:
    SpeculativeDecodingImpl(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc);

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 ov::genai::GenerationConfig sampling_params) override;
    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 ov::genai::GenerationConfig sampling_params) override;

    bool has_non_finished_requests() override;

    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer) override;

    SpeculativeDecodingMetrics get_speculative_decoding_metrics();
};

class ContinuousBatchingPipeline::EagleDecodingImpl : public ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    std::shared_ptr<ContinuousBatchingForEagleDecodingImpl> m_main_pipeline, m_draft_pipeline; // bell: see if we can reuse this class impl for eagle pipelines
    // Metrics
    SpeculativeDecodingMetrics m_sd_metrics;
    ov::genai::SDPerModelsPerfMetrics m_perf_metrics;
    ov::Tensor hiddenstates_tensor; // Tensor to store hidden states for draft model
    // Mutex protecting access to m_draft_generations, so add_request and step methods can be called from different threads
    std::mutex m_draft_generations_mutex;
    std::map<uint64_t, GenerationHandle> m_draft_generations;

    void drop_requests();
    void initialize_tree();
    bool is_requests_empty();
    std::vector<SequenceGroup::Ptr> get_awaiting_requests();
    ov::Tensor create_draft_input_ids(const ov::Tensor& original_input_ids);
    ov::Tensor update_main_input_ids(const ov::Tensor& original_input_ids);
    std::string m_eagle_version;
    
public:
    EagleDecodingImpl(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc, const std::string& eagle_version);

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 ov::genai::GenerationConfig sampling_params) override;
    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 ov::genai::GenerationConfig sampling_params) override;

    bool has_non_finished_requests() override;

    void fill_hidden_states(const ov::Tensor& hidden_states) {
        hiddenstates_tensor = hidden_states;
    }
    void set_d2t_for_draft_decoding(std::shared_ptr<ov::op::v0::Constant>& d2t_tensor) {
        m_draft_pipeline->set_d2t_for_draft_decoding(d2t_tensor);
    };
    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer) override;

    SpeculativeDecodingMetrics get_speculative_decoding_metrics();
};

using NodePtr = std::shared_ptr<ov::Node>;
using namespace ov::op;

class EagleBaseTransform : public ov::pass::MatcherPass {
public:
    using NodePtr = std::shared_ptr<ov::Node>;
    OPENVINO_MATCHER_PASS_RTTI("EagleBaseTransform");
    EagleBaseTransform(const std::vector<int>& layers, std::vector<std::shared_ptr<ov::op::v0::Result>>& results);

    ~EagleBaseTransform() = default;

private:
    bool apply(NodePtr node, std::vector<std::shared_ptr<ov::op::v0::Result>>& results);
    size_t applied = 0;
    std::shared_ptr<ov::Node> find_last_hidden_node(const std::shared_ptr<ov::Node>& start_node);
    std::shared_ptr<ov::Node> find_last_hidden_node(const std::shared_ptr<ov::Node>& start_node, 
                                                   std::set<ov::Node*>& visited_nodes);
    std::vector<int> m_layers; // layers to be abstracted
};

class Eagle3Transform : public ov::pass::MatcherPass {
public:
    using NodePtr = std::shared_ptr<ov::Node>;
    OPENVINO_MATCHER_PASS_RTTI("Eagle3Transform");
    Eagle3Transform(const std::vector<int>& layers, std::vector<Output<Node>>& hidden_state_outputs);

    ~Eagle3Transform() = default;

private:
    bool apply(NodePtr node, std::vector<Output<Node>>& hidden_state_outputs);
    size_t applied = 0;
    std::vector<int> m_layers; // layers to be abstracted
};

class EagleModelTransform : public ov::pass::ModelPass {
public:
    EagleModelTransform(const std::vector<int>& layer_ids);
    bool run_on_model(const std::shared_ptr<Model>& model) override;

private:
    const std::vector<int> m_layer_ids;
    std::vector<std::shared_ptr<ov::op::v0::Result>> m_new_results;
    std::vector<Output<Node>> m_hidden_layer_outputs;
};
}
