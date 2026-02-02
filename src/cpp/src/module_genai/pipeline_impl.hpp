// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <oneapi/tbb/flow_graph.h>

#include <optional>

#include "module.hpp"
#include "utils/yaml_utils.hpp"

namespace ov {
namespace genai {

namespace module {
class ModulePipelineImpl {
private:
    PipelineModuleInstance m_modules;

public:
    // config_yaml_path: yaml file.
    ModulePipelineImpl(const PipelineModulesDesc& pipeline_modules_desc, const PipelineDesc::PTR& pipeline_desc);

    ~ModulePipelineImpl();

    using PTR = std::shared_ptr<ModulePipelineImpl>;

    // input all parameters in config.yaml
    // "prompt": string
    // "image": image ov::Tensor or std::vector<ov::Tensor>
    // "video": video ov::Tensor
    void generate(ov::AnyMap& inputs, StreamerVariant streamer = std::monostate());

    // execute generate asynchronously
    void generate_async(ov::AnyMap& inputs, StreamerVariant streamer = std::monostate());

    ov::Any get_output(const std::string& output_name);
    ov::Any get_output();

    void start_chat(const std::string& system_message = {});

    void finish_chat();

    bool has_set_sync_mode();

private:
    std::map<std::string, ov::Any> m_outputs;

    // Only initialize oneTBB threading for ansync generate
    void init_onetbb_threading();
    void init_fake_edge();
    oneapi::tbb::flow::graph _flow_graph;  // internal flow graph for async execution
    using FlowNode = oneapi::tbb::flow::continue_node<oneapi::tbb::flow::continue_msg>;
    std::vector<std::unique_ptr<FlowNode>> _flow_nodes;     // cached flow nodes
    std::unordered_map<IBaseModule::PTR, FlowNode*> _node_flow_map;  // map from logical node to flow node
    std::unique_ptr<oneapi::tbb::flow::broadcast_node<oneapi::tbb::flow::continue_msg>> _starter;
    ov::AnyMap* m_current_inputs = nullptr;
};

// Initialize sub-pipeline for modules like VAEDecoderTilingModule
// sub_pipeline_name: name of the sub-pipeline
// pipeline_desc: the overall pipeline description containing sub-pipelines
// module_desc: the module description of the current module
ModulePipelineImpl::PTR init_sub_pipeline(const std::string& sub_pipeline_name,
                                          const PipelineDesc::PTR& pipeline_desc,
                                          IBaseModuleDesc::PTR module_desc);

}  // namespace module
}  // namespace genai
}  // namespace ov
