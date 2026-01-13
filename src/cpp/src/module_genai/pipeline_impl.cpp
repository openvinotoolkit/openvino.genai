// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_impl.hpp"

#include "module.hpp"
#include "modules/md_io.hpp"
#include "utils/yaml_utils.hpp"
#include "utils/profiler.hpp"

namespace ov {
namespace genai {

namespace module {

// config_yaml_path: yaml file.
ModulePipelineImpl::ModulePipelineImpl(const PipelineModulesDesc& pipeline_modules_desc,
                                       const PipelineDesc::PTR& pipeline_desc) {
    // Construct pipeline
    construct_pipeline(pipeline_modules_desc, m_modules, pipeline_desc);

    // Sort pipeline
    m_modules = sort_pipeline(m_modules);
}

ModulePipelineImpl::~ModulePipelineImpl() {}

// input all parameters in config.yaml
// "prompt": string
// "image": image ov::Tensor or std::vector<ov::Tensor>
// "video": video ov::Tensor
void ModulePipelineImpl::generate(ov::AnyMap& inputs, StreamerVariant streamer) {
    PROFILE(p, "generate");
    for (auto& module : m_modules) {
        PROFILE(pm, module->module_desc->name);
        if (module->is_input()) {
            std::dynamic_pointer_cast<ParameterModule>(module)->run(inputs);
        } else if (module->is_output()) {
            std::dynamic_pointer_cast<ResultModule>(module)->run(this->m_outputs);
        } else {
            module->run();
        }
    }
}

// execute generate asynchronously
void ModulePipelineImpl::generate_async(ov::AnyMap& inputs, StreamerVariant streamer) {
    PROFILE(p, "generate_async");
    using namespace oneapi::tbb::flow;

    if (_flow_nodes.empty()) {
        init_onetbb_threading();
    }

    m_current_inputs = &inputs;

    if (_starter) {
        _starter->try_put(continue_msg{});
    }
    _flow_graph.wait_for_all();
    
    m_current_inputs = nullptr;
}

void ModulePipelineImpl::init_onetbb_threading() {
    using namespace oneapi::tbb::flow;

    _flow_graph.reset();
    _flow_nodes.clear();
    _node_flow_map.clear();
    _starter.reset();

    _flow_nodes.reserve(m_modules.size());

    // register all nodes execution fucntion.
    for (auto& module : m_modules) {
        _flow_nodes.emplace_back(std::make_unique<FlowNode>(_flow_graph, [this, module](const continue_msg&) {
            PROFILE(pm, module->module_desc->name);
            if (module->is_input()) {
                if (this->m_current_inputs) {
                    std::dynamic_pointer_cast<ParameterModule>(module)->run(*(this->m_current_inputs));
                }
            } else if (module->is_output()) {
                std::dynamic_pointer_cast<ResultModule>(module)->run(this->m_outputs);
            } else {
                module->run();
            }
        }));
        _node_flow_map[module] = _flow_nodes.back().get();
    }

    // register all edges between nodes.
    for (auto& module : m_modules) {
        // loop all son modules
        for (auto& output : module->outputs) {
            for (auto& son_node : output.second.module_ptrs) {
                if (son_node->is_output()) {
                    continue;
                }

                auto it_son = _node_flow_map.find(son_node);
                if (it_son == _node_flow_map.end()) {
                    continue;
                }
                auto it_parent = _node_flow_map.find(module);
                if (it_parent == _node_flow_map.end()) {
                    continue;
                }
                make_edge(*it_parent->second, *it_son->second);
            }
        }       
    }

    // register starter node
    _starter = std::make_unique<broadcast_node<continue_msg>>(_flow_graph);
    for (auto& module : m_modules) {
        if (module->is_input()) {
            auto it = _node_flow_map.find(module);
            if (it != _node_flow_map.end()) {
                make_edge(*_starter, *it->second);
            }
        }
    }

    init_fake_edge();
}

// When async mode: 
// If node A and B have no data dependency, but still want them to be executed in order,
// we can add a fake edge between them.
void ModulePipelineImpl::init_fake_edge() {
    using namespace oneapi::tbb::flow;
    FlowNode* last_node = nullptr;
    for (auto& module : m_modules) {
        if (module->module_desc->thread_mode == ThreadMode::SYNC) {
            auto it = _node_flow_map.find(module);
            if (it != _node_flow_map.end()) {
                if (last_node != nullptr) {
                    make_edge(*last_node, *it->second);
                }
                last_node = it->second;
            }
        }
    }
}

ov::Any ModulePipelineImpl::get_output(const std::string& output_name) {
    return m_outputs[output_name];
}

void ModulePipelineImpl::start_chat(const std::string& system_message) {}

void ModulePipelineImpl::finish_chat() {}

bool ModulePipelineImpl::has_set_sync_mode() {
    bool is_set = false;
    for (auto& module : m_modules) {
        if (module->module_desc->thread_mode == ThreadMode::SYNC) {
            return true;
        }
    }
    return false;
}

ModulePipelineImpl::PTR init_sub_pipeline(const std::string& sub_pipeline_name, const PipelineDesc::PTR& pipeline_desc, IBaseModuleDesc::PTR module_desc) {
    bool found = false;
    ModulePipelineImpl::PTR sub_pipeline_impl = nullptr;

    for (auto& sub_module : pipeline_desc->sub_pipeline_descs) {
        if (sub_module.first == sub_pipeline_name) {
            sub_pipeline_impl = std::make_shared<ModulePipelineImpl>(sub_module.second, pipeline_desc);
            OPENVINO_ASSERT(
                sub_pipeline_impl != nullptr,
                "VAEDecoderTilingModule[" + module_desc->name + "]: Failed to create sub-pipeline instance");
            found = true;

            // Check if sub-pipeline has set sync mode, if set, change current module thread_mode to SYNC
            if (sub_pipeline_impl->has_set_sync_mode()) {
                module_desc->thread_mode = ThreadMode::SYNC;
                GENAI_INFO("VAEDecoderTilingModule[" + module_desc->name +
                           "]: Detected SYNC mode in sub-pipeline. This may impact performance.");
            }
            break;
        }
    }

    OPENVINO_ASSERT(found,
                    "VAEDecoderTilingModule[" + module_desc->name + "]: sub_pipeline_name '" + sub_pipeline_name +
                        "' not found in pipeline_desc");

    return sub_pipeline_impl;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
