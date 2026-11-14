// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cgraph_to_ov.hpp"

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ggml-decoder.h"
#include "ggml-impl.h"
#include "llama-context.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "openvino/frontend.h"
#include "openvino/input_model.h"

namespace ov::genai::gguf_utils {

namespace {

struct LlamaRuntimeResources {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
};

class ScopedLlamaLogFilter {
public:
    ScopedLlamaLogFilter() {
        llama_log_get(&m_original_callback, &m_original_user_data);
        llama_log_set(
            [](ggml_log_level level, const char* text, void* user_data) {
                auto* filter = static_cast<ScopedLlamaLogFilter*>(user_data);
                if (level >= GGML_LOG_LEVEL_WARN) {
                    filter->m_original_callback(level, text, filter->m_original_user_data);
                }
            },
            this);
    }

    ~ScopedLlamaLogFilter() {
        llama_log_set(m_original_callback, m_original_user_data);
    }

    ScopedLlamaLogFilter(const ScopedLlamaLogFilter&) = delete;
    ScopedLlamaLogFilter& operator=(const ScopedLlamaLogFilter&) = delete;

private:
    ggml_log_callback m_original_callback = nullptr;
    void* m_original_user_data = nullptr;
};

// The upstream GGML decoder reads buffer usage while discovering graph inputs.
// Borrow a valid buffer only during its constructor, then restore the graph tensors.
class ScopedGraphBufferFallback {
public:
    explicit ScopedGraphBufferFallback(ggml_cgraph* cgraph) {
        std::vector<ggml_tensor*> tensors;
        std::unordered_set<ggml_tensor*> seen;
        auto collect = [&](ggml_tensor* tensor) {
            if (tensor && seen.insert(tensor).second) {
                tensors.push_back(tensor);
            }
        };

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            auto* node = cgraph->nodes[i];
            collect(node);
            collect(node->view_src);
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                collect(node->src[j]);
                if (node->src[j]) {
                    collect(node->src[j]->view_src);
                }
            }
        }
        for (int i = 0; i < cgraph->n_leafs; ++i) {
            collect(cgraph->leafs[i]);
            if (cgraph->leafs[i]) {
                collect(cgraph->leafs[i]->view_src);
            }
        }

        ggml_backend_buffer* fallback_buffer = nullptr;
        for (const auto* tensor : tensors) {
            if (tensor->buffer && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                fallback_buffer = tensor->buffer;
                break;
            }
        }
        OPENVINO_ASSERT(fallback_buffer != nullptr,
                        "llama.cpp graph has no weight buffer to use while inspecting bufferless inputs");

        m_original_buffers.reserve(tensors.size());
        for (auto* tensor : tensors) {
            if (tensor->buffer) {
                continue;
            }

            auto* buffer_source = tensor->view_src;
            while (buffer_source && !buffer_source->buffer) {
                buffer_source = buffer_source->view_src;
            }
            auto* temporary_buffer = buffer_source ? buffer_source->buffer : fallback_buffer;
            m_original_buffers.emplace_back(tensor, tensor->buffer);
            tensor->buffer = temporary_buffer;
        }
    }

    ScopedGraphBufferFallback(const ScopedGraphBufferFallback&) = delete;
    ScopedGraphBufferFallback& operator=(const ScopedGraphBufferFallback&) = delete;

    ~ScopedGraphBufferFallback() {
        for (auto it = m_original_buffers.rbegin(); it != m_original_buffers.rend(); ++it) {
            it->first->buffer = it->second;
        }
    }

private:
    std::vector<std::pair<ggml_tensor*, ggml_backend_buffer*>> m_original_buffers;
};

}  // namespace

std::shared_ptr<ov::Model> convert_gguf_with_llama_cpp(const std::string& model_path) {
    static std::mutex llama_log_mutex;
    std::lock_guard<std::mutex> llama_log_lock(llama_log_mutex);

    static std::once_flag backend_initialized;
    std::call_once(backend_initialized, [] {
        llama_backend_init();
    });

    ScopedLlamaLogFilter llama_log_filter;
    auto resources = std::make_shared<LlamaRuntimeResources>();
    auto model_params = llama_model_default_params();
    auto* cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    OPENVINO_ASSERT(cpu_device != nullptr, "llama.cpp CPU backend is unavailable for GGUF graph extraction");
    ggml_backend_dev_t devices[] = {cpu_device, nullptr};
    model_params.devices = devices;
    model_params.n_gpu_layers = 0;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    resources->model.reset(llama_model_load_from_file(model_path.c_str(), model_params));
    if (!resources->model) {
        OPENVINO_THROW("llama.cpp failed to load GGUF model: ", model_path);
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = 0;
    context_params.n_batch = 1;
    context_params.n_ubatch = 1;
    context_params.n_seq_max = 1;
    context_params.n_outputs_max = 1;
    context_params.offload_kqv = false;
    resources->context.reset(llama_init_from_model(resources->model.get(), context_params));
    if (!resources->context) {
        OPENVINO_THROW("llama.cpp failed to create a context for GGUF graph construction: ", model_path);
    }

    auto* internal_context = reinterpret_cast<llama_context*>(resources->context.get());
    auto memory = internal_context->get_memory();
    auto memory_context = memory->init_full();
    if (!memory_context) {
        OPENVINO_THROW("llama.cpp failed to initialize model memory for GGUF graph construction: ", model_path);
    }

    auto* cgraph = internal_context->graph_reserve(1, 1, 1, memory_context.get());
    if (!cgraph) {
        OPENVINO_THROW("llama.cpp failed to construct a GGML graph for GGUF model: ", model_path);
    }

    std::vector<ggml_tensor*> compute_nodes;
    compute_nodes.reserve(cgraph->n_nodes);
    std::copy_if(cgraph->nodes,
                 cgraph->nodes + cgraph->n_nodes,
                 std::back_inserter(compute_nodes),
                 [](const ggml_tensor* node) {
                     return node->op == GGML_OP_NONE || (node->flags & GGML_TENSOR_FLAG_COMPUTE);
                 });
    OPENVINO_ASSERT(!compute_nodes.empty(), "llama.cpp graph has no nodes selected for computation");
    auto compute_graph = *cgraph;
    compute_graph.nodes = compute_nodes.data();
    compute_graph.n_nodes = static_cast<int>(compute_nodes.size());
    cgraph = &compute_graph;

    auto [model_params_for_graph, compute_params] = GgmlOvDecoder::compute_llm_params(cgraph, false);

    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    std::set<std::string> weight_names;
    for (int node_index = 0; node_index < cgraph->n_nodes; ++node_index) {
        auto* node = cgraph->nodes[node_index];
        for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
            auto* source = node->src[source_index];
            if (source == nullptr) {
                continue;
            }

            auto* base = source->view_src ? source->view_src : source;
            if (base->buffer == nullptr ||
                (base->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS && !ggml_is_quantized(base->type))) {
                continue;
            }

            auto name = GgmlOvDecoder::get_tensor_name(cgraph, base);
            if (GgmlOvDecoder::is_rope_freqs_weight(base, node)) {
                name = "rope_freqs.weight";
            }
            if (weight_names.insert(name).second) {
                model_weights.emplace(name, GgmlOvDecoder::create_weight_node(base));
            }
        }
    }
    std::shared_ptr<ov::Model> ov_model;
    {
        ScopedGraphBufferFallback temporary_input_buffers(cgraph);
        auto decoder = std::make_shared<GgmlOvDecoder>(
            cgraph,
            model_params_for_graph,
            compute_params,
            model_weights,
            false);
        auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
        ov_model = ov::frontend::ggml::FrontEnd::convert(input_model);
    }
    ov_model->set_rt_info(resources, "llama_cpp_resources");
    return ov_model;
}

}  // namespace ov::genai::gguf_utils
