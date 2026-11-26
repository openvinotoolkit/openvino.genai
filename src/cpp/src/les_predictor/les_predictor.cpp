// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "les_predictor/les_predictor.hpp"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <regex>
#include <unordered_set>

#include "logger.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/util/topk_base.hpp"
#include "openvino/runtime/tensor.hpp"

extern "C" {
#include "safetensors.h"
}

namespace ov {
namespace genai {

namespace {

using namespace ov::op;
using NodeOutput = ov::Output<ov::Node>;

// The only input mode seen so far: `weight` takes the current token, `previous_weight` the previous token.
constexpr int64_t INPUT_MODE_CURRENT_PREVIOUS = 2;

ov::element::Type to_ov_type(int dtype) {
    switch (dtype) {
    case SAFETENSORS_F32:
        return ov::element::f32;
    case SAFETENSORS_F16:
        return ov::element::f16;
    case SAFETENSORS_BF16:
        return ov::element::bf16;
    case SAFETENSORS_I64:
        return ov::element::i64;
    case SAFETENSORS_I32:
        return ov::element::i32;
    default:
        OPENVINO_THROW("LES predictor: unsupported safetensors dtype ", dtype);
    }
}

std::map<std::string, std::shared_ptr<v0::Constant>> read_safetensors(const std::filesystem::path& path) {
    const ov::Tensor file = ov::read_tensor_data(path);  // read-only mapping of the file
    safetensors_File st{};
    // safetensors_file_init only reads the buffer, it needs a non-const pointer
    auto* data = const_cast<char*>(file.data<const char>());
    const char* error = safetensors_file_init(data, file.get_byte_size(), &st);
    OPENVINO_ASSERT(error == nullptr, "LES predictor: cannot parse ", path, " as a safetensors file: ", error);

    std::map<std::string, std::shared_ptr<v0::Constant>> tensors;
    for (int i = 0; i < st.num_tensors; ++i) {
        const safetensors_TensorDescriptor& t = st.tensors[i];
        std::string name(t.name.ptr, t.name.ptr + t.name.len);
        ov::Shape shape(t.shape, t.shape + t.n_dimensions);
        tensors[name] = std::make_shared<v0::Constant>(to_ov_type(t.dtype), shape, t.ptr);  // copies the data
    }
    std::free(st.tensors);
    std::free(st.metadata);
    return tensors;
}

NodeOutput as_f32(const NodeOutput& x) {
    if (x.get_element_type() == ov::element::f32) {
        return x;
    }
    return std::make_shared<v0::Convert>(x, ov::element::f32);
}

// Output of `model.layers.<N>.post_attention_layernorm`, the router input of MoE layer N, found by the friendly names
// that the PyTorch frontend gives to the nodes of a module: the last node of the module scope whose output is used
// outside of the scope.
std::map<size_t, NodeOutput> find_router_inputs(const std::shared_ptr<ov::Model>& model) {
    static const std::regex layer_scope(R"(layers\.(\d+)\.post_attention_layernorm([/.]|$))");
    std::map<size_t, std::vector<std::shared_ptr<ov::Node>>> scopes;
    for (const auto& op : model->get_ordered_ops()) {
        const std::string& name = op->get_friendly_name();
        std::smatch match;
        if (std::regex_search(name, match, layer_scope)) {
            scopes[std::stoul(match[1].str())].push_back(op);
        }
    }

    std::map<size_t, NodeOutput> router_inputs;
    for (const auto& [layer, nodes] : scopes) {
        std::unordered_set<const ov::Node*> in_scope;
        for (const auto& node : nodes) {
            in_scope.insert(node.get());
        }
        std::shared_ptr<ov::Node> scope_output;
        for (const auto& node : nodes) {  // topological order: the last match is the norm output
            if (node->get_output_size() != 1) {
                continue;
            }
            const auto& shape = node->get_output_partial_shape(0);
            if (shape.rank().is_dynamic() || shape.rank().get_length() != 3) {
                continue;
            }
            for (const auto& target : node->output(0).get_target_inputs()) {
                if (!in_scope.count(target.get_node())) {
                    scope_output = node;
                    break;
                }
            }
        }
        if (scope_output) {
            router_inputs[layer] = scope_output->output(0);
        }
    }
    return router_inputs;
}

// TopK of the router that consumes the router input (MatMul -> Sigmoid -> Add -> TopK): the closest TopK downstream.
std::shared_ptr<ov::op::util::TopKBase> find_router_topk(const NodeOutput& router_input) {
    constexpr size_t max_depth = 6;
    std::deque<std::pair<ov::Node*, size_t>> queue;
    std::unordered_set<ov::Node*> visited;
    for (const auto& target : router_input.get_target_inputs()) {
        queue.emplace_back(target.get_node(), 1);
    }
    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop_front();
        if (!visited.insert(node).second) {
            continue;
        }
        if (auto topk = ov::as_type<ov::op::util::TopKBase>(node)) {
            return std::static_pointer_cast<ov::op::util::TopKBase>(topk->shared_from_this());
        }
        if (depth == max_depth) {
            continue;
        }
        for (const auto& output : node->outputs()) {
            for (const auto& target : output.get_target_inputs()) {
                queue.emplace_back(target.get_node(), depth + 1);
            }
        }
    }
    return nullptr;
}

// top_k(sigmoid(W cur + W' prev) + b): the indices of the predicted experts, i32. The device plugin reads the weights
// of this exact form: MatMul(input, weight), Add, Sigmoid, Add(bias), TopK.
NodeOutput build_head(const LESPredictorImpl::Head& head, const NodeOutput& cur, const NodeOutput& prev, size_t top_k) {
    auto logits = std::make_shared<v1::Add>(std::make_shared<v0::MatMul>(cur, as_f32(head.weight), false, true),
                                            std::make_shared<v0::MatMul>(prev, as_f32(head.previous_weight), false, true));
    auto scores = std::make_shared<v0::Sigmoid>(logits);
    // The selection bias is large (170..280 for MiniCPM5-MoE) and varies by less than 1 within a layer: subtract its
    // maximum. It does not change which experts are selected, and keeps the scores distinguishable in f16.
    std::vector<float> bias = head.selection_bias->cast_vector<float>();
    const float max_bias = *std::max_element(bias.begin(), bias.end());
    for (auto& b : bias) {
        b -= max_bias;
    }
    auto selection = std::make_shared<v1::Add>(scores, v0::Constant::create(ov::element::f32, {bias.size()}, bias));
    auto topk = std::make_shared<v11::TopK>(selection,
                                            v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(top_k)}),
                                            -1,
                                            ov::op::TopKMode::MAX,
                                            ov::op::TopKSortType::SORT_VALUES,
                                            ov::element::i32);
    return topk->output(1);
}

}  // namespace

LESPredictorImpl::LESPredictorImpl(const std::filesystem::path& path) {
    OPENVINO_ASSERT(path.extension() == ".safetensors", "LES predictor: expected a .safetensors file, got ", path);
    static const std::regex tensor_name(R"(expert_predictors\.(\d+)\.(weight|previous_weight|selection_bias|input_mode_code))");
    std::map<size_t, std::map<std::string, std::shared_ptr<v0::Constant>>> parts;
    for (const auto& [name, constant] : read_safetensors(path)) {
        std::smatch match;
        if (std::regex_match(name, match, tensor_name)) {
            parts[std::stoul(match[1].str())][match[2].str()] = constant;
        }
    }
    OPENVINO_ASSERT(!parts.empty(), "LES predictor: no expert_predictors.<i>.* tensors in ", path);

    for (auto& [index, p] : parts) {
        OPENVINO_ASSERT(index >= 1, "LES predictor: expert_predictors.<i> uses 1-based layer numbers, got ", index);
        OPENVINO_ASSERT(p.count("weight") && p.count("previous_weight") && p.count("selection_bias"),
                        "LES predictor: expert_predictors.", index,
                        " needs weight, previous_weight and selection_bias");
        Head head;
        head.weight = p["weight"];
        head.previous_weight = p["previous_weight"];
        head.selection_bias = p["selection_bias"];
        head.input_mode = p.count("input_mode_code") ? p["input_mode_code"]->cast_vector<int64_t>().at(0)
                                                     : INPUT_MODE_CURRENT_PREVIOUS;
        OPENVINO_ASSERT(head.input_mode == INPUT_MODE_CURRENT_PREVIOUS,
                        "LES predictor: expert_predictors.", index, " has input_mode_code ", head.input_mode,
                        ", only ", INPUT_MODE_CURRENT_PREVIOUS, " (current + previous token) is supported");

        const auto& w_shape = head.weight->get_shape();
        OPENVINO_ASSERT(w_shape.size() == 2, "LES predictor: weight must be [num_experts, hidden_size]");
        if (num_experts == 0) {
            num_experts = w_shape[0];
            hidden_size = w_shape[1];
        }
        OPENVINO_ASSERT(w_shape[0] == num_experts && w_shape[1] == hidden_size,
                        "LES predictor: all heads must have the same shape, expert_predictors.", index, " has ",
                        w_shape);
        OPENVINO_ASSERT(head.previous_weight->get_shape() == w_shape,
                        "LES predictor: previous_weight must have the shape of weight");
        OPENVINO_ASSERT(head.selection_bias->get_shape() == ov::Shape{num_experts},
                        "LES predictor: selection_bias must be [num_experts]");
        heads[index - 1] = std::move(head);
    }
}

LESPredictor::LESPredictor(const std::filesystem::path& path) : m_pimpl(std::make_shared<LESPredictorImpl>(path)) {}

namespace les {

bool extract_from_properties(utils::SharedOptional<const ov::AnyMap>& properties,
                             std::optional<LESPredictor>& predictor) {
    const std::string name = ov::genai::les_predictor.name();
    auto it = properties->find(name);
    if (it == properties->end()) {
        return false;
    }
    // a path: std::filesystem::path from Python pathlib.Path, std::string from Python str or a JSON config
    if (it->second.is<LESPredictor>()) {
        predictor = it->second.as<LESPredictor>();
    } else if (it->second.is<std::filesystem::path>()) {
        predictor = LESPredictor(it->second.as<std::filesystem::path>());
    } else {
        OPENVINO_ASSERT(it->second.is<std::string>(), "LES predictor: the ", name,
                        " property must be a LESPredictor or the path to the predictor file");
        predictor = LESPredictor(it->second.as<std::string>());
    }
    properties.fork().erase(name);
    return true;
}

PredictorInfo analyze(const std::shared_ptr<ov::Model>& model, const LESPredictor& predictor) {
    OPENVINO_ASSERT(predictor, "LES predictor: the predictor is empty");
    const LESPredictorImpl& impl = *predictor.get_impl();
    const auto router_inputs = find_router_inputs(model);
    const auto hidden = static_cast<int64_t>(impl.hidden_size);

    PredictorInfo info;
    info.hidden_size = impl.hidden_size;
    for (const auto& [layer, head] : impl.heads) {
        OPENVINO_ASSERT(router_inputs.count(layer - 1) && router_inputs.count(layer),
                        "LES predictor: cannot find the post_attention_layernorm outputs of decoder layers ",
                        layer - 1, " and ", layer, " in the model");
        const auto& shape = router_inputs.at(layer).get_partial_shape();
        OPENVINO_ASSERT(shape[2].compatible(hidden), "LES predictor: the model hidden size ", shape[2],
                        " differs from the predictor hidden size ", hidden);
        auto topk = find_router_topk(router_inputs.at(layer));
        OPENVINO_ASSERT(topk, "LES predictor: decoder layer ", layer, " has no MoE router (TopK) in the model");
        const size_t router_k = topk->get_k();
        OPENVINO_ASSERT(router_k > 0, "LES predictor: the router TopK of layer ", layer, " has no constant k");
        OPENVINO_ASSERT(info.top_k == 0 || info.top_k == router_k,
                        "LES predictor: the routers select different numbers of experts");
        info.top_k = router_k;
        info.layers.push_back(layer);
    }
    OPENVINO_ASSERT(info.top_k <= impl.num_experts, "LES predictor: the router selects ", info.top_k,
                    " experts, the predictor has ", impl.num_experts);
    return info;
}

std::shared_ptr<ov::Model> build_predictor_model(const LESPredictor& predictor, const PredictorInfo& info) {
    const LESPredictorImpl& impl = *predictor.get_impl();
    const auto hidden = static_cast<int64_t>(impl.hidden_size);
    auto make_input = [&](const std::string& name) {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, hidden});
        input->set_friendly_name(name);
        input->output(0).get_tensor().set_names({name});
        return input;
    };
    ov::ParameterVector parameters;
    ov::ResultVector results;
    for (size_t layer : info.layers) {
        const std::string suffix = std::to_string(layer);
        auto cur = make_input(MODEL_HIDDEN_PREFIX + suffix);
        auto prev = make_input(MODEL_PREV_HIDDEN_PREFIX + suffix);
        parameters.push_back(cur);
        parameters.push_back(prev);
        auto result = std::make_shared<v0::Result>(build_head(impl.heads.at(layer), cur, prev, info.top_k));
        result->set_friendly_name(MODEL_OUTPUT_PREFIX + suffix);
        result->output(0).get_tensor().set_names({MODEL_OUTPUT_PREFIX + suffix});
        results.push_back(result);
    }
    return std::make_shared<ov::Model>(results, parameters, "les_predictor");
}

bool add_plugin_properties(utils::SharedOptional<const ov::AnyMap>& properties,
                           const std::string& device,
                           const std::shared_ptr<ov::Model>& model,
                           const LESPredictor& predictor) {
    if (properties->count(PLUGIN_PREDICTOR_PROPERTY)) {
        return true;  // set by the user
    }
    std::vector<ov::PropertyName> supported;
    try {
        supported = utils::singleton_core().get_property(device, ov::supported_properties);
    } catch (const ov::Exception&) {
        return false;
    }
    if (std::find(supported.begin(), supported.end(), PLUGIN_PREDICTOR_PROPERTY) == supported.end()) {
        GENAI_WARN("LES predictor: device %s does not support %s, the predictor is not used", device.c_str(),
                   PLUGIN_PREDICTOR_PROPERTY);
        return false;
    }
    const PredictorInfo info = analyze(model, predictor);
    auto& forked = properties.fork();
    forked[PLUGIN_PREDICTOR_PROPERTY] = build_predictor_model(predictor, info);
    if (!forked.count(PLUGIN_LOOKAHEAD_PROPERTY)) {
        forked[PLUGIN_LOOKAHEAD_PROPERTY] = true;
    }
    auto cache_dir = forked.find(ov::cache_dir.name());
    if (cache_dir != forked.end()) {
        if (!cache_dir->second.as<std::string>().empty()) {
            GENAI_WARN("LES predictor: the model cache (%s) is disabled, it does not keep the predictor model",
                       ov::cache_dir.name());
        }
        forked.erase(cache_dir);
    }
    return true;
}

}  // namespace les
}  // namespace genai
}  // namespace ov
