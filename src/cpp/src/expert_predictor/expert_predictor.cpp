// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "expert_predictor/expert_predictor.hpp"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>

#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/topk_base.hpp"
#include "openvino/op/util/variable.hpp"
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
        OPENVINO_THROW("Expert predictor: unsupported safetensors dtype ", dtype);
    }
}

std::map<std::string, std::shared_ptr<v0::Constant>> read_safetensors(const std::filesystem::path& path) {
    const ov::Tensor file = ov::read_tensor_data(path);  // read-only mapping of the file
    safetensors_File st{};
    // safetensors_file_init only reads the buffer, it needs a non-const pointer
    auto* data = const_cast<char*>(file.data<const char>());
    const char* error = safetensors_file_init(data, file.get_byte_size(), &st);
    OPENVINO_ASSERT(error == nullptr, "Expert predictor: cannot parse ", path, " as a safetensors file: ", error);

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

std::shared_ptr<v0::Constant> i64_vec(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

std::shared_ptr<v0::Constant> i64_scalar(int64_t value) {
    return v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

void add_output(const std::shared_ptr<ov::Model>& model, const NodeOutput& output, const std::string& name) {
    auto result = std::make_shared<v0::Result>(output);
    result->set_friendly_name(name);
    result->output(0).get_tensor().set_names({name});
    model->add_results({result});
}

std::shared_ptr<v0::Parameter> find_parameter(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_output_tensor(0).get_names().count(name)) {
            return parameter;
        }
    }
    return nullptr;
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

NodeOutput weight_f32(const std::shared_ptr<v0::Constant>& weight) {
    return as_f32(weight->output(0));
}

// top_k(sigmoid(W cur + W' prev) + b): the indices of the predicted experts, i32
NodeOutput build_head(const ExpertPredictorImpl::Head& head,
                      const NodeOutput& cur,
                      const std::optional<NodeOutput>& prev,
                      size_t top_k) {
    NodeOutput logits = std::make_shared<v0::MatMul>(cur, weight_f32(head.weight), false, true);
    if (prev) {
        auto prev_logits = std::make_shared<v0::MatMul>(*prev, weight_f32(head.previous_weight), false, true);
        logits = std::make_shared<v1::Add>(logits, prev_logits);
    }
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
                                            i64_scalar(static_cast<int64_t>(top_k)),
                                            -1,
                                            ov::op::TopKMode::MAX,
                                            ov::op::TopKSortType::SORT_VALUES,
                                            ov::element::i32);
    return topk->output(1);
}

int64_t index_at(const ov::Tensor& tensor, size_t i) {
    if (tensor.get_element_type() == ov::element::i32) {
        return tensor.data<const int32_t>()[i];
    }
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::i64,
                    "Expert predictor: expected i32 or i64 expert indices, got ", tensor.get_element_type());
    return tensor.data<const int64_t>()[i];
}

}  // namespace

ExpertPredictorImpl::ExpertPredictorImpl(const std::filesystem::path& path) {
    OPENVINO_ASSERT(path.extension() == ".safetensors", "Expert predictor: expected a .safetensors file, got ", path);
    static const std::regex tensor_name(R"(expert_predictors\.(\d+)\.(weight|previous_weight|selection_bias|input_mode_code))");
    std::map<size_t, std::map<std::string, std::shared_ptr<v0::Constant>>> parts;
    for (const auto& [name, constant] : read_safetensors(path)) {
        std::smatch match;
        if (std::regex_match(name, match, tensor_name)) {
            parts[std::stoul(match[1].str())][match[2].str()] = constant;
        }
    }
    OPENVINO_ASSERT(!parts.empty(), "Expert predictor: no expert_predictors.<i>.* tensors in ", path);

    for (auto& [index, p] : parts) {
        OPENVINO_ASSERT(index >= 1, "Expert predictor: expert_predictors.<i> uses 1-based layer numbers, got ", index);
        OPENVINO_ASSERT(p.count("weight") && p.count("selection_bias"),
                        "Expert predictor: expert_predictors.", index, " needs weight and selection_bias");
        Head head;
        head.weight = p["weight"];
        head.selection_bias = p["selection_bias"];
        if (p.count("previous_weight")) {
            head.previous_weight = p["previous_weight"];
        }
        head.input_mode = p.count("input_mode_code") ? p["input_mode_code"]->cast_vector<int64_t>().at(0)
                                                     : INPUT_MODE_CURRENT_PREVIOUS;
        OPENVINO_ASSERT(head.input_mode == INPUT_MODE_CURRENT_PREVIOUS,
                        "Expert predictor: expert_predictors.", index, " has input_mode_code ", head.input_mode,
                        ", only ", INPUT_MODE_CURRENT_PREVIOUS, " (current + previous token) is supported");

        const auto& w_shape = head.weight->get_shape();
        OPENVINO_ASSERT(w_shape.size() == 2, "Expert predictor: weight must be [num_experts, hidden_size]");
        if (num_experts == 0) {
            num_experts = w_shape[0];
            hidden_size = w_shape[1];
        }
        OPENVINO_ASSERT(w_shape[0] == num_experts && w_shape[1] == hidden_size,
                        "Expert predictor: all heads must have the same shape, expert_predictors.", index, " has ",
                        w_shape);
        OPENVINO_ASSERT(!head.previous_weight || head.previous_weight->get_shape() == w_shape,
                        "Expert predictor: previous_weight must have the shape of weight");
        OPENVINO_ASSERT(head.selection_bias->get_shape() == ov::Shape{num_experts},
                        "Expert predictor: selection_bias must be [num_experts]");
        heads[index - 1] = std::move(head);
    }
}

ExpertPredictor::ExpertPredictor(const std::filesystem::path& path, size_t top_k, bool use_previous_token)
    : m_pimpl(std::make_shared<ExpertPredictorImpl>(path)),
      m_top_k(top_k),
      m_use_previous_token(use_previous_token) {}

namespace les {

bool stats_enabled() {
    const char* value = std::getenv("OPENVINO_GENAI_EXPERT_PREDICTOR_STATS");
    return value && std::string(value) != "0" && std::string(value) != "";
}

bool extract_from_properties(utils::SharedOptional<const ov::AnyMap>& properties,
                             std::optional<ExpertPredictor>& predictor) {
    const std::string name = ov::genai::expert_predictor.name();
    auto it = properties->find(name);
    if (it == properties->end()) {
        return false;
    }
    if (it->second.is<ExpertPredictor>()) {
        predictor = it->second.as<ExpertPredictor>();
    } else {
        // a path, e.g. from a JSON config of a benchmarking tool
        predictor = ExpertPredictor(it->second.as<std::string>());
    }
    properties.fork().erase(name);
    return true;
}

GraphInfo apply(const std::shared_ptr<ov::Model>& model, const ExpertPredictor& predictor, GraphMode mode) {
    OPENVINO_ASSERT(predictor, "Expert predictor: the predictor is empty");
    const ExpertPredictorImpl& impl = *predictor.get_impl();
    const auto router_inputs = find_router_inputs(model);
    const auto hidden = static_cast<int64_t>(impl.hidden_size);

    GraphInfo info;
    info.hidden_size = impl.hidden_size;
    info.use_previous_token = predictor.get_use_previous_token();
    info.router_outputs = stats_enabled();

    std::map<size_t, std::shared_ptr<ov::op::util::TopKBase>> routers;
    for (const auto& [layer, head] : impl.heads) {
        OPENVINO_ASSERT(router_inputs.count(layer - 1) && router_inputs.count(layer),
                        "Expert predictor: cannot find the post_attention_layernorm outputs of decoder layers ",
                        layer - 1, " and ", layer, " in the model");
        const auto& shape = router_inputs.at(layer).get_partial_shape();
        OPENVINO_ASSERT(shape[2].compatible(hidden), "Expert predictor: the model hidden size ", shape[2],
                        " differs from the predictor hidden size ", hidden);
        auto topk = find_router_topk(router_inputs.at(layer));
        OPENVINO_ASSERT(topk, "Expert predictor: decoder layer ", layer, " has no MoE router (TopK) in the model");
        const size_t router_k = topk->get_k();
        OPENVINO_ASSERT(router_k > 0, "Expert predictor: the router TopK of layer ", layer, " has no constant k");
        OPENVINO_ASSERT(info.router_top_k == 0 || info.router_top_k == router_k,
                        "Expert predictor: the routers select different numbers of experts");
        info.router_top_k = router_k;
        OPENVINO_ASSERT(!predictor.get_use_previous_token() || head.previous_weight,
                        "Expert predictor: use_previous_token needs previous_weight in expert_predictors.", layer + 1);
        routers[layer] = topk;
        info.layers.push_back(layer);
    }
    info.top_k = predictor.get_top_k() ? predictor.get_top_k() : info.router_top_k;
    OPENVINO_ASSERT(info.top_k <= impl.num_experts, "Expert predictor: top_k ", info.top_k, " exceeds the ",
                    impl.num_experts, " experts");

    if (mode == GraphMode::STATEFUL) {
        // hidden states [batch, seq_len, hidden]; the previous token of the first token comes from a state
        auto beam_idx = find_parameter(model, "beam_idx");
        for (size_t layer : info.layers) {
            const auto& head = impl.heads.at(layer);
            NodeOutput cur = as_f32(router_inputs.at(layer - 1));
            NodeOutput same_layer = as_f32(router_inputs.at(layer));
            std::optional<NodeOutput> prev;
            if (predictor.get_use_previous_token()) {
                auto shape = std::make_shared<v3::ShapeOf>(same_layer, ov::element::i64);
                auto batch = std::make_shared<v8::Gather>(shape, i64_vec({0}), i64_scalar(0));
                auto seq_len = std::make_shared<v8::Gather>(shape, i64_vec({1}), i64_scalar(0));
                auto init_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch, i64_vec({hidden})}, 0);
                auto init = std::make_shared<v3::Broadcast>(v0::Constant::create(ov::element::f32, {}, {0.0f}), init_shape);
                auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
                    ov::PartialShape{-1, hidden}, ov::element::f32, STATE_PREFIX + std::to_string(layer)});
                NodeOutput state = std::make_shared<v6::ReadValue>(init, variable);
                if (beam_idx) {
                    state = std::make_shared<v8::Gather>(state, beam_idx, i64_scalar(0));
                }
                // [batch, seq_len + 1, hidden] -> first seq_len rows: row t holds token t - 1
                auto with_state = std::make_shared<v0::Concat>(
                    ov::OutputVector{std::make_shared<v0::Unsqueeze>(state, i64_vec({1})), same_layer}, 1);
                prev = std::make_shared<v8::Slice>(with_state, i64_vec({0}), seq_len, i64_vec({1}), i64_vec({1}));
                auto last = std::make_shared<v8::Gather>(same_layer, i64_scalar(-1), i64_scalar(1));
                auto assign = std::make_shared<v6::Assign>(last, variable);
                model->add_variables({variable});
                model->add_sinks({assign});
            }
            add_output(model, build_head(head, cur, prev, info.top_k), PREDICTED_EXPERTS_PREFIX + std::to_string(layer));
        }
    } else {
        // hidden states [total_tokens, 1, hidden] of all scheduled sequences, split by subsequence_begins; GenAI
        // passes the previous token of the first token of every sequence and reads the last token back
        auto subsequence_begins = find_parameter(model, "subsequence_begins");
        OPENVINO_ASSERT(subsequence_begins, "Expert predictor: the PA model has no subsequence_begins input");
        NodeOutput begins_i = subsequence_begins->output(0);
        auto begins = std::make_shared<v8::Slice>(begins_i, i64_vec({0}), i64_vec({-1}), i64_vec({1}), i64_vec({0}));
        auto ends = std::make_shared<v8::Slice>(begins_i,
                                                i64_vec({1}),
                                                i64_vec({std::numeric_limits<int64_t>::max()}),
                                                i64_vec({1}),
                                                i64_vec({0}));
        auto last_rows = std::make_shared<v1::Subtract>(ends, v0::Constant::create(begins_i.get_element_type(), {}, {1}));
        auto zero_row = v0::Constant::create(ov::element::f32, {1, impl.hidden_size}, std::vector<float>(impl.hidden_size, 0.0f));

        std::shared_ptr<v0::Parameter> prev_input;
        if (predictor.get_use_previous_token()) {
            prev_input = std::make_shared<v0::Parameter>(
                ov::element::f32, ov::PartialShape{static_cast<int64_t>(info.layers.size()), -1, hidden});
            prev_input->set_friendly_name(PREV_HIDDEN_INPUT);
            prev_input->output(0).get_tensor().set_names({PREV_HIDDEN_INPUT});
        }
        auto flatten = [&](const NodeOutput& x) -> NodeOutput {
            return std::make_shared<v1::Reshape>(as_f32(x), i64_vec({-1, hidden}), false);
        };

        ov::OutputVector last_hidden;
        for (size_t i = 0; i < info.layers.size(); ++i) {
            const size_t layer = info.layers[i];
            const auto& head = impl.heads.at(layer);
            NodeOutput cur = flatten(router_inputs.at(layer - 1));
            NodeOutput same_layer = flatten(router_inputs.at(layer));
            std::optional<NodeOutput> prev;
            if (predictor.get_use_previous_token()) {
                auto num_tokens = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(same_layer, ov::element::i64),
                                                               i64_vec({0}),
                                                               i64_scalar(0));
                // row t holds token t - 1, then the first row of every sequence takes the cached previous token
                auto shifted = std::make_shared<v8::Slice>(std::make_shared<v0::Concat>(ov::OutputVector{zero_row, same_layer}, 0),
                                                           i64_vec({0}),
                                                           num_tokens,
                                                           i64_vec({1}),
                                                           i64_vec({0}));
                auto cached = std::make_shared<v8::Gather>(prev_input, i64_scalar(static_cast<int64_t>(i)), i64_scalar(0));
                prev = std::make_shared<v3::ScatterUpdate>(shifted, begins, cached, i64_scalar(0));
                auto last = std::make_shared<v8::Gather>(same_layer, last_rows, i64_scalar(0));
                last_hidden.push_back(std::make_shared<v0::Unsqueeze>(last, i64_vec({0})));
            }
            add_output(model, build_head(head, cur, prev, info.top_k), PREDICTED_EXPERTS_PREFIX + std::to_string(layer));
        }
        if (predictor.get_use_previous_token()) {
            model->add_parameters({prev_input});
            add_output(model, std::make_shared<v0::Concat>(last_hidden, 0), LAST_HIDDEN_OUTPUT);
        }
    }

    if (info.router_outputs) {
        for (size_t layer : info.layers) {
            add_output(model, routers.at(layer)->output(1), ROUTER_EXPERTS_PREFIX + std::to_string(layer));
        }
    }
    model->validate_nodes_and_infer_types();
    return info;
}

Stats::Stats(GraphInfo info)
    : m_info(std::move(info)),
      m_hits(m_info.layers.size(), 0),
      m_total(m_info.layers.size(), 0) {}

Stats::~Stats() {
    if (m_tokens > 0) {
        print(std::cout);
    }
}

void Stats::accumulate(ov::InferRequest& request, const std::vector<size_t>& rows) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t router_k = m_info.router_top_k;
    size_t counted = 0;
    for (size_t i = 0; i < m_info.layers.size(); ++i) {
        const std::string layer = std::to_string(m_info.layers[i]);
        ov::Tensor predicted = request.get_tensor(PREDICTED_EXPERTS_PREFIX + layer);
        ov::Tensor router = request.get_tensor(ROUTER_EXPERTS_PREFIX + layer);
        const size_t num_rows = router.get_size() / router_k;
        if (num_rows == 0) {
            return;
        }
        const size_t predicted_k = predicted.get_size() / num_rows;
        auto count_row = [&](size_t row) {
            for (size_t a = 0; a < router_k; ++a) {
                const int64_t expert = index_at(router, row * router_k + a);
                for (size_t b = 0; b < predicted_k; ++b) {
                    if (index_at(predicted, row * predicted_k + b) == expert) {
                        ++m_hits[i];
                        break;
                    }
                }
            }
            m_total[i] += router_k;
        };
        counted = 0;
        if (rows.empty()) {
            for (size_t row = 0; row < num_rows; ++row, ++counted) {
                count_row(row);
            }
        } else {
            for (size_t row : rows) {
                if (row < num_rows) {
                    count_row(row);
                    ++counted;
                }
            }
        }
    }
    m_tokens += counted;
}

void Stats::accumulate_stateful_decode(ov::InferRequest& request) {
    if (m_info.layers.empty()) {
        return;
    }
    // predicted experts of the stateful model: [batch, seq_len, top_k]
    const auto shape = request.get_tensor(PREDICTED_EXPERTS_PREFIX + std::to_string(m_info.layers[0])).get_shape();
    if (shape.size() == 3 && shape[1] == 1) {
        accumulate(request);
    }
}

void Stats::print(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    uint64_t hits = 0, total = 0;
    size_t worst = 0, best = 0;
    auto recall = [&](size_t i) {
        return m_total[i] ? static_cast<double>(m_hits[i]) / static_cast<double>(m_total[i]) : 0.0;
    };
    for (size_t i = 0; i < m_info.layers.size(); ++i) {
        hits += m_hits[i];
        total += m_total[i];
        worst = recall(i) < recall(worst) ? i : worst;
        best = recall(i) > recall(best) ? i : best;
    }
    os << std::fixed << std::setprecision(3) << "[ExpertPredictor] decode tokens: " << m_tokens
       << ", predicted experts: " << m_info.top_k << ", router experts: " << m_info.router_top_k
       << ", previous token: " << (m_info.use_previous_token ? "yes" : "no")
       << ", recall: " << (total ? static_cast<double>(hits) / static_cast<double>(total) : 0.0)
       << " (layer " << m_info.layers[worst] << " " << recall(worst) << " .. layer " << m_info.layers[best] << " "
       << recall(best) << ")" << std::endl;
}

PagedState::PagedState(GraphInfo info) : m_info(std::move(info)) {
    if (m_info.router_outputs) {
        m_stats = std::make_unique<Stats>(m_info);
    }
}

void PagedState::set_inputs(ov::InferRequest& request, const std::vector<ScheduledSequence>& sequences) {
    if (!m_info.use_previous_token) {
        return;
    }
    const size_t num_heads = m_info.layers.size(), num_sequences = sequences.size(), hidden = m_info.hidden_size;
    ov::Tensor prev(ov::element::f32, {num_heads, num_sequences, hidden});
    float* dst = prev.data<float>();
    for (size_t s = 0; s < num_sequences; ++s) {
        auto it = m_prev.find(sequences[s].id);
        for (size_t i = 0; i < num_heads; ++i) {
            float* row = dst + (i * num_sequences + s) * hidden;
            if (it != m_prev.end()) {
                std::copy_n(it->second.data() + i * hidden, hidden, row);
            } else {
                std::fill_n(row, hidden, 0.0f);  // a new sequence has no previous token yet
            }
        }
    }
    request.set_tensor(PREV_HIDDEN_INPUT, prev);
}

void PagedState::update(ov::InferRequest& request,
                        const std::vector<ScheduledSequence>& sequences,
                        const std::unordered_set<uint64_t>& alive_sequence_ids) {
    if (m_info.use_previous_token) {
        const size_t num_heads = m_info.layers.size(), num_sequences = sequences.size(), hidden = m_info.hidden_size;
        ov::Tensor last = request.get_tensor(LAST_HIDDEN_OUTPUT);
        OPENVINO_ASSERT(last.get_element_type() == ov::element::f32 && last.get_size() == num_heads * num_sequences * hidden,
                        "Expert predictor: unexpected ", LAST_HIDDEN_OUTPUT, " output ", last.get_shape());
        const float* src = last.data<const float>();
        for (size_t s = 0; s < num_sequences; ++s) {
            auto& stored = m_prev[sequences[s].id];
            stored.resize(num_heads * hidden);
            for (size_t i = 0; i < num_heads; ++i) {
                std::copy_n(src + (i * num_sequences + s) * hidden, hidden, stored.data() + i * hidden);
            }
        }
        for (auto it = m_prev.begin(); it != m_prev.end();) {
            it = alive_sequence_ids.count(it->first) ? std::next(it) : m_prev.erase(it);
        }
    }
    if (m_stats) {
        // count decode tokens only: the first row of every sequence with one scheduled token
        std::vector<size_t> rows;
        size_t begin = 0;
        for (const auto& sequence : sequences) {
            if (sequence.num_tokens == 1) {
                rows.push_back(begin);
            }
            begin += sequence.num_tokens;
        }
        if (!rows.empty()) {
            m_stats->accumulate(request, rows);
        }
    }
}

}  // namespace les
}  // namespace genai
}  // namespace ov
