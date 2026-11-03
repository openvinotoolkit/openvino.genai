// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_model_transforms.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/topk_base.hpp"
#include "openvino/runtime/properties.hpp"

#include "eagle3_model_transforms.hpp"
#include "logger.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

namespace {

constexpr const char* DFLASH_HIDDEN_STATES_RT_INFO_KEY = "hidden_states_decoder_layers";
constexpr const char* LAST_HIDDEN_STATE_OUTPUT_NAME = "last_hidden_state";
constexpr const char* DFLASH_GPU_MIN_ACTIVATIONS_SCALE_FACTOR = "dflash_gpu_min_activations_scale_factor";

float parse_dflash_gpu_min_activations_scale_factor(const std::string& value) {
    size_t parsed_characters = 0;
    float scale_factor = 0.0f;
    try {
        scale_factor = std::stof(value, &parsed_characters);
    } catch (const std::logic_error&) {
        parsed_characters = 0;
    }
    OPENVINO_ASSERT(parsed_characters > 0 && parsed_characters == value.size() && std::isfinite(scale_factor) &&
                        scale_factor > 0.0f,
                    "DFlash v2 gpu_min_activations_scale_factor must be a finite positive number, got '",
                    value,
                    "'.");
    return scale_factor;
}

constexpr const char* DFLASH_MARKOV_RANK = "dflash_markov_rank";
constexpr const char* DFLASH_MARKOV_HEAD_TYPE = "dflash_markov_head_type";
// The draft property that selects the Markov head computation; read_markov_head() consumes it.
constexpr const char* MARKOV_HEAD_TOP_K = "markov_head_top_k";
// The internal property that carries the selected computation from draft_model() to the pipeline.
constexpr const char* DFLASH_MARKOV_HEAD_COMPUTATION = "dflash_markov_head_computation";
constexpr const char* MARKOV_HEAD_MODEL_FILE_NAME = "openvino_markov_head_model.xml";
constexpr const char* VANILLA_MARKOV_HEAD = "vanilla";
// The DFlash-2 selector: its transitions are gated by the draft hidden states, so it has no dense step.
constexpr const char* DFLASH2_MARKOV_HEAD = "dflash2";
// Below this K, the candidate scorer often misses the token the Markov head would pick, and acceptance drops.
constexpr size_t MIN_RECOMMENDED_MARKOV_HEAD_TOP_K = 8;

void add_dflash_hidden_state_result(std::shared_ptr<ov::Model>& model,
                                    const std::vector<ov::Output<ov::Node>>& hidden_state_outputs) {
    ov::Output<ov::Node> output_to_operate;
    if (hidden_state_outputs.size() > 1) {
        auto concat = std::make_shared<ov::op::v0::Concat>(hidden_state_outputs, -1);
        concat->set_friendly_name("dflash_hidden_states_concat");
        output_to_operate = concat->output(0);
    } else {
        output_to_operate = hidden_state_outputs[0];
    }

    auto result = std::make_shared<ov::op::v0::Result>(output_to_operate);
    result->output(0).set_names({LAST_HIDDEN_STATE_OUTPUT_NAME});
    result->set_friendly_name(LAST_HIDDEN_STATE_OUTPUT_NAME);
    result->get_rt_info()["manually_added_output"] = true;
    model->add_results({result});
}

bool is_rank3_hidden_output(const ov::Output<ov::Node>& output,
                            std::optional<size_t> expected_hidden_size = std::nullopt) {
    const auto shape = output.get_partial_shape();
    return shape.rank().is_static() && shape.rank().get_length() == 3 && shape[2].is_static() &&
           (!expected_hidden_size || static_cast<size_t>(shape[2].get_length()) == *expected_hidden_size);
}

std::shared_ptr<ov::Node> resolve_unique_live_node(const std::vector<std::shared_ptr<ov::Node>>& live_nodes,
                                                   const std::string& producer) {
    std::shared_ptr<ov::Node> match;
    size_t node_count = 0;
    for (const auto& node : live_nodes) {
        if (node->get_friendly_name() == producer) {
            match = node;
            ++node_count;
        }
    }
    OPENVINO_ASSERT(node_count == 1, "Hidden-state producer ", producer, " resolved to ", node_count, " live nodes.");
    return match;
}

ov::Output<ov::Node> resolve_live_hidden_state_output(
    const std::vector<std::shared_ptr<ov::Node>>& live_nodes,
    const std::string& producer,
    nlohmann::json::number_unsigned_t output_index,
    std::optional<size_t> expected_hidden_size = std::nullopt) {
    const auto node = resolve_unique_live_node(live_nodes, producer);
    OPENVINO_ASSERT(output_index < node->get_output_size(),
                    "Hidden-state producer ",
                    producer,
                    " has no output ",
                    output_index,
                    ".");
    const auto output = node->output(static_cast<size_t>(output_index));
    OPENVINO_ASSERT(is_rank3_hidden_output(output, expected_hidden_size),
                    "Hidden-state producer ",
                    producer,
                    " output ",
                    output_index,
                    " must be rank-3 with a static hidden width",
                    expected_hidden_size ? " matching the retained hidden width." : ".");
    return output;
}

template <typename T>
std::optional<T> get_rt_info_value(const std::shared_ptr<ov::Model>& model, const std::vector<std::string>& path) {
    if (!model->has_rt_info(path)) {
        return std::nullopt;
    }
    return model->get_rt_info<T>(path);
}

std::vector<int32_t> parse_layer_ids(const std::string& raw) {
    std::vector<int32_t> result;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            continue;
        }
        std::istringstream item_stream(item);
        int32_t layer_id{};
        if (!(item_stream >> layer_id) || (item_stream >> std::ws && !item_stream.eof())) {
            OPENVINO_THROW("Malformed dflash_target_layer_ids RT info value: '",
                           raw,
                           "'. Invalid layer id: '",
                           item,
                           "'.");
        }
        result.push_back(layer_id);
    }
    return result;
}

std::shared_ptr<ov::op::v0::Parameter> find_inputs_embeds_parameter(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "inputs_embeds" ||
            parameter->output(0).get_names().count("inputs_embeds") != 0) {
            return parameter;
        }
    }
    return nullptr;
}

std::vector<ov::Input<ov::Node>> get_live_consumers(const std::shared_ptr<ov::Model>& model,
                                                     const ov::Output<ov::Node>& output) {
    std::unordered_set<const ov::Node*> live_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        live_nodes.insert(node.get());
    }

    std::vector<ov::Input<ov::Node>> consumers;
    for (auto consumer : output.get_target_inputs()) {
        if (live_nodes.count(consumer.get_node()) != 0) {
            consumers.push_back(consumer);
        }
    }
    return consumers;
}

ov::Output<ov::Node> scale_draft_embeddings(const ov::Output<ov::Node>& embeddings,
                                             float input_embedding_scale) {
    if (input_embedding_scale == 1.0f) {
        return embeddings;
    }
    auto scale = ov::op::v0::Constant::create(embeddings.get_element_type(),
                                               ov::Shape{},
                                               std::vector<float>{input_embedding_scale});
    auto scaled_embeddings = std::make_shared<ov::op::v1::Multiply>(embeddings, scale);
    scaled_embeddings->set_friendly_name("dflash_draft_embedding_scale");
    return scaled_embeddings;
}

std::string read_markov_head_rt_info(const std::shared_ptr<ov::Model>& model, const char* model_name, const char* key) {
    auto value = get_rt_info_value<std::string>(model, {"dflash", key});
    OPENVINO_ASSERT(value, "The ", model_name, " is missing dflash/", key, " RT info.");
    return *value;
}

// Checks the RT info that the dense step and the candidate scorer share with the draft backbone and returns their
// vocabulary size.
size_t validate_markov_head_rt_info(const std::shared_ptr<ov::Model>& model,
                                    const char* model_name,
                                    const DFlashRTInfo& info) {
    OPENVINO_ASSERT(read_markov_head_rt_info(model, model_name, "markov_head_type") == info.markov_head_type &&
                        std::stoull(read_markov_head_rt_info(model, model_name, "markov_rank")) == info.markov_rank,
                    "The ",
                    model_name,
                    " does not match the Markov head of the draft backbone.");
    OPENVINO_ASSERT(model->get_variables().empty(), "The ", model_name, " must be stateless.");
    return std::stoull(read_markov_head_rt_info(model, model_name, "vocab_size"));
}

ov::PartialShape validate_markov_head_port(const ov::Output<ov::Node>& port,
                                           const char* model_name,
                                           const ov::element::Type& element_type,
                                           int64_t rank) {
    const auto shape = port.get_partial_shape();
    OPENVINO_ASSERT(
        port.get_element_type() == element_type && shape.rank().is_static() && shape.rank().get_length() == rank,
        "The ",
        model_name,
        " '",
        port.get_any_name(),
        "' must be a rank-",
        rank,
        " ",
        element_type,
        " tensor.");
    return shape;
}

bool dimension_matches(const ov::Dimension& dimension, size_t expected) {
    return dimension.is_dynamic() || static_cast<size_t>(dimension.get_length()) == expected;
}

bool has_port_named(const std::vector<ov::Output<ov::Node>>& ports, const std::string& name) {
    return std::any_of(ports.begin(), ports.end(), [&](const ov::Output<ov::Node>& port) {
        return port.get_names().count(name) != 0;
    });
}

// Builds a model from one computation of the Markov head model. ov::Model rejects the computation if it reads an
// input that is not listed, so the dense step and the candidate scorer must not share inputs.
std::shared_ptr<ov::Model> keep_markov_head_computation(const std::shared_ptr<ov::Model>& markov_head_model,
                                                        const std::vector<std::string>& input_names,
                                                        const std::vector<std::string>& output_names) {
    ov::ParameterVector parameters;
    for (const auto& name : input_names) {
        OPENVINO_ASSERT(has_port_named(markov_head_model->inputs(), name),
                        "The Markov head model has no '",
                        name,
                        "' input.");
        const auto parameter = markov_head_model->input(name).get_node_shared_ptr();
        parameters.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(parameter));
    }
    ov::ResultVector results;
    for (const auto& name : output_names) {
        OPENVINO_ASSERT(has_port_named(markov_head_model->outputs(), name),
                        "The Markov head model has no '",
                        name,
                        "' output.");
        const auto result = markov_head_model->output(name).get_node_shared_ptr();
        results.push_back(ov::as_type_ptr<ov::op::v0::Result>(result));
    }
    auto computation = std::make_shared<ov::Model>(results, parameters, markov_head_model->get_friendly_name());
    for (const char* key : {"vocab_size", "markov_rank", "markov_head_type"}) {
        if (auto value = get_rt_info_value<std::string>(markov_head_model, {"dflash", key})) {
            computation->set_rt_info(*value, {"dflash", key});
        }
    }
    return computation;
}

// The candidate scorer derives every K-sized shape from its TopK, so replacing the TopK's K resizes the whole scorer.
void replace_markov_head_top_k(const std::shared_ptr<ov::Model>& candidate_scorer, size_t top_k) {
    std::shared_ptr<ov::op::util::TopKBase> topk;
    for (const auto& node : candidate_scorer->get_ordered_ops()) {
        if (auto candidate = ov::as_type_ptr<ov::op::util::TopKBase>(node)) {
            OPENVINO_ASSERT(!topk, "The Markov head candidate scorer must have exactly one TopK.");
            topk = candidate;
        }
    }
    OPENVINO_ASSERT(topk, "The Markov head candidate scorer must have exactly one TopK.");
    const auto exported_k = topk->input_value(1);
    topk->input(1).replace_source_output(
        ov::op::v0::Constant::create(exported_k.get_element_type(), exported_k.get_shape(), {top_k}));
    candidate_scorer->validate_nodes_and_infer_types();
}

}  // namespace

std::optional<std::vector<DFlashHiddenStateLocator>> resolve_target_hidden_state_locators(
    const std::shared_ptr<ov::Model>& model,
    const std::vector<int32_t>& target_layer_ids) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    OPENVINO_ASSERT(!target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    if (!model->has_rt_info(DFLASH_HIDDEN_STATES_RT_INFO_KEY)) {
        return std::nullopt;
    }

    nlohmann::json annotation;
    try {
        annotation = nlohmann::json::parse(model->get_rt_info<std::string>(DFLASH_HIDDEN_STATES_RT_INFO_KEY));
    } catch (const nlohmann::json::exception& error) {
        OPENVINO_THROW("Malformed hidden-state annotation metadata in model rt_info: ", error.what());
    }

    std::set<std::pair<std::string, nlohmann::json::number_unsigned_t>> producer_outputs;
    std::vector<DFlashHiddenStateLocator> resolved;
    resolved.reserve(target_layer_ids.size());
    try {
        const auto& layers = annotation.at("layers");
        const auto live_nodes = model->get_ordered_ops();
        for (const auto layer_id : target_layer_ids) {
            const auto key = std::to_string(layer_id);
            const auto& locator = layers.at(key);
            const auto producer = locator.at("producer").get<std::string>();
            const auto output_index =
                locator.at("output_index").get_ref<const nlohmann::json::number_unsigned_t&>();
            OPENVINO_ASSERT(producer_outputs.emplace(producer, output_index).second,
                            "Duplicate producer/output locator for requested decoder layers: ",
                            producer,
                            "[",
                            output_index,
                            "].");

            const auto output = resolve_live_hidden_state_output(live_nodes, producer, output_index);
            resolved.push_back({producer, output, static_cast<size_t>(output.get_partial_shape()[2].get_length())});
        }
    } catch (const nlohmann::json::exception& error) {
        OPENVINO_THROW("Malformed hidden-state annotation metadata: ", error.what());
    }
    return resolved;
}

void expose_target_hidden_states(std::shared_ptr<ov::Model>& model,
                                 const std::optional<std::vector<DFlashHiddenStateLocator>>& retained_locators,
                                 const std::vector<int32_t>& target_layer_ids) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    OPENVINO_ASSERT(!target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    std::vector<ov::Output<ov::Node>> outputs;

    if (retained_locators) {
        OPENVINO_ASSERT(!retained_locators->empty(), "DFlash hidden-state locators cannot be empty.");
        OPENVINO_ASSERT(retained_locators->size() == target_layer_ids.size(),
                        "DFlash hidden-state locator count does not match target_layer_ids.");
        const auto live_nodes = model->get_ordered_ops();
        outputs.reserve(retained_locators->size());
        for (const auto& retained : *retained_locators) {
            const auto current_output = resolve_live_hidden_state_output(
                live_nodes,
                retained.producer,
                retained.output.get_index(),
                retained.hidden_size);
            OPENVINO_ASSERT(current_output.get_node_shared_ptr() == retained.output.get_node_shared_ptr(),
                            "DFlash hidden-state producer ",
                            retained.producer,
                            " no longer resolves to the retained output identity.");
            outputs.push_back(retained.output);
        }
    } else {
        OPENVINO_ASSERT(!model->has_rt_info(DFLASH_HIDDEN_STATES_RT_INFO_KEY),
                        "DFlash hidden-state fallback cannot be used when target metadata is present.");
        outputs = eagle3::find_decoder_layer_hidden_state_outputs(model, target_layer_ids);
        OPENVINO_ASSERT(outputs.size() == target_layer_ids.size(),
                        "DFlash target model has no hidden-state RT info and Eagle3 fallback found ",
                        outputs.size(),
                        " outputs for ",
                        target_layer_ids.size(),
                        " requested target layers.");
    }
    add_dflash_hidden_state_result(model, outputs);
}

namespace {

constexpr const char* DFLASH_DRAFT_FC_WEIGHTS_HASH = "dflash_draft_fc_weights_hash";

// A hash of the values of every Constant the output depends on.
std::string hash_constant_values(const ov::Output<ov::Node>& output) {
    size_t seed = 0;
    std::vector<ov::Node*> pending{output.get_node()};
    std::unordered_set<ov::Node*> visited;
    while (!pending.empty()) {
        auto* node = pending.back();
        pending.pop_back();
        if (!visited.insert(node).second) {
            continue;
        }
        if (const auto* constant = ov::as_type<ov::op::v0::Constant>(node)) {
            const std::string_view bytes(static_cast<const char*>(constant->get_data_ptr()), constant->get_byte_size());
            seed ^= std::hash<std::string_view>{}(bytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        for (const auto& input : node->input_values()) {
            pending.push_back(input.get_node());
        }
    }
    return std::to_string(seed);
}

}  // namespace

void move_fc_from_draft_to_main(const std::shared_ptr<ov::Model>& draft_model,
                                const std::shared_ptr<ov::Model>& main_model) {
    OPENVINO_ASSERT(draft_model && main_model, "DFlash FC move requires both target and draft models.");
    const auto hidden_states =
        ov::as_type_ptr<ov::op::v0::Parameter>(draft_model->input("hidden_states").get_node_shared_ptr());
    const auto hidden_result =
        ov::as_type_ptr<ov::op::v0::Result>(main_model->output(LAST_HIDDEN_STATE_OUTPUT_NAME).get_node_shared_ptr());

    // hidden_states is resized to the FC output width, so the FC data input must be its only consumer.
    const auto consumers = hidden_states->output(0).get_target_inputs();
    const auto fc = consumers.size() == 1 && consumers.begin()->get_index() == 0
                        ? ov::as_type_ptr<ov::op::v0::MatMul>(consumers.begin()->get_node()->shared_from_this())
                        : nullptr;
    OPENVINO_ASSERT(fc, "DFlash draft hidden_states must feed only the data input of its hidden-state FC MatMul.");
    const auto target_width = hidden_result->get_input_partial_shape(0)[2];
    const auto draft_width = hidden_states->get_partial_shape()[2];
    OPENVINO_ASSERT(target_width == draft_width,
                    "DFlash target exports hidden states of width ", target_width,
                    ", but the draft hidden_states input has width ", draft_width, ".");

    // The weight subgraph moves to the target as is, keeping its decompression markers; once the draft FC
    // consumers are rewired below, nothing in the draft uses it.
    auto target_fc = std::make_shared<ov::op::v0::MatMul>(hidden_result->input_value(0),
                                                          fc->input_value(1),
                                                          fc->get_transpose_a(),
                                                          fc->get_transpose_b());
    target_fc->set_friendly_name("dflash_hidden_state_fc");
    // The target compiles with ov::cache_model_path, so its model cache key leaves out weight values but takes plain
    // rt_info. Without the FC weight hash, drafts of one architecture would share the cached target of the first.
    target_fc->get_rt_info()[DFLASH_DRAFT_FC_WEIGHTS_HASH] = hash_constant_values(fc->input_value(1));
    hidden_result->input(0).replace_source_output(target_fc);

    auto hidden_states_shape = hidden_states->get_partial_shape();
    hidden_states_shape[2] = fc->get_output_partial_shape(0)[2];
    hidden_states->set_partial_shape(hidden_states_shape);
    for (auto fc_consumer : fc->output(0).get_target_inputs()) {
        fc_consumer.replace_source_output(hidden_states);
    }

    main_model->validate_nodes_and_infer_types();
    draft_model->validate_nodes_and_infer_types();
}

void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (!model->has_rt_info("dflash_mode") || !model->get_rt_info<bool>("dflash_mode")) {
        return;
    }

    properties["dflash_mode"] = true;
    if (auto version = get_rt_info_value<std::string>(model, {"dflash", "version"})) {
        properties["dflash_version"] = static_cast<int64_t>(std::stoll(*version));
    }
    if (auto mask_token_id = get_rt_info_value<std::string>(model, {"dflash", "mask_token_id"})) {
        properties["dflash_mask_token_id"] = static_cast<int64_t>(std::stoll(*mask_token_id));
    }
    if (auto target_layer_ids = get_rt_info_value<std::string>(model, {"dflash", "target_layer_ids"})) {
        properties["dflash_target_layer_ids"] = parse_layer_ids(*target_layer_ids);
    }
    if (auto offset = get_rt_info_value<std::string>(model, {"dflash", "candidate_position_offset"})) {
        properties["dflash_candidate_position_offset"] = static_cast<size_t>(std::stoul(*offset));
    }
    if (auto gpu_min_scale = get_rt_info_value<std::string>(
            model,
            {"dflash", "gpu_min_activations_scale_factor"})) {
        properties[DFLASH_GPU_MIN_ACTIVATIONS_SCALE_FACTOR] =
            parse_dflash_gpu_min_activations_scale_factor(*gpu_min_scale);
    }
    if (auto input_scale = get_rt_info_value<std::string>(model, {"dflash", "input_embedding_scale"})) {
        properties["dflash_input_embedding_scale"] = std::stof(*input_scale);
    }
    if (auto output_multiplier = get_rt_info_value<std::string>(model, {"dflash", "output_multiplier"})) {
        properties["dflash_output_multiplier"] = std::stof(*output_multiplier);
    }
    if (auto softcap = get_rt_info_value<std::string>(model, {"dflash", "final_logit_softcapping"})) {
        properties["dflash_final_logit_softcapping"] = std::stof(*softcap);
    }
    // Drafts with a Markov head, such as DSpark. A missing or zero markov_rank is a plain DFlash draft.
    if (auto markov_rank = get_rt_info_value<std::string>(model, {"dflash", "markov_rank"})) {
        properties[DFLASH_MARKOV_RANK] = static_cast<size_t>(std::stoull(*markov_rank));
    }
    if (auto head_type = get_rt_info_value<std::string>(model, {"dflash", "markov_head_type"})) {
        properties[DFLASH_MARKOV_HEAD_TYPE] = *head_type;
    }
}

DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config) {
    DFlashRTInfo info;
    auto mode_it = config.find("dflash_mode");
    if (mode_it == config.end()) {
        return info;
    }

    info.dflash_mode = mode_it->second.as<bool>();
    config.erase(mode_it);
    if (!info.dflash_mode) {
        return info;
    }

    if (auto version_it = config.find("dflash_version"); version_it != config.end()) {
        info.dflash_version = version_it->second.as<int64_t>();
        config.erase(version_it);
    }
    OPENVINO_ASSERT(info.dflash_version == 1 || info.dflash_version == 2,
                    "Unsupported DFlash version: ",
                    info.dflash_version,
                    ". Supported versions are 1 and 2.");

    auto mask_it = config.find("dflash_mask_token_id");
    OPENVINO_ASSERT(mask_it != config.end(), "DFlash draft model is missing dflash_mask_token_id RT info.");
    info.mask_token_id = mask_it->second.as<int64_t>();
    OPENVINO_ASSERT(info.mask_token_id >= 0, "DFlash dflash_mask_token_id must be non-negative, got ", info.mask_token_id, ".");
    config.erase(mask_it);

    auto layers_it = config.find("dflash_target_layer_ids");
    OPENVINO_ASSERT(layers_it != config.end(), "DFlash draft model is missing dflash_target_layer_ids RT info.");
    info.target_layer_ids = layers_it->second.as<std::vector<int32_t>>();
    config.erase(layers_it);
    OPENVINO_ASSERT(!info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");

    auto offset_it = config.find("dflash_candidate_position_offset");
    if (offset_it != config.end()) {
        info.candidate_position_offset = offset_it->second.as<size_t>();
        OPENVINO_ASSERT(info.candidate_position_offset == 0 || info.candidate_position_offset == 1,
                        "DFlash candidate_position_offset must be 0 or 1, got ",
                        info.candidate_position_offset,
                        ".");
        config.erase(offset_it);
    } else {
        info.candidate_position_offset = 1;
    }
    if (auto gpu_min_scale_it = config.find(DFLASH_GPU_MIN_ACTIVATIONS_SCALE_FACTOR);
        gpu_min_scale_it != config.end()) {
        info.gpu_min_activations_scale_factor = gpu_min_scale_it->second.as<float>();
        config.erase(gpu_min_scale_it);
        OPENVINO_ASSERT(
            std::isfinite(info.gpu_min_activations_scale_factor) && info.gpu_min_activations_scale_factor > 0.0f,
            "DFlash gpu_min_activations_scale_factor must be a finite positive number, got ",
            info.gpu_min_activations_scale_factor,
            ".");
    }
    OPENVINO_ASSERT(info.dflash_version != 2 || info.gpu_min_activations_scale_factor > 0.0f,
                    "DFlash v2 draft model is missing dflash/gpu_min_activations_scale_factor metadata.");
    if (auto input_scale_it = config.find("dflash_input_embedding_scale"); input_scale_it != config.end()) {
        info.input_embedding_scale = input_scale_it->second.as<float>();
        config.erase(input_scale_it);
    }
    if (auto output_multiplier_it = config.find("dflash_output_multiplier"); output_multiplier_it != config.end()) {
        info.output_multiplier = output_multiplier_it->second.as<float>();
        config.erase(output_multiplier_it);
    }
    if (auto softcap_it = config.find("dflash_final_logit_softcapping"); softcap_it != config.end()) {
        info.final_logit_softcapping = softcap_it->second.as<float>();
        config.erase(softcap_it);
    }
    if (auto rank_it = config.find(DFLASH_MARKOV_RANK); rank_it != config.end()) {
        info.markov_rank = rank_it->second.as<size_t>();
        config.erase(rank_it);
    }
    if (auto head_type_it = config.find(DFLASH_MARKOV_HEAD_TYPE); head_type_it != config.end()) {
        info.markov_head_type = head_type_it->second.as<std::string>();
        config.erase(head_type_it);
    }
    if (auto computation_it = config.find(DFLASH_MARKOV_HEAD_COMPUTATION); computation_it != config.end()) {
        info.markov_head_computation = computation_it->second.as<std::shared_ptr<ov::Model>>();
        config.erase(computation_it);
    }
    OPENVINO_ASSERT(config.count(MARKOV_HEAD_TOP_K) == 0,
                    MARKOV_HEAD_TOP_K,
                    " applies only to drafts with a Markov head that are loaded with draft_model(<draft folder>).");
    if (info.dflash_version == 2) {
        OPENVINO_ASSERT(info.markov_rank > 0 && info.markov_head_type == DFLASH2_MARKOV_HEAD,
                        "A DFlash v2 draft must have a '",
                        DFLASH2_MARKOV_HEAD,
                        "' Markov head. Re-export the draft with an optimum-intel version that exports ",
                        MARKOV_HEAD_MODEL_FILE_NAME,
                        ".");
    }
    if (info.markov_rank > 0) {
        OPENVINO_ASSERT(info.markov_head_type == VANILLA_MARKOV_HEAD ||
                            (info.markov_head_type == DFLASH2_MARKOV_HEAD && info.dflash_version == 2),
                        "Unsupported markov_head_type '",
                        info.markov_head_type,
                        "' for a DFlash v",
                        info.dflash_version,
                        " draft. Supported Markov heads: vanilla, and dflash2 for DFlash v2 drafts.");
        OPENVINO_ASSERT(info.markov_head_computation,
                        "The draft has a Markov head but no Markov head model. Load the draft with "
                        "draft_model(<draft folder>) so that ",
                        MARKOV_HEAD_MODEL_FILE_NAME,
                        " is read next to the draft backbone.");
        if (is_markov_step(info.markov_head_computation)) {
            validate_markov_step(info.markov_head_computation, info);
        } else {
            validate_candidate_scorer(info.markov_head_computation, info);
        }
    }

    return info;
}

void apply_dflash_gpu_compile_properties(const DFlashRTInfo& info,
                                         const std::string& device,
                                         bool inherits_target_properties,
                                         ov::AnyMap& compile_properties) {
    if (!info.dflash_mode || info.dflash_version != 2 || device.find("GPU") == std::string::npos) {
        return;
    }

    const auto property_name = ov::hint::activations_scale_factor.name();
    float requested_scale_factor = info.gpu_min_activations_scale_factor;
    if (auto requested_it = compile_properties.find(property_name); requested_it != compile_properties.end()) {
        requested_scale_factor = requested_it->second.as<float>();
        OPENVINO_ASSERT(std::isfinite(requested_scale_factor),
                        property_name,
                        " must be finite when compiling a DFlash v2 GPU draft, got ",
                        requested_scale_factor,
                        ".");
        OPENVINO_ASSERT(inherits_target_properties || requested_scale_factor >= info.gpu_min_activations_scale_factor,
                        "The DFlash v2 GPU draft ",
                        property_name,
                        " ",
                        requested_scale_factor,
                        " is below the minimum ",
                        info.gpu_min_activations_scale_factor,
                        " in the draft metadata; set at least the minimum or leave it unset.");
    }
    compile_properties[property_name] = std::max(requested_scale_factor, info.gpu_min_activations_scale_factor);
}

void read_markov_head(const std::filesystem::path& models_path, ov::AnyMap& properties) {
    std::optional<int64_t> top_k;
    if (auto top_k_it = properties.find(MARKOV_HEAD_TOP_K); top_k_it != properties.end()) {
        top_k = top_k_it->second.as<int64_t>();
        properties.erase(top_k_it);
    }
    auto rank_it = properties.find(DFLASH_MARKOV_RANK);
    const bool has_markov_head = rank_it != properties.end() && rank_it->second.as<size_t>() > 0;
    if (!has_markov_head) {
        OPENVINO_ASSERT(!top_k, MARKOV_HEAD_TOP_K, " applies only to drafts with a Markov head.");
        return;
    }
    const auto markov_head_path = models_path / MARKOV_HEAD_MODEL_FILE_NAME;
    OPENVINO_ASSERT(std::filesystem::exists(markov_head_path),
                    "The draft in ",
                    models_path,
                    " has a Markov head, but ",
                    MARKOV_HEAD_MODEL_FILE_NAME,
                    " is not next to it. Re-export the draft with an optimum-intel version that exports it.");
    select_markov_head_computation(utils::singleton_core().read_model(markov_head_path, {}, properties),
                                   top_k,
                                   properties);
}

void select_markov_head_computation(const std::shared_ptr<ov::Model>& markov_head_model,
                                    std::optional<int64_t> top_k,
                                    ov::AnyMap& properties) {
    constexpr const char* model_name = "Markov head model";
    const int64_t vocab_size = std::stoll(read_markov_head_rt_info(markov_head_model, model_name, "vocab_size"));
    const int64_t exported_top_k = std::stoll(read_markov_head_rt_info(markov_head_model, model_name, "top_k"));
    const int64_t selected_top_k = top_k.value_or(exported_top_k);
    OPENVINO_ASSERT(selected_top_k == TOP_K_FOR_MARKOV_STEP || (selected_top_k > 0 && selected_top_k <= vocab_size),
                    MARKOV_HEAD_TOP_K,
                    " must be ",
                    TOP_K_FOR_MARKOV_STEP,
                    " (dense Markov walk) or a K between 1 and the draft vocabulary size ",
                    vocab_size,
                    ", got ",
                    selected_top_k,
                    ".");

    const bool has_markov_step = has_port_named(markov_head_model->inputs(), "prev_token_ids");
    OPENVINO_ASSERT(selected_top_k != TOP_K_FOR_MARKOV_STEP || has_markov_step,
                    MARKOV_HEAD_TOP_K,
                    " ",
                    TOP_K_FOR_MARKOV_STEP,
                    " selects the dense Markov walk, which the '",
                    read_markov_head_rt_info(markov_head_model, model_name, "markov_head_type"),
                    "' Markov head does not have. Set a K between 1 and ",
                    vocab_size,
                    ".");

    // The computations share the input model's nodes, so work on a copy; constants keep sharing their data.
    const auto model = markov_head_model->clone();
    std::shared_ptr<ov::Model> computation;
    if (selected_top_k == TOP_K_FOR_MARKOV_STEP) {
        computation = keep_markov_head_computation(model, {"prev_token_ids"}, {"markov_bias"});
    } else {
        std::vector<std::string> scorer_inputs{"draft_logits", "anchor_token_ids", "subsequence_begins"};
        if (has_port_named(model->inputs(), "draft_hidden_states")) {
            scorer_inputs.emplace_back("draft_hidden_states");
        }
        computation = keep_markov_head_computation(model, scorer_inputs, {"edge_scores", "candidate_ids"});
        if (selected_top_k != exported_top_k) {
            replace_markov_head_top_k(computation, static_cast<size_t>(selected_top_k));
        }
        computation->set_rt_info(std::to_string(selected_top_k), {"dflash", "top_k"});
    }
    properties[DFLASH_MARKOV_HEAD_COMPUTATION] = computation;
}

bool is_markov_step(const std::shared_ptr<ov::Model>& markov_head_computation) {
    return has_port_named(markov_head_computation->inputs(), "prev_token_ids");
}

void validate_markov_step(const std::shared_ptr<ov::Model>& markov_step, const DFlashRTInfo& info) {
    constexpr const char* model_name = "Markov step";
    const size_t vocab_size = validate_markov_head_rt_info(markov_step, model_name, info);
    OPENVINO_ASSERT(markov_step->inputs().size() == 1 && markov_step->outputs().size() == 1,
                    "The Markov step must map 'prev_token_ids' to 'markov_bias'.");
    validate_markov_head_port(markov_step->input("prev_token_ids"), model_name, ov::element::i64, 1);
    const auto bias_shape =
        validate_markov_head_port(markov_step->output("markov_bias"), model_name, ov::element::f32, 2);
    OPENVINO_ASSERT(dimension_matches(bias_shape[1], vocab_size),
                    "The Markov step vocabulary size does not match its metadata.");
}

void validate_candidate_scorer(const std::shared_ptr<ov::Model>& candidate_scorer, const DFlashRTInfo& info) {
    constexpr const char* model_name = "candidate scorer";
    const size_t vocab_size = validate_markov_head_rt_info(candidate_scorer, model_name, info);
    const size_t top_k = std::stoull(read_markov_head_rt_info(candidate_scorer, model_name, "top_k"));
    OPENVINO_ASSERT(top_k > 0 && top_k <= vocab_size, "The candidate scorer top_k must be in [1, vocab_size].");
    if (top_k == 1) {
        GENAI_WARN(
            "The candidate scorer with top_k 1 drafts the backbone's best token at every position, so the "
            "Markov head has no effect. Set markov_head_top_k to a larger K, or to 0 for the dense Markov walk.");
    } else if (top_k < MIN_RECOMMENDED_MARKOV_HEAD_TOP_K) {
        GENAI_WARN("The candidate scorer keeps only %zu candidates per drafted position, so fewer draft tokens "
                   "may be accepted. Consider a markov_head_top_k of at least %zu.",
                   top_k,
                   MIN_RECOMMENDED_MARKOV_HEAD_TOP_K);
    }
    const bool gated = info.markov_head_type == DFLASH2_MARKOV_HEAD;
    OPENVINO_ASSERT(candidate_scorer->inputs().size() == (gated ? 4 : 3) && candidate_scorer->outputs().size() == 2,
                    "The candidate scorer must map 'draft_logits', ",
                    gated ? "'draft_hidden_states', " : "",
                    "'anchor_token_ids' and 'subsequence_begins' to 'edge_scores' and 'candidate_ids'.");
    const auto logits_shape =
        validate_markov_head_port(candidate_scorer->input("draft_logits"), model_name, ov::element::f32, 2);
    if (gated) {
        validate_markov_head_port(candidate_scorer->input("draft_hidden_states"), model_name, ov::element::f32, 2);
    }
    validate_markov_head_port(candidate_scorer->input("anchor_token_ids"), model_name, ov::element::i64, 1);
    validate_markov_head_port(candidate_scorer->input("subsequence_begins"), model_name, ov::element::i64, 1);
    const auto edge_scores_shape =
        validate_markov_head_port(candidate_scorer->output("edge_scores"), model_name, ov::element::f32, 3);
    const auto candidate_ids_shape =
        validate_markov_head_port(candidate_scorer->output("candidate_ids"), model_name, ov::element::i64, 2);
    OPENVINO_ASSERT(dimension_matches(logits_shape[1], vocab_size),
                    "The candidate scorer vocabulary size does not match its metadata.");
    OPENVINO_ASSERT(dimension_matches(edge_scores_shape[1], top_k) && dimension_matches(edge_scores_shape[2], top_k) &&
                        dimension_matches(candidate_ids_shape[1], top_k),
                    "The candidate scorer K dimensions do not match its top_k metadata.");
}

void attach_target_lm_head_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                    const std::shared_ptr<ov::Model>& draft_model,
                                    bool keep_hidden_state) {
    OPENVINO_ASSERT(main_model && draft_model, "DFlash lm_head graft requires both target and draft models.");

    auto target_head = std::get<0>(ov::genai::utils::find_llm_matmul(main_model));
    auto target_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(target_head);
    OPENVINO_ASSERT(target_matmul,
                    "DFlash: could not locate the target lm_head MatMul to graft onto the draft.");

    // Clone ONLY the weight side (input(1)); input(0) is the target activation and is not reused.
    // Cloning recursively carries any INT4 decompression subgraph intact.
    auto weight_source = target_matmul->input_value(1);
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight = eagle3::clone_node_recursive(weight_source.get_node_shared_ptr(), cloned_nodes);
    OPENVINO_ASSERT(cloned_weight, "DFlash: failed to clone the target lm_head weight subgraph.");
    auto cloned_weight_out = cloned_weight->output(weight_source.get_index());

    std::shared_ptr<ov::op::v0::Result> hidden_result;
    for (const auto& result : draft_model->get_results()) {
        if (result->output(0).get_names().count(LAST_HIDDEN_STATE_OUTPUT_NAME) != 0 ||
            result->get_friendly_name() == LAST_HIDDEN_STATE_OUTPUT_NAME) {
            hidden_result = result;
            break;
        }
    }
    OPENVINO_ASSERT(hidden_result,
                    "DFlash draft model must expose a 'last_hidden_state' output to graft the target lm_head.");

    // Feed the draft post-norm hidden into a bare MatMul with the target head weight + transpose flags.
    auto draft_hidden = hidden_result->input_value(0);
    auto logits_matmul = std::make_shared<ov::op::v0::MatMul>(draft_hidden,
                                                              cloned_weight_out,
                                                              target_matmul->get_transpose_a(),
                                                              target_matmul->get_transpose_b());
    logits_matmul->set_friendly_name("dflash_grafted_lm_head");

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits_matmul);
    logits_result->output(0).set_names({"logits"});
    logits_result->set_friendly_name("logits");
    logits_result->get_rt_info()["manually_added_output"] = true;

    draft_model->add_results({logits_result});
    if (!keep_hidden_state) {
        draft_model->remove_result(hidden_result);
    }
    draft_model->validate_nodes_and_infer_types();
}

void gather_last_hidden_state_with_logits(const std::shared_ptr<ov::Model>& draft_model) {
    OPENVINO_ASSERT(draft_model, "DFlash draft model cannot be null.");
    const auto matmul = std::get<0>(ov::genai::utils::find_llm_matmul(draft_model));
    OPENVINO_ASSERT(matmul, "DFlash: could not locate the draft lm_head MatMul.");
    const auto gathered_rows = matmul->input_value(0);
    OPENVINO_ASSERT(ov::as_type_ptr<ov::op::v8::Gather>(gathered_rows.get_node_shared_ptr()),
                    "DFlash: the draft lm_head must read the sampled_tokens_indices rows; apply the gather before "
                    "the lm_head first.");
    for (const auto& result : draft_model->get_results()) {
        if (result->output(0).get_names().count(LAST_HIDDEN_STATE_OUTPUT_NAME) != 0 ||
            result->get_friendly_name() == LAST_HIDDEN_STATE_OUTPUT_NAME) {
            result->input(0).replace_source_output(gathered_rows);
            draft_model->validate_nodes_and_infer_types();
            return;
        }
    }
    OPENVINO_THROW("DFlash draft model has no 'last_hidden_state' output to gather.");
}

void attach_target_embedding_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                      const std::shared_ptr<ov::Model>& draft_model,
                                      float input_embedding_scale) {
    OPENVINO_ASSERT(main_model && draft_model, "DFlash embedding attach requires both target and draft models.");

    auto gather = eagle3::find_embedding_gather(main_model);
    OPENVINO_ASSERT(gather,
                    "DFlash: could not find the target token-embedding Gather to attach onto the draft.");

    // Clone the embedding weight (input(0)). The recursive clone is precision-agnostic: it copies a
    // plain f16/f32 Constant or an int8/int4 dequant chain intact, so the draft owns its embedding.
    auto weight_source = gather->input_value(0);
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight = eagle3::clone_node_recursive(weight_source.get_node_shared_ptr(), cloned_nodes);
    OPENVINO_ASSERT(cloned_weight, "DFlash: failed to clone the target embedding weight subgraph.");
    auto cloned_weight_out = cloned_weight->output(weight_source.get_index());

    auto inputs_embeds_param = find_inputs_embeds_parameter(draft_model);
    OPENVINO_ASSERT(inputs_embeds_param,
                    "DFlash draft model must expose an 'inputs_embeds' input to attach the target embedding.");

    // Snapshot the live consumers of inputs_embeds before rewiring them onto the new Gather.
    auto original_consumers = get_live_consumers(draft_model, inputs_embeds_param->output(0));
    OPENVINO_ASSERT(!original_consumers.empty(), "DFlash draft inputs_embeds input has no live consumers.");

    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto embed_gather = std::make_shared<ov::op::v8::Gather>(cloned_weight_out, input_ids, axis);
    embed_gather->set_friendly_name("dflash_draft_embedding");
    auto embedding_output = scale_draft_embeddings(embed_gather, input_embedding_scale);

    for (auto consumer : original_consumers) {
        consumer.replace_source_output(embedding_output);
    }
    draft_model->add_parameters({input_ids});
    draft_model->remove_parameter(inputs_embeds_param);
    draft_model->validate_nodes_and_infer_types();
}

void scale_draft_inputs_embeds(const std::shared_ptr<ov::Model>& draft_model,
                               float input_embedding_scale) {
    OPENVINO_ASSERT(draft_model, "DFlash draft model cannot be null.");
    if (input_embedding_scale == 1.0f) {
        return;
    }
    auto inputs_embeds = find_inputs_embeds_parameter(draft_model);
    OPENVINO_ASSERT(inputs_embeds, "DFlash VLM draft model must expose an 'inputs_embeds' input.");
    auto original_consumers = get_live_consumers(draft_model, inputs_embeds->output(0));
    OPENVINO_ASSERT(!original_consumers.empty(), "DFlash VLM draft inputs_embeds input has no live consumers.");
    const auto scaled_embeddings = scale_draft_embeddings(inputs_embeds->output(0), input_embedding_scale);
    for (auto consumer : original_consumers) {
        consumer.replace_source_output(scaled_embeddings);
    }
    draft_model->validate_nodes_and_infer_types();
}

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
