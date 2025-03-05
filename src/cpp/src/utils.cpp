// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include <variant>
#include <fstream>
#include <memory>

#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/genai/text_streamer.hpp"


#include "sampler.hpp"

namespace ov {
namespace genai {
namespace utils {

Tensor init_attention_mask(const Tensor& input_ids) {
    auto shape = input_ids.get_shape();
    auto attention_mask = ov::Tensor{input_ids.get_element_type(), shape};
    std::fill_n(attention_mask.data<int64_t>(), shape[0] * shape[1], 1);
    return attention_mask;
}

void print_tensor(const ov::Tensor& tensor) {
    std::vector<int64_t> res;

    auto t_shape = tensor.get_shape();
    std::cout << "[";
    for (size_t i = 0; i < t_shape[0]; ++i) {
        std::cout << "|";
        for (size_t j = 0; j < t_shape[1]; ++j) {
            if (tensor.get_element_type() == ov::element::i64) {
                res.emplace_back(tensor.data<int64_t>()[t_shape[1] * i + j]);
                std::cout << tensor.data<int64_t>()[t_shape[1] * i + j] << " ";
            }
        }
        std::cout << "|";
    }
    std::cout << "]" << std::endl;
}

/**
 * Initializes position ids based on attention mask and starting position
 */
void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask, int64_t start_pos) {
    OPENVINO_ASSERT(position_ids.get_element_type() == ov::element::i64,
                    "position_ids tensor element type should be an i64");
    OPENVINO_ASSERT(position_ids.get_shape().size() == 2,
                    "position_ids tensor should of rank 2 with shape [batch_size, seq_len]");
    OPENVINO_ASSERT(attention_mask.get_element_type() == ov::element::i64,
                    "attention_mask tensor element type should be an i64");
    OPENVINO_ASSERT(attention_mask.get_shape().size() == 2,
                    "attention_mask tensor should of rank 2 with shape [batch_size, seq_len]");

    const size_t batch_size = attention_mask.get_shape()[0];
    const size_t seq_length = attention_mask.get_shape()[1];

    const int64_t* attention_mask_data = attention_mask.data<int64_t>();
    int64_t* position_ids_data = position_ids.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t sum = start_pos;
        for (size_t i = 0; i < seq_length; i++) {
            const size_t element_offset = batch * seq_length + i;
            position_ids_data[element_offset] = sum;
            if (attention_mask_data[element_offset] == 1) {
                sum += 1;
            }
        }
    }
}

void initialize_beam_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, ov::InferRequest& request) {
    request.set_tensor("input_ids", input_ids);
    request.set_tensor("attention_mask", attention_mask);

    ov::Shape input_shape = input_ids.get_shape();

    ov::Tensor position_ids = request.get_tensor("position_ids");
    position_ids.set_shape(input_shape);
    initialize_position_ids(position_ids, attention_mask);

    ov::Tensor beam_idx = request.get_tensor("beam_idx");
    beam_idx.set_shape({input_shape.at(0)});
    std::fill_n(beam_idx.data<int32_t>(), input_shape.at(0), 0);
}

void set_attention_mask(ov::Tensor&& attention_mask, std::vector<int32_t> next_beams) {
    ov::Tensor original_mask{ov::element::i64, attention_mask.get_shape()};
    ov::Shape original_shape = original_mask.get_shape();
    attention_mask.copy_to(original_mask);

    ov::Shape new_shape{next_beams.size(), original_mask.get_shape().at(1) + 1};
    attention_mask.set_shape(new_shape);

    for (size_t beam_id = 0; beam_id < next_beams.size(); beam_id++) {
        const size_t original_prompt_offset = next_beams.at(beam_id) * original_shape.at(1);
        const size_t result_prompt_offset = beam_id * new_shape.at(1);

        int64_t* dest = attention_mask.data<int64_t>() + result_prompt_offset;
        const int64_t* src = original_mask.data<int64_t>() + original_prompt_offset;

        std::memcpy(dest, src, original_shape.at(1) * sizeof(int64_t));
        attention_mask.data<int64_t>()[result_prompt_offset + new_shape.at(1) - 1] = 1;
    }
}

/**
 * Get attention mask tensor for next token inference
 * Supports multi batch
 * Supports sparse attention_mask
 */
ov::Tensor extend_attention(ov::Tensor attention_mask) {
    auto shape = attention_mask.get_shape();
    auto batch_size = shape[0];
    auto seq_len = shape[1];

    ov::Tensor new_atten_mask = ov::Tensor{attention_mask.get_element_type(), {batch_size, seq_len + 1}};
    auto old_data = attention_mask.data<int64_t>();
    auto new_data = new_atten_mask.data<int64_t>();
    for (size_t batch = 0; batch < batch_size; ++batch) {
        std::memcpy(new_data + batch * (seq_len + 1), old_data + batch * seq_len, seq_len * sizeof(int64_t));
        new_data[batch * (seq_len + 1) + seq_len] = 1;
    }
    return new_atten_mask;
}

ov::genai::StreamerVariant get_streamer_from_map(const ov::AnyMap& config_map) {
    ov::genai::StreamerVariant streamer = std::monostate();

    if (config_map.count(STREAMER_ARG_NAME)) {
        auto any_val = config_map.at(STREAMER_ARG_NAME);
        if (any_val.is<std::shared_ptr<ov::genai::StreamerBase>>()) {
            streamer = any_val.as<std::shared_ptr<ov::genai::StreamerBase>>();
        } else if (any_val.is<std::function<bool(std::string)>>()) {
            streamer = any_val.as<std::function<bool(std::string)>>();
        } else if (any_val.is<std::function<StreamingStatus(std::string)>>()) {
            streamer = any_val.as<std::function<StreamingStatus(std::string)>>();
        }
    }
    return streamer;
}

std::shared_ptr<StreamerBase> create_streamer(StreamerVariant streamer, Tokenizer tokenizer) {
    std::shared_ptr<StreamerBase> streamer_ptr = std::visit(overloaded{
        [](std::monostate) -> std::shared_ptr<StreamerBase> {
            return nullptr;
        },
        [](const std::shared_ptr<StreamerBase>& streamer) {
            return streamer;
        },
        [&tokenizer = tokenizer](const std::function<bool(std::string)>& streamer) -> std::shared_ptr<StreamerBase> {
            return std::make_unique<TextStreamer>(tokenizer, streamer);
        },
        [&tokenizer = tokenizer](const std::function<ov::genai::StreamingStatus(std::string)>& streamer) -> std::shared_ptr<StreamerBase> {
            return std::make_unique<TextStreamer>(tokenizer, streamer);
        }
    }, streamer);

    return streamer_ptr;
}

ov::genai::OptionalGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count(CONFIG_ARG_NAME))
        return config_map.at(CONFIG_ARG_NAME).as<ov::genai::GenerationConfig>();
    else
        return std::nullopt;
}

ProcessorConfig from_any_map(
    const ov::AnyMap& config_map,
    const ProcessorConfig& initial
) {
    auto iter = config_map.find("processor_config");
    ProcessorConfig extracted_config = config_map.end() != iter ?
        iter->second.as<ProcessorConfig>() : initial;
    using utils::read_anymap_param;
    read_anymap_param(config_map, "patch_size", extracted_config.patch_size);
    read_anymap_param(config_map, "scale_resolution", extracted_config.scale_resolution);
    read_anymap_param(config_map, "max_slice_nums", extracted_config.max_slice_nums);
    read_anymap_param(config_map, "norm_mean", extracted_config.norm_mean);
    read_anymap_param(config_map, "norm_std", extracted_config.norm_std);
    return extracted_config;
}

ov::genai::TokenizedInputs subtract_chat_tokenized_inputs(const ov::genai::TokenizedInputs& minuend, const ov::genai::TokenizedInputs& subtrahend) {
    auto minuend_size = minuend.input_ids.get_size();
    auto subtrahend_size = subtrahend.input_ids.get_size();
    ov::Shape new_shape{1, minuend_size - subtrahend_size};

    ov::Tensor new_input_ids(ov::element::i64, new_shape);
    auto data_ptr = minuend.input_ids.data<int64_t>();
    std::copy(data_ptr + subtrahend_size, data_ptr + minuend_size, new_input_ids.data<int64_t>());

    ov::Tensor new_attention_mask(ov::element::i64, new_shape);
    std::fill_n(new_attention_mask.data<int64_t>(), new_shape[1], 1);

    return {new_input_ids, new_attention_mask};
}

namespace {

bool has_op_with_type(const std::shared_ptr<const ov::Model>& function, const std::string& type_name) {
    for (const auto& op : function->get_ops()) {
        if (op->get_type_name() == type_name) {
            return true;
        }
    }
    return false;
}

std::tuple<std::shared_ptr<ov::Node>, int64_t> find_llm_matmul(const std::shared_ptr<ov::Model>& model) {
    auto last_node = model->output(0).get_node()->input_value(0).get_node_shared_ptr();
    std::shared_ptr<ov::Node> matmul = ov::as_type_ptr<ov::op::v0::MatMul>(last_node);

    // in case of PA all tokens are moved to batch dimension and we have to slice / gather accordingly
    const bool pa_based_model = has_op_with_type(model, "PagedAttentionExtension");
    int64_t slice_gather_dim = pa_based_model ? 0 : 1;

    // There are several patterns for matmul we are looking for:
    // Matmul -> Result
    // Matmul -> Add -> Result
    // Matmul -> Transpose -> Result
    // MatMul -> Divide -> Tanh -> Multiply -> Result
    if (!matmul) {
        if (auto add = ov::as_type_ptr<ov::op::v1::Add>(last_node)) {
            matmul = ov::as_type_ptr<ov::op::v0::MatMul>(add->input_value(0).get_node_shared_ptr());
        } else if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(last_node)) {
            matmul = ov::as_type_ptr<ov::op::v0::MatMul>(transpose->input_value(0).get_node_shared_ptr());
            auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr())->get_axis_vector_val();
            slice_gather_dim = order[slice_gather_dim];
        } else if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(last_node)) {
            if (auto tanh = ov::as_type_ptr<ov::op::v0::Tanh>(multiply->input_value(0).get_node_shared_ptr())) {
                if (auto divide = ov::as_type_ptr<ov::op::v1::Divide>(tanh->input_value(0).get_node_shared_ptr())) {
                    matmul = as_type_ptr<ov::op::v0::MatMul>(divide->input_value(0).get_node_shared_ptr());
                }
            }
        }
    }
    return std::make_tuple(matmul, slice_gather_dim);
}

} // namespace

void apply_slice_before_matmul_transformation(std::shared_ptr<ov::Model> model) {
    std::shared_ptr<ov::Node> matmul = nullptr;
    int64_t slice_gather_dim = -1;
    std::tie(matmul, slice_gather_dim) = find_llm_matmul(model);

    if (matmul && matmul->input(0).get_partial_shape().rank().get_length() == 3) {
        auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-2});
        auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{slice_gather_dim});
        auto slice = std::make_shared<ov::op::v8::Slice>(matmul->input_value(0), start, stop, step, axis);
        matmul->input(0).replace_source_output(slice);
    }
}

void apply_gather_before_matmul_transformation(std::shared_ptr<ov::Model> model) {
    std::shared_ptr<ov::Node> matmul = nullptr;
    int64_t slice_gather_dim = -1;
    std::tie(matmul, slice_gather_dim) = find_llm_matmul(model);

    if (matmul && matmul->input(0).get_partial_shape().rank().get_length() == 3) {
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
        indices->set_friendly_name("sampled_tokens_indices");
        indices->output(0).get_tensor().set_names({"sampled_tokens_indices"});
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{slice_gather_dim});
        auto gather = std::make_shared<ov::op::v8::Gather>(matmul->input_value(0), indices, axis);
        matmul->input(0).replace_source_output(gather);
        model->add_parameters({indices});
    }
}

ov::Core singleton_core() {
    static ov::Core core;
    return core;
}

void release_core_plugin(const std::string& device) {
    try {
        // Ugly WA to clean up oneDNN lru cache
        // dummy model created, compiled and cleaned for device (usefull for GPU only)
        {
            ov::PartialShape shape;
            auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
            const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(parameter)};
            ov::ParameterVector params = {parameter};
            auto dummy = std::make_shared<ov::Model>(results, params, "ParameterResult");

            auto compiled_model = stCore.compile_model(dummy, device);
            compiled_model.release_memory();
        }
        singleton_core().unload_plugin(device);
    } catch (const ov::Exception&) {
        // Note: in a theory it can throw an exception when 2 different pipelines are created from
        // different threads and then both of them unload plugin for 'device' from ov::Core
    }
}

size_t get_first_history_difference(const ov::Tensor& encoded_history, const std::vector<int64_t> tokenized_history) {
    size_t idx = 0;
    auto encoded_history_data = encoded_history.data<int64_t>();
    while(idx < encoded_history.get_size() && idx < tokenized_history.size()) {
        if (encoded_history_data[idx] != tokenized_history[idx])
            break;
        idx++;
    }

    return idx;
}

KVAxesPosition get_kv_axes_pos(std::shared_ptr<const ov::Model> model) {
    // sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
    // therefore usually seq_length_axis = 2 and batch = 0
    KVAxesPosition kv_pos { 0u, 2u };

    // "ReadValue" node is KV cache representation in stateful model
    std::string kv_node_type_name = std::string(ov::op::v6::ReadValue::get_type_info_static().name);

    for (const auto op : model->get_ops()) {
        // check input size, as in LoRA adapters case it could be 0
        if (op->get_type_name() != kv_node_type_name || op->get_input_size() < 1) {
            continue;
        }

        // Shape example: [-1,4,0,64]
        auto shape = op->get_input_partial_shape(0);

        for (size_t i = 0; i < shape.rank().get_length(); i++) {
            // Find axis = 0. This would be sequence length axis.
            if (shape[i] == 0) {
                kv_pos.seq_len = i;
            } else if (shape[i].is_dynamic()) {
                // Dynamic axis is a batch
                kv_pos.batch = i;
            }
        }
        break;
    }

    return kv_pos;
}

void trim_kv_cache(ov::InferRequest request, uint64_t remove_from_end, size_t seq_length_axis, std::optional<AdapterController> adapter_controller) {
    // nothing to trim in this case
    if (remove_from_end == 0)
        return;

    auto states = request.query_state();
    
    OPENVINO_ASSERT(states.size() > 0, "Request contains no states.");

    for (auto& state : states) {
        if(adapter_controller && adapter_controller->has_state_name(state.get_name()))
            continue;

        ov::Tensor old_tensor = state.get_state();
        // [BATCH_SIZE, num_kv_heads, seq_len, head_size]
        auto shape = old_tensor.get_shape();
        shape[seq_length_axis] -= remove_from_end;

        ov::Coordinate new_shape_begin{0, 0, 0, 0};
        ov::Coordinate new_shape_end{shape};

        auto trimmed_tensor = ov::Tensor(old_tensor, new_shape_begin, new_shape_end);

        ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
        trimmed_tensor.copy_to(new_tensor);

        state.set_state(new_tensor);
    }
}

ov::Tensor push_front_inputs(const ov::Tensor& base_tensor, int64_t add_to_front) {
    ov::Tensor new_tensor = ov::Tensor{ov::element::i64, {base_tensor.get_shape().at(0), base_tensor.get_shape().at(1) + 1}};
    auto new_tensor_data = new_tensor.data<int64_t>();
    new_tensor_data[0] = add_to_front;
    std::copy_n(base_tensor.data<int64_t>(), base_tensor.get_size(), new_tensor_data + 1);
    return new_tensor;
}

void print_compiled_model_properties(ov::CompiledModel& compiled_Model, const char* model_title) {
    // Specify the name of the environment variable
    const char* env_var_name = "OPENVINO_LOG_LEVEL";
    const char* env_var_value = std::getenv(env_var_name);

    // Check if the environment variable was found
    if (env_var_value != nullptr && atoi(env_var_value) > static_cast<int>(ov::log::Level::WARNING)) {
        // output of the actual settings that the device selected
        auto supported_properties = compiled_Model.get_property(ov::supported_properties);
        std::cout << "Model: " << model_title << std::endl;
        for (const auto& cfg : supported_properties) {
            if (cfg == ov::supported_properties)
                continue;
            auto prop = compiled_Model.get_property(cfg);
            if (cfg == ov::device::properties) {
                auto devices_properties = prop.as<ov::AnyMap>();
                for (auto& item : devices_properties) {
                    std::cout << "  " << item.first << ": " << std::endl;
                    for (auto& item2 : item.second.as<ov::AnyMap>()) {
                        std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
                    }
                }
            } else {
                std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
            }
        }

        ov::Core core;
        std::vector<std::string> exeTargets;
        exeTargets = compiled_Model.get_property(ov::execution_devices);
        std::cout << "EXECUTION_DEVICES:" << std::endl;
        for (const auto& device : exeTargets) {
            std::cout << " " << device << ": " << core.get_property(device, ov::device::full_name) << std::endl;
        }
    }
}

const ModelsMap::mapped_type& get_model_weights_pair(const ModelsMap& models_map, const std::string& key) {
    auto it = models_map.find(key);
    if (it != models_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Model with key '", key, "' not found in models map.");
}

std::pair<ov::AnyMap, SchedulerConfig> extract_scheduler_config(const ov::AnyMap& properties, std::optional<SchedulerConfig> default_config) {
    ov::AnyMap plugin_config = properties;
    auto it = plugin_config.find(ov::genai::scheduler_config.name());
    SchedulerConfig scheduler_config;
    if (it != plugin_config.end()) {
        scheduler_config = it->second.as<SchedulerConfig>();
        plugin_config.erase(it);
    } else if (default_config.has_value()) {
        scheduler_config = *default_config;
    }
    return {plugin_config, scheduler_config};
};

}  // namespace utils
}  // namespace genai
}  // namespace ov
