// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include <fstream>

#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"

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

int64_t argmax(const ov::Tensor& logits, const size_t batch_idx) {
    if (logits.get_shape()[0] <= batch_idx) {
        OPENVINO_THROW("logits batch size doesn't match the number of beams");
    }

    size_t vocab_size = logits.get_shape().back();
    size_t batch_offset = batch_idx * logits.get_shape()[1] * vocab_size;
    size_t sequence_offset = (logits.get_shape()[1] - 1) * vocab_size;
    const float* logits_data = logits.data<const float>() + batch_offset + sequence_offset;

    int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
    float max_logit = logits_data[out_token];

    return out_token;
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
 * Set position ids tensor data for next token inference based on provided attention mask
 * Supports multi batch
 * Supports sparse attention_mask
 */
void update_position_ids(ov::Tensor&& position_ids, const ov::Tensor&& attention_mask) {
    const size_t batch_size = attention_mask.get_shape().at(0);
    const size_t atten_length = attention_mask.get_shape().at(1);
    position_ids.set_shape({batch_size, 1});

    for (size_t batch = 0; batch < batch_size; batch++) {
        int64_t* start = attention_mask.data<int64_t>() + batch * atten_length;
        // todo: be careful with start + atten_length, probably need to replace with start + atten_length -1
        position_ids.data<int64_t>()[batch] = std::accumulate(start, start + atten_length, 0);
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
        }
    }
    return streamer;
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

/**
 * Split config by core and compile configs
 * There are not supported by `core.compile` function plugin options like `ENABLE_MMAP`
 * Move this options to `core.set_property` config
 */
std::pair<ov::AnyMap, ov::AnyMap> split_core_compile_config(const ov::AnyMap& properties) {
    const std::vector<std::string> unsupported_by_compile_properties{"ENABLE_MMAP"};
    ov::AnyMap core_properties;
    ov::AnyMap compile_properties{properties};

    for (const auto option : unsupported_by_compile_properties) {
        auto iter = properties.find(option);
        if (iter != properties.end()) {
            core_properties[option] = iter->second;
            compile_properties.erase(option);
        }
    }

    return {core_properties, compile_properties};
};

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

void slice_matmul_statefull_model(std::shared_ptr<ov::Model> model) {
    auto last_node = model->output(0).get_node()->input_value(0).get_node();
    ov::Node* matmul = dynamic_cast<ov::op::v0::MatMul*>(last_node);
    if (matmul) {
        // we have found matmul, do nothing
    } else if(auto add = dynamic_cast<ov::op::v1::Add*>(last_node)) {
        matmul = dynamic_cast<ov::op::v0::MatMul*>(add->input_value(0).get_node());
    } else if (auto transpose = dynamic_cast<ov::op::v1::Transpose*>(last_node)) {
        matmul = dynamic_cast<ov::op::v0::MatMul*>(transpose->input_value(0).get_node());
    } else if (auto multiply = dynamic_cast<ov::op::v1::Multiply*>(last_node)) {
        if (auto tanh = dynamic_cast<ov::op::v0::Tanh*>(multiply->input_value(0).get_node())) {
            if (auto divide = dynamic_cast<ov::op::v1::Divide*>(tanh->input_value(0).get_node())) {
                matmul = dynamic_cast<ov::op::v0::MatMul*>(divide->input_value(0).get_node());
            }
        }
    }

    if (matmul && matmul->input(0).get_partial_shape().rank().get_length() == 3) {
        auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-2});
        auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto slice = std::make_shared<ov::op::v8::Slice>(matmul->input_value(0), start, stop, step, axis);
        matmul->input(0).replace_source_output(slice);
    }
}

ov::Core singleton_core() {
    static ov::Core core;
    return core;
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
