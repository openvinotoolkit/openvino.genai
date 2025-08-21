#include "add_second_input_pass.hpp"
#include <openvino/pass/manager.hpp>
#include <openvino/opsets/opset15.hpp>
#include <openvino/op/util/op_types.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/select.hpp>
#include <openvino/op/equal.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/multiply.hpp>

#include <iostream>
#include <memory>
#include <vector>

using namespace ov;
using namespace ov::opset15;
using namespace ov::genai;

template <typename T = int>
std::shared_ptr<Constant> make_constant(const std::vector<T>& vals, element::Type_t type = element::i32) {
    return std::make_shared<Constant>(element::from<T>(), Shape{vals.size()}, vals);
}

template <typename T>
std::shared_ptr<Constant> make_constant(T val, element::Type_t type = element::i32) {
    return std::make_shared<Constant>(element::from<T>(), Shape{}, std::vector<T>{val});
}


bool AddSecondInputPass::parse_inputs(std::shared_ptr<ov::Node>& combine_seg) {
    size_t num_segments = combine_seg->get_input_size() / 3;

    std::vector<int> input_signature(num_segments, 0);
    std::vector<ov::Output<ov::Node>> inputs;

    // We go through the inputs of the CombineSegments node and check if they are
    // either Constant or Sequence. If begin is Constant, we check if the corresponding
    // end input is also Constant.
    // If begin is Sequence, we check if input has truncation operation.
    for (size_t i = 0; i < num_segments; ++i) {
        auto begin_input = combine_seg->input_value(3 * i);
        auto end_input = combine_seg->input_value(3 * i + 1);
        auto data_input = combine_seg->input_value(3 * i + 2);

        if (std::dynamic_pointer_cast<Constant>(begin_input.get_node_shared_ptr())) {
            // Constant input
            if (!std::dynamic_pointer_cast<Constant>(data_input.get_node_shared_ptr())) {
                return false;
            }
            auto const_node = std::dynamic_pointer_cast<Constant>(data_input.get_node_shared_ptr());
            input_signature[i] = const_node->get_vector<int>()[0];
            inputs.push_back(begin_input);
            inputs.push_back(end_input);
            inputs.push_back(data_input);
        } else {
            // Sequence input
            input_signature[i] = -1;
            auto trunc_node = begin_input.get_node_shared_ptr();
            if (std::string(trunc_node->get_type_name()) != "Truncate") {
                return false;
            }
            auto trunc_inputs = trunc_node->input_values();
            for (size_t j = 0; j < 3 && j < trunc_inputs.size(); ++j) {
                inputs.push_back(trunc_inputs[j]);
            }
            
            for (size_t i = 3; i < trunc_inputs.size(); ++i) {
                this->m_trunc_values.push_back(trunc_inputs[i]);
            }
        }
    }
    this->m_inputs = inputs;
    this->m_input_signature = input_signature;
    return true;  // means successfully parsed inputs
}

bool AddSecondInputPass::parse_and_assert_postprocessor(const std::shared_ptr<ov::Model>& model) {
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    if (model->get_rt_info().count(PROCESSED_POST_PROCESSOR_NAME) == 0) {
        m_pass_errors << "Could not add second input. Post processor is not present in the model." << std::endl;
        return false;
    }
    auto rt_info_value = model->get_rt_info().at(PROCESSED_POST_PROCESSOR_NAME);
    std::string json_str = rt_info_value.as<std::string>();

    nlohmann::json post_processor = nlohmann::json::parse(json_str);

    if (!post_processor.contains("pair")) {
        m_pass_errors << "Could not add second input. post_processor does not contain input signature for paired input" << std::endl;
        return false;
    }

    if (post_processor["single"]["ids"].get<std::vector<int>>() != m_input_signature) {
        m_pass_errors << "Could not add second input. Input signature from rt_info does not match to the CombineSegments node inputs." << std::endl;
        return false;
    }

    auto pair_ids = post_processor["pair"]["ids"].get<std::vector<int>>();
    if (std::vector<int>(pair_ids.begin(), pair_ids.begin() + m_input_signature.size()) != m_input_signature) {
        m_pass_errors << "Could not add second input. Paired inputs are allowed only when it's widening the single input." << std::endl;
        return false;
    }

    int num_main_inputs = std::count_if(pair_ids.begin(), pair_ids.end(), [](int v) { return v <= -1; });
    if (num_main_inputs != 2) {
        m_pass_errors << "Could not add second input. Only 2 inputs are allowed for the paired input" << std::endl;
        return false;
    }

    auto single_ids = post_processor["single"]["ids"].get<std::vector<int>>();
    int single_neg_count = std::count_if(single_ids.begin(), single_ids.end(), [](int v) { return v <= -1; });
    if (single_neg_count != 1) {
        m_pass_errors << "Could not add second input. There should be exactly one sequence input in the single signature." << std::endl;
        return false;
    }

    this->m_post_processor = post_processor;
    return true;
}

void AddSecondInputPass::insert_splits() {
    // Find the index of the first sequence input (-1)
    auto input_signature = this->m_input_signature;
    
    auto it = std::find(input_signature.begin(), input_signature.end(), -1);
    if (it == input_signature.end()) {
        m_pass_errors << "No sequence input found in input_signature" << std::endl;
        return;
    }
    size_t first_input_idx = std::distance(input_signature.begin(), it);

    auto begin = m_inputs[3 * first_input_idx];
    auto end = m_inputs[3 * first_input_idx + 1];
    auto data = m_inputs[3 * first_input_idx + 2];

    // Create two new Parameters for paired input
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> new_parameters;
    for (int i = 0; i < 2; ++i) {
        auto param = std::make_shared<ov::op::v0::Parameter>(element::string, PartialShape{-1});
        param->set_friendly_name("string_input_" + std::to_string(i + 1));
        new_parameters.push_back(param);
    }
    this->m_new_parameters = new_parameters;

    auto param_1_shape = std::make_shared<ShapeOf>(new_parameters[0], element::i32);
    auto param_2_shape = std::make_shared<ShapeOf>(new_parameters[1], element::i32);
    auto total_size = std::make_shared<ShapeOf>(begin.get_node_shared_ptr(), element::i32);

    // For the first input begins_1/ends_1, it's a slice till the Parameter_1 shape.
    auto begins_1 = std::make_shared<Slice>(begin, make_constant({0}), param_1_shape, make_constant({1}));
    auto ends_1 = std::make_shared<Slice>(end, make_constant({0}), param_1_shape, make_constant({1}));

    // If the second input is empty we need to slice at least one element.
    // If we don't do that input with shape [0] could not be specified together with input with shape [1] 
    // in Select and broadcasted. This garbage values will be zeroed in Select.
    auto one_const = make_constant({1});
    auto second_start = std::make_shared<Minimum>(
        std::make_shared<Subtract>(total_size, param_2_shape),
        std::make_shared<Subtract>(total_size, one_const)
    );

    // Slice for second input
    std::shared_ptr<ov::Node> begins_2 = std::make_shared<Slice>(begin, second_start, total_size, make_constant({1}));
    std::shared_ptr<ov::Node> ends_2 = std::make_shared<Slice>(end, second_start, total_size, make_constant({1}));

    // If inputs_2 is empty, we need to zero the second dimension of the broadcasted begins and ends tensors.
    auto zero_const = make_constant({0});
    this->m_equal_node = std::make_shared<Equal>(param_2_shape, zero_const);
    begins_2 = std::make_shared<Select>(m_equal_node, zero_const, begins_2);
    // TODO: For correct behavior, 'ends_2' should be set to 1 when the second input is empty.
    // However, due to bug CSV-160624 (incorrect handling of empty second input in Select and broadcast),
    // we temporarily set it to zero as a workaround. 
    // The following line to use 1 instead of 0 is kept for future reference: once bug CSV-160624 is resolved
    // auto one_const = make_constant({1});
    ends_2 = std::make_shared<Select>(m_equal_node, zero_const, ends_2);

    // Broadcast shapes
    auto broadcasted_shape = std::make_shared<Maximum>(param_1_shape, param_2_shape);

    OutputVector first_input = {
        std::make_shared<Broadcast>(begins_1, broadcasted_shape),
        std::make_shared<Broadcast>(ends_1, broadcasted_shape),
        data
    };
    OutputVector second_input = {
        std::make_shared<Broadcast>(begins_2, broadcasted_shape),
        std::make_shared<Broadcast>(ends_2, broadcasted_shape),
        data
    };

    // Extend signature and adjust truncation
    auto pair_ids = m_post_processor["pair"]["ids"].get<std::vector<int>>();
    std::vector<int> signature_to_extend(pair_ids.begin() + input_signature.size(), pair_ids.end());

    // Adjust truncation values
    auto& trunc_values = this->m_trunc_values;
    if (!trunc_values.empty()) {
        auto trunc_const_max_len = std::dynamic_pointer_cast<Constant>(trunc_values[0].get_node_shared_ptr());
        if (trunc_const_max_len) {
            int trunc_adj = trunc_const_max_len->get_vector<int>()[0] - (static_cast<int>(signature_to_extend.size()) - 1);
            trunc_values[0] = std::make_shared<Constant>(element::i32, Shape{}, std::vector<int32_t>{trunc_adj});
        }
    }

    // Create Truncate node
    std::vector<ov::Output<ov::Node>> truncate_inputs;
    truncate_inputs.insert(truncate_inputs.end(), first_input.begin(), first_input.end());
    truncate_inputs.insert(truncate_inputs.end(), second_input.begin(), second_input.end());
    for (auto& t : trunc_values) {
        truncate_inputs.push_back(t);
    }

    auto trunc = m_node_factory("Truncate", truncate_inputs, {});
    this->m_first_input = {trunc[0], trunc[1], trunc[2]};
    this->m_second_input = {trunc[3], trunc[4], trunc[5]};
}

std::vector<ov::Output<ov::Node>> AddSecondInputPass::get_new_inputs() {
    // This part of the code is responsible for creating new inputs for the CombineSegments node.
    // It combines inputs for the first and second input, and adds special tokens for the second input.
    // The new inputs are then returned as a vector.
    auto inputs = this->m_inputs;
    auto input_signature = this->m_input_signature;
    auto first_input = this->m_first_input;
    auto second_input = this->m_second_input;

    std::vector<ov::Output<ov::Node>> new_inputs = inputs;
    // Replace original input with first_input
    auto it = std::find(input_signature.begin(), input_signature.end(), -1);
    if (it == input_signature.end()) {
        m_pass_errors << "No sequence input found in input_signature" << std::endl;
        return {};
    }
    size_t first_input_idx = std::distance(input_signature.begin(), it);
    std::copy(first_input.begin(), first_input.end(), new_inputs.begin() + 3 * first_input_idx);

    // Extend signature with additional inputs
    auto pair_ids = m_post_processor["pair"]["ids"].get<std::vector<int>>();
    std::vector<int> signature_to_extend(pair_ids.begin() + input_signature.size(), pair_ids.end());
    for (auto value : signature_to_extend) {
        if (value <= -1) {
            // Only one additional input is possible
            new_inputs.insert(new_inputs.end(), second_input.begin(), second_input.end());
            continue;
        }

        auto added_spec_begins = make_constant(0);
        auto added_spec_ends = make_constant(1);
        auto added_spec_data = make_constant({value});

        // Nullify special tokens constant if ends for sequence_2 is nullified
        auto select_node = std::make_shared<Select>(
            m_equal_node,
            make_constant({0}),
            make_constant({1})
        );
        auto multiplied_ends = std::make_shared<Multiply>(added_spec_ends, select_node);

        new_inputs.insert(new_inputs.end(), {
            added_spec_begins,
            multiplied_ends,
            added_spec_data
        });
    }
    if (!m_post_processor["pair"].contains("type_ids")) {
        m_pass_errors << "Could not add second input. post_processor does not contain 'type_ids' for paired input" << std::endl;
        return {};
    }
    auto type_ids = m_post_processor["pair"]["type_ids"].get<std::vector<int>>();
    auto new_segment_ids = make_constant(type_ids);
    new_inputs.push_back(new_segment_ids);

    return new_inputs;
}

bool AddSecondInputPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    auto parameters = model->get_parameters();
    if (parameters.size() != 1) {
        m_pass_errors << "Model must have only one input.\n";
        return false;
    }

    std::shared_ptr<Node> combine_seg = nullptr;
    for (const auto& node : model->get_ops()) {
        if (!std::strcmp(node->get_type_name(), "CombineSegments")) {
            combine_seg = node;
            break;
        }
    }

    if (!combine_seg) {
        m_pass_errors << "CombineSegments node not found.\n";
        return false;
    }

    if (!parse_inputs(combine_seg)) {
        return false;
    }

    if (!parse_and_assert_postprocessor(model)) {
        return false;
    }

    // Insert splits for begins, ends before the CombineSegments node and return pair of new inputs.
    // Also adds a modified Truncate
    insert_splits();

    // Get new inputs for the CombineSegments node, which combine
    // the first and second input, and add special tokens for the second input.
    auto new_inputs = get_new_inputs();
    if (new_inputs.empty()) {
        return false;
    }

    // Replace the CombineSegments node with a new one that takes the pair of inputs
    auto new_combine_segments = m_node_factory("CombineSegments", new_inputs, {})[0].get_node_shared_ptr();
    ov::replace_node(combine_seg, new_combine_segments);

    // Find StringTensorUnpack node connected to the model input
    auto target_inputs = parameters[0]->output(0).get_target_inputs();
    if (target_inputs.empty()) {
        m_pass_errors << "No target inputs found for model parameter." << std::endl;
        return false;
    }
    auto str_unpack = target_inputs.begin()->get_node();
    if (std::string(str_unpack->get_type_name()) != "StringTensorUnpack") {
        m_pass_errors << "Expected node type 'StringTensorUnpack', but got '" << str_unpack->get_type_name() << "'.\n";
        return false;
    }

    // Concatenate new parameters for input
    auto new_input = std::make_shared<Concat>(ov::OutputVector{m_new_parameters[0], m_new_parameters[1]}, 0);

    str_unpack->input(0).replace_source_output(new_input->output(0));

    model->replace_parameter(0, m_new_parameters[0]);
    model->add_parameters({m_new_parameters[1]});

    return true;
}
