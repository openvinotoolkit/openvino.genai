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
#include "add_second_input_transformation.hpp"

#include <iostream>
#include <memory>
#include <vector>

using namespace ov;
using namespace ov::opset15;

std::shared_ptr<Constant> make_constant(std::vector<int> vals, element::Type_t type = element::i32) {
    return std::make_shared<Constant>(type, Shape{vals.size()}, vals);
}


class ModifyCombineSegmentsForPairInput: public ov::pass::ModelPass {
public:

bool parse_inputs(std::shared_ptr<ov::Node>& node) {
    size_t num_segments = combine_seg->get_input_size() / 3;
    
    std::vector<int> input_signature(num_segments, 0);
    std::vector<ov::Output<ov::Node>> inputs;

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
            // If you need to store trunc_values, add a member variable and assign here
            // Example: this->trunc_values.assign(trunc_inputs.begin() + 3, trunc_inputs.end());
        }
    }
    // If you need to store inputs and input_signature as member variables, assign here
    // Example: this->inputs = inputs; this->input_signature = input_signature;
    return true;

}

// bool ModifyCombineSegmentsForPairInput::run_on_model(const std::shared_ptr<ov::Model>& model) {
bool run_on_model(const std::shared_ptr<ov::Model>& model) override {

    auto parameters = model->get_parameters();
    if (parameters.size() != 1) {
        std::cerr << "Model must have only one input.\n";
        return false;
    }
    
    // input_signature = [[]] * num_segments;


    std::shared_ptr<Node> combine_seg = nullptr;
    for (const auto& node : model->get_ops()) {
        if (!std::strcmp(node->get_type_name(), "CombineSegments")) {
            combine_seg = node;
            break;
        }
    }

    if (!combine_seg) {
        std::cerr << "CombineSegments node not found.\n";
        return false;
    }

    // Assume CombineSegments has at least 3 inputs (begin, end, data) for first sequence
    auto begin = combine_seg->input_value(0);
    auto end = combine_seg->input_value(1);
    auto data = combine_seg->input_value(2);

    // Create two new parameters for paired input
    auto input1 = std::make_shared<Parameter>(element::string, PartialShape{-1});
    input1->set_friendly_name("string_input_1");
    auto input2 = std::make_shared<Parameter>(element::string, PartialShape{-1});
    input2->set_friendly_name("string_input_2");

    auto shape1 = std::make_shared<ShapeOf>(input1, element::i32);
    auto shape2 = std::make_shared<ShapeOf>(input2, element::i32);
    auto total_shape = std::make_shared<ShapeOf>(begin, element::i32);

    // First slice
    auto begins_1 = std::make_shared<Slice>(begin, make_constant({0}), shape1, make_constant({1}));
    auto ends_1 = std::make_shared<Slice>(end, make_constant({0}), shape1, make_constant({1}));

    auto min_const = make_constant({1});
    auto one_const = make_constant({1});
    auto zero_const = make_constant({0});

    auto start_2 = std::make_shared<Minimum>(
        std::make_shared<Subtract>(total_shape, shape2),
        std::make_shared<Subtract>(total_shape, one_const)
    );

    auto begins_2 = std::make_shared<Slice>(begin, start_2, total_shape, make_constant({1}));
    auto ends_2 = std::make_shared<Slice>(end, start_2, total_shape, make_constant({1}));

    // Select to zero out if input2 is empty
    auto is_empty = std::make_shared<Equal>(shape2, zero_const);
    // begins_2 = std::make_shared<Select>(is_empty, zero_const, begins_2);
    // ends_2 = std::make_shared<Select>(is_empty, zero_const, ends_2);

    auto max_shape = std::make_shared<Maximum>(shape1, shape2);

    auto b1 = std::make_shared<Broadcast>(begins_1, max_shape);
    auto e1 = std::make_shared<Broadcast>(ends_1, max_shape);
    auto b2 = std::make_shared<Broadcast>(begins_2, max_shape);
    auto e2 = std::make_shared<Broadcast>(ends_2, max_shape);

    // Combine new inputs
    OutputVector new_inputs = {
        b1, e1, data,
        b2, e2, data,
        // Add a dummy segment id = 1 for the second part (as constant)
        make_constant({1})
    };

    auto new_combine = combine_seg->clone_with_new_inputs(new_inputs);
    // ov::replace_node(combine_seg, new_combine);

    // Replace parameter in model
    model->replace_parameter(0, input1);
    model->add_parameters({input2});

    return true;
}

};
