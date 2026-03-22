// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/op_extension.hpp>
#include <openvino/openvino.hpp>

#include "openvino/genai/extensions.hpp"
#include "utils.hpp"

using namespace ov::genai::utils;

namespace {
class CustomAdd : public ov::op::Op {
public:
    OPENVINO_OP("CustomAdd", "extension");

    CustomAdd() = default;

    CustomAdd(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) : ov::op::Op({lhs, rhs}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const ov::element::Type input_type_0 = get_input_element_type(0);
        const ov::element::Type input_type_1 = get_input_element_type(1);
        if (!input_type_0.compatible(input_type_1)) {
            throw std::runtime_error("CustomAdd input element types must match");
        }

        set_output_type(0, input_type_0, get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        if (new_args.size() != 2) {
            throw std::runtime_error("CustomAdd expects exactly 2 new inputs");
        }
        return std::make_shared<CustomAdd>(new_args.at(0), new_args.at(1));
    }

    bool visit_attributes(ov::AttributeVisitor&) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        std::cout << "custom add evaluate\n";
        if (inputs.size() != 2) {
            throw std::runtime_error("CustomAdd expects exactly 2 inputs");
        }
        if (outputs.size() != 1) {
            throw std::runtime_error("CustomAdd expects exactly 1 output");
        }

        outputs[0].set_shape(inputs[0].get_shape());

        const ov::element::Type element_type = inputs[0].get_element_type();
        if (element_type != inputs[1].get_element_type()) {
            throw std::runtime_error("CustomAdd input element types must match");
        }

        if (element_type == ov::element::f32) {
            return evaluate_typed<float>(outputs[0], inputs[0], inputs[1]);
        }
        if (element_type == ov::element::f16) {
            return evaluate_typed<ov::float16>(outputs[0], inputs[0], inputs[1]);
        }

        throw std::runtime_error("CustomAdd unsupported element type");
    }

    bool has_evaluate() const override {
        return true;
    }

private:
    template <typename T>
    static bool evaluate_typed(ov::Tensor& output, const ov::Tensor& lhs, const ov::Tensor& rhs) {
        const T* lhs_data = lhs.data<const T>();
        const T* rhs_data = rhs.data<const T>();
        T* out_data = output.data<T>();

        const size_t elements_count = ov::shape_size(output.get_shape());
        for (size_t index = 0; index < elements_count; ++index) {
            out_data[index] = lhs_data[index] + rhs_data[index];
        }
        return true;
    }
};
}  // namespace

TEST(TestAddExtensions, test_extract_extensions) {
    ov::AnyMap properties_path = {ov::genai::extensions(std::vector<std::filesystem::path>{"non_existent_path"})};
    std::shared_ptr<ov::Extension> op = std::make_shared<ov::OpExtension<CustomAdd>>();
    ov::AnyMap properties_op = {ov::genai::extensions(std::vector<std::shared_ptr<ov::Extension>>{op})};
    ov::genai::ExtensionList extension_path{"non_existent_path"};
    ov::genai::ExtensionList extension_op{op};

    EXPECT_EQ(extract_extensions(properties_path), extension_path);
    EXPECT_EQ(extract_extensions(properties_op), extension_op);
}

TEST(TestAddExtensions, test_extract_extensions_to_core) {
    // Use intentionally non-existent, platform-agnostic extension paths to trigger error handling.
    ov::AnyMap properties_path = {ov::genai::extensions(std::vector<std::filesystem::path>{"non_existent_path"})};
    ov::AnyMap properties_custom_op = {ov::genai::extensions(
        std::vector<std::shared_ptr<ov::Extension>>{std::make_shared<ov::OpExtension<CustomAdd>>()})};

    EXPECT_THROW(extract_extensions_to_core(properties_path), ov::Exception);
    EXPECT_NO_THROW(extract_extensions_to_core(properties_custom_op));
}
