// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ut_modules_base.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace module {

std::string ModuleTestBase::check_yaml(const std::string& yaml_content) {
    YAML::Node config = YAML::Load(yaml_content);

    OPENVINO_ASSERT(config["global_context"], "Test yaml config miss 'global_context'.");
    OPENVINO_ASSERT(config["global_context"]["model_type"], "Test yaml config miss 'model_type' in 'global_context'.");
    OPENVINO_ASSERT(config["pipeline_modules"], "Test yaml config miss 'pipeline_modules'.");
    return yaml_content;
}

ov::Tensor ModuleTestBase::ut_randn_tensor(const ov::Shape& shape, size_t seed) {
    ov::Tensor rand_tensor(ov::element::f32, shape);
    float* rand_tensor_data = rand_tensor.data<float>();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < rand_tensor.get_size(); ++i) {
        rand_tensor_data[i] = dist(rng);
    }

    return rand_tensor;
}

bool ModuleTestBase::compare_shape(const ov::Shape& shape1, const ov::Shape& shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }
    return true;
}

bool ModuleTestBase::compare_tensors(const ov::Tensor& output, const ov::Tensor& expected) {
    if (output.get_shape() != expected.get_shape() || output.get_element_type() != expected.get_element_type()) {
        return false;
    }
    size_t byte_size = output.get_byte_size();
    return std::memcmp(output.data(), expected.data(), byte_size) == 0;
}

bool ModuleTestBase::compare_big_tensor(const ov::Tensor& output,
                                        const std::vector<float>& expected_top,
                                        const float& thr) {
    int real_size = std::min(expected_top.size(), output.get_size());
    bool bresult = true;
    for (int i = 0; i < real_size; ++i) {
        float val = static_cast<float>(output.data<float>()[i]);
        if (std::fabs(val - expected_top[i]) > thr) {
            bresult = false;
            std::cout << "Mismatch at index " << i << ": expected " << expected_top[i] << ", got " << val << std::endl;
        }
    }
    return bresult;
}

bool ModuleTestBase::print_tensor_top(const ov::Tensor& tensor, size_t top_k) {
    if (tensor.get_element_type() != ov::element::f32) {
        std::cerr << "Only support printing float tensor" << std::endl;
        return false;
    }
    size_t real_size = std::min(tensor.get_size(), top_k);
    const float* data = tensor.data<float>();
    std::cout << "Tensor top " << real_size << " values: ";
    for (size_t i = 0; i < real_size; ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << std::endl;
    return true;
}

void ModuleTestBase::set_test_name(const std::string& test_name) {
    // replace / and space with _
    m_test_name = test_name;
    std::replace(m_test_name.begin(), m_test_name.end(), '/', '_');
    std::replace(m_test_name.begin(), m_test_name.end(), ' ', '_');
}

}  // namespace module
}  // namespace genai
}  // namespace ov