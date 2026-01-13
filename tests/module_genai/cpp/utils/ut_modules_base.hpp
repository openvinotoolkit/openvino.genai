// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <openvino/genai/module_genai/pipeline.hpp>
#include <string>
#include <vector>
#include <random>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "utils.hpp"
#include "load_image.hpp"

class ModuleTestBase {
public:
    virtual ~ModuleTestBase() = default;

    void run() {
        std::string yaml_content = generate_yaml_content();
        if (check_env_variable("DUMP_YAML")) {
            std::string filename = "dumped_" + m_test_name + ".yaml";
            std::ofstream out(filename);
            out << yaml_content;
            out.close();
            std::cout << "Saved YAML content to " << filename << std::endl;
        }

        ov::genai::module::ModulePipeline pipe(yaml_content);

        ov::AnyMap inputs = prepare_inputs();
        if (m_async) {
            pipe.generate_async(inputs);
        } else {
            pipe.generate(inputs);
        }

        check_outputs(pipe);
    }

    static ov::Tensor ut_randn_tensor(const ov::Shape& shape, size_t seed);

protected:
    std::string m_test_name;
    void set_test_name(const std::string& test_name);
#ifndef REGISTER_TEST_NAME
#    define REGISTER_TEST_NAME()                                                                                      \
        set_test_name(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name() + std::string("_") + \
                      ::testing::UnitTest::GetInstance()->current_test_info()->name());
#endif

    bool m_async = false;

    virtual std::string get_yaml_content() = 0;
    virtual ov::AnyMap prepare_inputs() = 0;
    virtual void check_outputs(ov::genai::module::ModulePipeline& pipe) = 0;

    bool compare_tensors(const ov::Tensor& output, const ov::Tensor& expected);

    bool compare_big_tensor(const ov::Tensor& output, const std::vector<float>& expected_top, const float& thr = 1e-3);

    template <typename T>
    bool compare_big_tensor(const ov::Tensor& output, const std::vector<T>& expected_top) {
        int real_size = std::min(expected_top.size(), output.get_size());
        bool bresult = true;
        for (int i = 0; i < real_size; ++i) {
            T val = static_cast<T>(output.data<T>()[i]);
            if (val != expected_top[i]) {
                bresult = false;
                std::cout << "Mismatch at index " << i << ": expected " << expected_top[i] << ", got " << val
                          << std::endl;
            }
        }

        return bresult;
    }

    template<typename T>
    bool compare_big_tensor(const ov::Tensor& output, const ov::Tensor& expected, const float& thr = 1e-3) {
        if (output.get_shape() != expected.get_shape() || output.get_element_type() != expected.get_element_type()) {
            return false;
        }
        size_t real_size = std::min(expected.get_size(), output.get_size());
        bool bresult = true;
        for (size_t i = 0; i < real_size; ++i) {
            T val = static_cast<T>(output.data<T>()[i]);
            T exp_val = static_cast<T>(expected.data<T>()[i]);
            if (std::fabs(val - exp_val) > thr) {
                bresult = false;
                break;
            }
        }
        return bresult;
    }

    bool compare_shape(const ov::Shape& shape1, const ov::Shape& shape2);

    std::string check_yaml(const std::string& yaml_content);

    std::filesystem::path generate_yaml_path() {
        std::string yaml_content = check_yaml(get_yaml_content());
        std::string filename = "temp_" + m_test_name + ".yaml";
        std::ofstream out(filename);
        out << yaml_content;
        out.close();
        return std::filesystem::path(filename);
    }

    std::string generate_yaml_content() {
        std::string yaml_content = check_yaml(get_yaml_content());
        return yaml_content;
    }
};
