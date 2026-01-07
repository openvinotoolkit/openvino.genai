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

#include "utils.hpp"
#include "load_image.hpp"
#include <yaml-cpp/yaml.h>

#ifndef CHECK
#    define CHECK(cond, msg)                                                   \
        do {                                                                   \
            if (!(cond)) {                                                     \
                throw std::runtime_error(std::string("Check failed: ") + msg); \
            }                                                                  \
        } while (0)
#endif

class ModuleTestBase {
public:
    virtual ~ModuleTestBase() = default;

    void run() {
#define EANBEL_YAML_CONTEXT 1
#if EANBEL_YAML_CONTEXT
        std::string yaml_content = generate_yaml_content();
        ov::genai::module::ModulePipeline pipe(yaml_content);
#else
        std::filesystem::path config_path = generate_yaml_path();
        ov::genai::module::ModulePipeline pipe(config_path);
#endif

        ov::AnyMap inputs = prepare_inputs();
        pipe.generate(inputs);

        verify_outputs(pipe);

// Cleanup
#ifndef EANBEL_YAML_CONTEXT
        if (std::filesystem::exists(config_path)) {
            std::filesystem::remove(config_path);
        }
#endif
    }

protected:
    std::string m_test_name;
    virtual std::string get_yaml_content() = 0;
    virtual ov::AnyMap prepare_inputs() = 0;
    virtual void verify_outputs(ov::genai::module::ModulePipeline& pipe) = 0;

    bool compare_tensors(const ov::Tensor& output, const ov::Tensor& expected);

    template <typename T>
    bool compare_big_tensor(const ov::Tensor& output, const std::vector<T>& expected_top, const float& thr = 0) {
        int real_size = std::min(expected_top.size(), output.get_size());
        bool bresult = true;
        for (int i = 0; i < real_size; ++i) {
            T val = static_cast<T>(output.data<T>()[i]);
            float expected_flt = static_cast<float>(expected_top[i]);
            float val_flt = static_cast<float>(val);
            if (std::fabs(val_flt - expected_flt) > thr) {
                bresult = false;
                std::cout << "Mismatch at index " << i << ": expected " << expected_flt << ", got " << val_flt
                          << std::endl;
            }
        }

        return bresult;
    }

    bool compare_shape(const ov::Shape& shape1, const ov::Shape& shape2);

    ov::Tensor ut_randn_tensor(const ov::Shape& shape, size_t seed);

private:
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

#ifndef DEFINE_MODULE_TEST_CONSTRUCTOR
#    define DEFINE_MODULE_TEST_CONSTRUCTOR(CLASS_NAME) \
        CLASS_NAME() = delete;                         \
        CLASS_NAME(const std::string& test_name) {     \
            m_test_name = test_name;                   \
        }
#endif

class TestRegistry {
public:
    using Creator = std::function<std::shared_ptr<ModuleTestBase>()>;

    static TestRegistry& get() {
        static TestRegistry instance;
        return instance;
    }

    void register_test(const std::string& name, Creator creator) {
        tests[name] = creator;
    }

    const std::map<std::string, Creator>& get_tests() const {
        return tests;
    }

private:
    std::map<std::string, Creator> tests;
};

struct TestRegistrar {
    TestRegistrar(const std::string& name, TestRegistry::Creator creator) {
        TestRegistry::get().register_test(name, creator);
    }
};

#define REGISTER_MODULE_TEST(CLASS_NAME)                                                               \
    static TestRegistrar registrar_##CLASS_NAME(#CLASS_NAME, []() -> std::shared_ptr<ModuleTestBase> { \
        return std::make_shared<CLASS_NAME>(#CLASS_NAME);                                                         \
    });
