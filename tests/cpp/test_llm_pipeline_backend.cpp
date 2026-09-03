// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"

namespace {

struct BackendModelParam {
    std::string model_id;
    bool has_kvcache;
    bool has_linear;
};

std::vector<BackendModelParam> load_cache_types_csv(const std::string& path) {
    std::vector<BackendModelParam> entries;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "WARNING: cannot open cache types CSV: " << path << "\n";
        return entries;
    }

    std::string line;
    while (std::getline(file, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }

        std::istringstream ss(line);
        std::string model_id;
        std::string kv_str;
        std::string lin_str;
        if (!std::getline(ss, model_id, ',') ||
            !std::getline(ss, kv_str, ',') ||
            !std::getline(ss, lin_str, ',')) {
            continue;
        }
        entries.push_back({model_id, kv_str == "true", lin_str == "true"});
    }
    return entries;
}

std::vector<BackendModelParam> get_csv_params() {
    const char* csv_env = std::getenv("CACHE_TYPES_CSV");
    if (csv_env) {
        return load_cache_types_csv(csv_env);
    }
    std::cerr << "WARNING: CACHE_TYPES_CSV is not set, skipping LLM backend routing tests.\n";
    return {};
}

std::filesystem::path get_model_dir(const BackendModelParam& param) {
    const char* base_dir = std::getenv("TEST_MODELS_BASE_DIR");
    if (!base_dir) {
        return {};
    }

    const std::string model_name = param.model_id.substr(param.model_id.rfind('/') + 1);
    return std::filesystem::path(base_dir) / model_name;
}

bool is_model_available(const std::filesystem::path& model_dir) {
    if (model_dir.empty()) {
        return false;
    }
    if (!std::filesystem::exists(model_dir / "openvino_model.xml")) {
        return false;
    }
    return true;
}

std::string make_test_name(const ::testing::TestParamInfo<BackendModelParam>& info) {
    std::string name = info.param.model_id.substr(info.param.model_id.rfind('/') + 1);
    for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
            c = '_';
        }
    }
    return name;
}

}  // namespace

class LLMPipelineBackendRealModel : public ::testing::TestWithParam<BackendModelParam> {};

TEST_P(LLMPipelineBackendRealModel, ExplicitSdpaBypassesDefaultPa) {
    const auto& param = GetParam();
    const std::filesystem::path model_dir = get_model_dir(param);
    if (!is_model_available(model_dir)) {
        GTEST_SKIP() << "Model not found, skipping: " << model_dir;
    }

    ov::AnyMap properties;
    properties["ATTENTION_BACKEND"] = ov::genai::SDPA_BACKEND;
    EXPECT_NO_THROW({
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(model_dir, "CPU", properties);
    }) << param.model_id;
}

TEST_P(LLMPipelineBackendRealModel, DefaultBackendInitializesWithFallback) {
    const auto& param = GetParam();
    const std::filesystem::path model_dir = get_model_dir(param);
    if (!is_model_available(model_dir)) {
        GTEST_SKIP() << "Model not found, skipping: " << model_dir;
    }

    EXPECT_NO_THROW({
        auto pipe = std::make_unique<ov::genai::LLMPipeline>(model_dir, "CPU");
    }) << param.model_id;
}

INSTANTIATE_TEST_SUITE_P(
    CsvModels,
    LLMPipelineBackendRealModel,
    ::testing::ValuesIn(get_csv_params()),
    make_test_name);
