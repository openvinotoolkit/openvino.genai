// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "utils.hpp"

using namespace ov::genai::utils;

struct CacheTypesModelParam {
    std::string model_id;  // HuggingFace model id (e.g. "org/name")
    bool expected_kvcache;
    bool expected_linear;
};

static std::vector<CacheTypesModelParam> load_cache_types_csv(const std::string& path) {
    std::vector<CacheTypesModelParam> entries;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "WARNING: cannot open cache types CSV: " << path << "\n";
        return entries;
    }
    std::string line;
    while (std::getline(file, line)) {
        // Strip leading whitespace
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        std::istringstream ss(line);
        std::string model_id, kv_str, lin_str;
        if (!std::getline(ss, model_id, ',') || !std::getline(ss, kv_str, ',') || !std::getline(ss, lin_str, ',')) {
            continue;
        }
        entries.push_back({model_id, kv_str == "true", lin_str == "true"});
    }
    return entries;
}

class GetCacheTypesRealModel : public ::testing::TestWithParam<CacheTypesModelParam> {};

TEST_P(GetCacheTypesRealModel, MatchesExpected) {
    const char* base_dir = std::getenv("TEST_MODELS_BASE_DIR");
    if (!base_dir) {
        GTEST_SKIP() << "TEST_MODELS_BASE_DIR not set, skipping real-model tests";
    }

    const auto& param = GetParam();

    // Derive local directory name from the model id: keep the part after the last '/'
    const std::string model_name = param.model_id.substr(param.model_id.rfind('/') + 1);
    const std::string xml_path = std::string(base_dir) + "/" + model_name + "/openvino_model.xml";

    if (!std::filesystem::exists(xml_path)) {
        GTEST_SKIP() << "Model not found, skipping: " << xml_path;
    }

    ov::Core core;
    const auto model = core.read_model(xml_path);
    const auto types = get_cache_types(*model);

    EXPECT_EQ(types.has_kvcache(), param.expected_kvcache) << param.model_id << ": unexpected has_kvcache";
    EXPECT_EQ(types.has_linear(), param.expected_linear) << param.model_id << ": unexpected has_linear";
}

static std::vector<CacheTypesModelParam> get_csv_params() {
    const char* csv_env = std::getenv("CACHE_TYPES_CSV");
    if (csv_env) {
        return load_cache_types_csv(csv_env);
    }
    std::cerr << "WARNING: CACHE_TYPES_CSV is not set, skipping cache-types tests.\n";
    return {};
}

INSTANTIATE_TEST_SUITE_P(
    CsvModels,
    GetCacheTypesRealModel,
    ::testing::ValuesIn(get_csv_params()),
    [](const ::testing::TestParamInfo<CacheTypesModelParam>& info) {
        // Sanitize model name for use as a gtest test-case label
        std::string name = info.param.model_id.substr(info.param.model_id.rfind('/') + 1);
        for (char& c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)))
                c = '_';
        }
        return name;
    }
);
