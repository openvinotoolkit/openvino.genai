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

#include "openvino/genai/llm_pipeline.hpp"
#include "llm/pipeline_base.hpp"
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

void skip_if_model_unavailable(const std::filesystem::path& model_dir) {
    if (model_dir.empty()) {
        GTEST_SKIP() << "TEST_MODELS_BASE_DIR not set, skipping real-model tests";
    }
    if (!std::filesystem::exists(model_dir / "openvino_model.xml")) {
        GTEST_SKIP() << "Model not found, skipping: " << model_dir;
    }
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

const std::string& get_expected_default_backend(const BackendModelParam& param) {
    if (!param.has_kvcache) {
        return ov::genai::SDPA_BACKEND;
    }
    if (param.has_linear &&
        param.model_id.find("tiny-random-lfm2") == std::string::npos &&
        param.model_id.find("tiny-random-qwen3-next") == std::string::npos) {
        return ov::genai::SDPA_BACKEND;
    }
    return ov::genai::PA_BACKEND;
}

}  // namespace

class DummyLLMPipelineImpl final : public ov::genai::LLMPipelineImplBase {
public:
    DummyLLMPipelineImpl() : ov::genai::LLMPipelineImplBase(ov::genai::Tokenizer(), ov::genai::GenerationConfig()) {}

    ov::genai::DecodedResults generate(
        ov::genai::StringInputs,
        ov::genai::OptionalGenerationConfig,
        ov::genai::StreamerVariant) override {
        return {};
    }

    ov::genai::DecodedResults generate(
        const ov::genai::ChatHistory&,
        ov::genai::OptionalGenerationConfig,
        ov::genai::StreamerVariant) override {
        return {};
    }

    ov::genai::EncodedResults generate(
        const ov::genai::EncodedInputs&,
        ov::genai::OptionalGenerationConfig,
        ov::genai::StreamerVariant) override {
        return {};
    }

    void start_chat(const std::string&) override {}
    void finish_chat() override {}
};

TEST(LLMPipelineBackendProperty, DefaultsToSdpaAndCanBeUpdated) {
    DummyLLMPipelineImpl impl;

    EXPECT_EQ(impl.get_attention_backend(), ov::genai::SDPA_BACKEND);

    impl.set_attention_backend(ov::genai::PA_BACKEND);

    EXPECT_EQ(impl.get_attention_backend(), ov::genai::PA_BACKEND);
}

class LLMPipelineBackendRealModel : public ::testing::TestWithParam<BackendModelParam> {};

TEST_P(LLMPipelineBackendRealModel, ExplicitSdpaBypassesDefaultPa) {
    const auto& param = GetParam();
    const std::filesystem::path model_dir = get_model_dir(param);
    skip_if_model_unavailable(model_dir);

    ov::AnyMap properties;
    properties["ATTENTION_BACKEND"] = ov::genai::SDPA_BACKEND;
    ov::genai::LLMPipeline pipe(model_dir, "CPU", properties);

    EXPECT_EQ(pipe.get_attention_backend(), ov::genai::SDPA_BACKEND) << param.model_id;
}

TEST_P(LLMPipelineBackendRealModel, DefaultSelectsExpectedBackend) {
    const auto& param = GetParam();
    const std::filesystem::path model_dir = get_model_dir(param);
    skip_if_model_unavailable(model_dir);

    ov::genai::LLMPipeline pipe(model_dir, "CPU");

    EXPECT_EQ(pipe.get_attention_backend(), get_expected_default_backend(param)) << param.model_id;
}

INSTANTIATE_TEST_SUITE_P(
    CsvModels,
    LLMPipelineBackendRealModel,
    ::testing::ValuesIn(get_csv_params()),
    make_test_name);
