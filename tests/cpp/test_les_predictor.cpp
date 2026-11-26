// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <set>

#include "les_predictor/les_predictor.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using namespace ov::op;

constexpr size_t HIDDEN = 8;
constexpr size_t EXPERTS = 6;
constexpr size_t ROUTER_K = 2;
constexpr size_t NUM_LAYERS = 3;  // layer 0 dense, layers 1 and 2 MoE

struct HeadData {
    std::vector<float> weight, previous_weight, bias;  // [E, H], [E, H], [E]
};

std::vector<float> random_vector(size_t size, std::mt19937& rng, float offset = 0.0f) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(size);
    for (auto& x : v) {
        x = dist(rng) + offset;
    }
    return v;
}

void write_safetensors(const std::filesystem::path& path,
                       const std::map<size_t, HeadData>& heads,
                       bool previous_weight = true) {
    std::string header = "{";
    std::string data;
    auto add = [&](const std::string& name, const std::string& dtype, const std::vector<size_t>& shape, const void* ptr,
                   size_t bytes) {
        std::string dims;
        for (size_t d : shape) {
            dims += (dims.empty() ? "" : ",") + std::to_string(d);
        }
        header += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[" + dims + "],\"data_offsets\":[" +
                  std::to_string(data.size()) + "," + std::to_string(data.size() + bytes) + "]},";
        data.append(static_cast<const char*>(ptr), bytes);
    };
    const int64_t mode = 2;
    for (const auto& [layer, h] : heads) {
        const std::string prefix = "expert_predictors." + std::to_string(layer + 1) + ".";
        add(prefix + "weight", "F32", {EXPERTS, HIDDEN}, h.weight.data(), h.weight.size() * sizeof(float));
        if (previous_weight) {
            add(prefix + "previous_weight", "F32", {EXPERTS, HIDDEN}, h.previous_weight.data(), h.previous_weight.size() * sizeof(float));
        }
        add(prefix + "selection_bias", "F32", {EXPERTS}, h.bias.data(), h.bias.size() * sizeof(float));
        add(prefix + "input_mode_code", "I64", {}, &mode, sizeof(mode));
    }
    header.back() = '}';
    while (header.size() % 8) {
        header += ' ';
    }
    std::ofstream out(path, std::ios::binary);
    const uint64_t header_size = header.size();
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out << header << data;
}

// Router inputs norm_i = x * gamma_i named like the PyTorch frontend names post_attention_layernorm, MoE routers
// MatMul -> Sigmoid -> Add -> TopK on layers 1 and 2. `paged`: x is [T, 1, H] and the model has subsequence_begins.
std::shared_ptr<ov::Model> make_model(bool paged, const std::vector<std::vector<float>>& gammas, std::mt19937& rng) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32,
                                             paged ? ov::PartialShape{-1, 1, HIDDEN} : ov::PartialShape{-1, -1, HIDDEN});
    x->output(0).set_names({"x"});
    ov::ParameterVector parameters{x};
    if (paged) {
        auto begins = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        begins->output(0).set_names({"subsequence_begins"});
        parameters.push_back(begins);
    } else {
        auto beam_idx = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        beam_idx->output(0).set_names({"beam_idx"});
        parameters.push_back(beam_idx);
    }
    ov::ResultVector results;
    for (size_t i = 0; i < NUM_LAYERS; ++i) {
        const std::string scope = "__module.model.layers." + std::to_string(i);
        auto gamma = v0::Constant::create(ov::element::f32, {HIDDEN}, gammas[i]);
        auto norm = std::make_shared<v1::Multiply>(x, gamma);
        norm->set_friendly_name(scope + ".post_attention_layernorm/aten::mul/Multiply_1");
        if (i == 0) {
            results.push_back(std::make_shared<v0::Result>(norm));
            continue;
        }
        auto flat = std::make_shared<v1::Reshape>(norm, v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{-1, int64_t(HIDDEN)}), false);
        flat->set_friendly_name(scope + ".mlp/aten::view/Reshape");
        auto w = v0::Constant::create(ov::element::f32, {EXPERTS, HIDDEN}, random_vector(EXPERTS * HIDDEN, rng));
        auto logits = std::make_shared<v0::MatMul>(flat, w, false, true);
        auto scores = std::make_shared<v0::Sigmoid>(logits);
        auto bias = v0::Constant::create(ov::element::f32, {EXPERTS}, random_vector(EXPERTS, rng));
        auto topk = std::make_shared<v11::TopK>(std::make_shared<v1::Add>(scores, bias),
                                                v0::Constant::create(ov::element::i64, {}, {int64_t(ROUTER_K)}),
                                                -1, ov::op::TopKMode::MAX, ov::op::TopKSortType::SORT_VALUES,
                                                ov::element::i64);
        topk->set_friendly_name(scope + ".mlp.gate/aten::topk/TopK");
        results.push_back(std::make_shared<v0::Result>(topk->output(1)));
    }
    return std::make_shared<ov::Model>(results, parameters);
}

std::set<int32_t> reference_experts(const HeadData& h, const float* cur, const std::vector<float>& prev, size_t k) {
    std::vector<float> selection(EXPERTS);
    const float max_bias = *std::max_element(h.bias.begin(), h.bias.end());
    for (size_t e = 0; e < EXPERTS; ++e) {
        float logit = 0.0f;
        for (size_t j = 0; j < HIDDEN; ++j) {
            logit += h.weight[e * HIDDEN + j] * cur[j] + h.previous_weight[e * HIDDEN + j] * prev[j];
        }
        selection[e] = 1.0f / (1.0f + std::exp(-logit)) + h.bias[e] - max_bias;
    }
    std::vector<int32_t> order(EXPERTS);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return selection[a] > selection[b]; });
    return {order.begin(), order.begin() + k};
}

std::set<int32_t> row_set(const ov::Tensor& t, size_t row, size_t k) {
    const int32_t* d = t.data<const int32_t>() + row * k;
    return {d, d + k};
}

class LESPredictorTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::mt19937 rng(7);
        for (size_t i = 0; i < NUM_LAYERS; ++i) {
            gammas.push_back(random_vector(HIDDEN, rng, 1.0f));
        }
        for (size_t layer : {1, 2}) {
            // a large common bias offset, as in MiniCPM5-MoE (the GenAI graph subtracts the maximum)
            heads[layer] = {random_vector(EXPERTS * HIDDEN, rng), random_vector(EXPERTS * HIDDEN, rng),
                            random_vector(EXPERTS, rng, 200.0f)};
        }
        path = std::filesystem::temp_directory_path() /
               ("les_predictor_test_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
        write_safetensors(path, heads);
        model_rng.seed(11);
    }
    void TearDown() override {
        std::filesystem::remove(path);
    }

    std::vector<std::vector<float>> gammas;
    std::map<size_t, HeadData> heads;
    std::filesystem::path path;
    std::mt19937 model_rng;
};

}  // namespace

TEST_F(LESPredictorTest, ReadsPredictorFile) {
    ov::genai::LESPredictor predictor(path);
    ASSERT_TRUE(predictor);
    const auto& impl = *predictor.get_impl();
    EXPECT_EQ(impl.hidden_size, HIDDEN);
    EXPECT_EQ(impl.num_experts, EXPERTS);
    ASSERT_EQ(impl.heads.size(), 2u);
    EXPECT_TRUE(impl.heads.count(1) && impl.heads.count(2));  // expert_predictors.2 / .3 -> MoE layers 1 / 2
}

TEST_F(LESPredictorTest, AnalyzeKeepsModel) {
    auto model = make_model(false, gammas, model_rng);
    const size_t num_ops = model->get_ops().size();
    const ov::genai::LESPredictor predictor(path);
    const auto info = ov::genai::les::analyze(model, predictor);
    EXPECT_EQ(info.layers, (std::vector<size_t>{1, 2}));
    EXPECT_EQ(info.top_k, ROUTER_K);
    EXPECT_EQ(model->get_ops().size(), num_ops);
    EXPECT_EQ(model->get_results().size(), NUM_LAYERS);
}

TEST_F(LESPredictorTest, PredictorModelMatchesReference) {
    auto model = make_model(true, gammas, model_rng);
    const ov::genai::LESPredictor predictor(path);
    const auto info = ov::genai::les::analyze(model, predictor);
    ASSERT_EQ(info.top_k, ROUTER_K);  // the heads predict as many experts as the router selects
    auto predictor_model = ov::genai::les::build_predictor_model(predictor, info);
    ASSERT_EQ(predictor_model->get_parameters().size(), 4u);
    ASSERT_EQ(predictor_model->get_results().size(), 2u);

    auto request = ov::Core().compile_model(predictor_model, "CPU").create_infer_request();
    std::mt19937 rng(5);
    constexpr size_t tokens = 3;
    std::map<size_t, std::vector<float>> cur, prev;
    for (size_t layer : info.layers) {
        cur[layer] = random_vector(tokens * HIDDEN, rng);
        prev[layer] = random_vector(HIDDEN, rng);
        const std::string suffix = std::to_string(layer);
        request.set_tensor(ov::genai::les::MODEL_HIDDEN_PREFIX + suffix, ov::Tensor(ov::element::f32, {tokens, HIDDEN}, cur[layer].data()));
        request.set_tensor(ov::genai::les::MODEL_PREV_HIDDEN_PREFIX + suffix, ov::Tensor(ov::element::f32, {1, HIDDEN}, prev[layer].data()));
    }
    request.infer();
    for (size_t layer : info.layers) {
        auto predicted = request.get_tensor(ov::genai::les::MODEL_OUTPUT_PREFIX + std::to_string(layer));
        ASSERT_EQ(predicted.get_shape(), (ov::Shape{tokens, ROUTER_K}));
        for (size_t t = 0; t < tokens; ++t) {
            EXPECT_EQ(row_set(predicted, t, ROUTER_K),
                      reference_experts(heads[layer], cur[layer].data() + t * HIDDEN, prev[layer], ROUTER_K))
                << "layer " << layer << " token " << t;
        }
    }
}

TEST_F(LESPredictorTest, ExtractsPropertyFromPath) {
    ov::AnyMap properties{{"les_predictor", path.string()}, {"CACHE_DIR", ""}};
    ov::genai::utils::SharedOptional<const ov::AnyMap> props(&properties);
    std::optional<ov::genai::LESPredictor> predictor;
    ASSERT_TRUE(ov::genai::les::extract_from_properties(props, predictor));
    ASSERT_TRUE(predictor && *predictor);
    EXPECT_EQ(props->count("les_predictor"), 0u);
    EXPECT_EQ(props->count("CACHE_DIR"), 1u);
}

TEST_F(LESPredictorTest, ExtractsPropertyFromObject) {
    const ov::genai::LESPredictor original(path);
    ov::AnyMap properties{ov::genai::les_predictor(original)};
    ov::genai::utils::SharedOptional<const ov::AnyMap> props(&properties);
    std::optional<ov::genai::LESPredictor> predictor;
    ASSERT_TRUE(ov::genai::les::extract_from_properties(props, predictor));
    ASSERT_TRUE(predictor && *predictor);
    EXPECT_EQ(predictor->get_impl(), original.get_impl());
    EXPECT_TRUE(props->empty());
}

TEST_F(LESPredictorTest, PropertyFromPathReadsFile) {
    ov::AnyMap properties{ov::genai::les_predictor(path)};
    ASSERT_TRUE(properties.at("les_predictor").is<ov::genai::LESPredictor>());
    ov::genai::utils::SharedOptional<const ov::AnyMap> props(&properties);
    std::optional<ov::genai::LESPredictor> predictor;
    ASSERT_TRUE(ov::genai::les::extract_from_properties(props, predictor));
    ASSERT_TRUE(predictor && *predictor);
    EXPECT_EQ(predictor->get_impl()->heads.size(), 2u);
}

TEST_F(LESPredictorTest, ExtractsPropertyFromFilesystemPath) {
    ov::AnyMap properties{{"les_predictor", path}};  // as Python passes pathlib.Path
    ov::genai::utils::SharedOptional<const ov::AnyMap> props(&properties);
    std::optional<ov::genai::LESPredictor> predictor;
    ASSERT_TRUE(ov::genai::les::extract_from_properties(props, predictor));
    ASSERT_TRUE(predictor && *predictor);
    EXPECT_TRUE(props->empty());
}

TEST_F(LESPredictorTest, RejectsOtherPropertyTypes) {
    ov::AnyMap properties{{"les_predictor", 5}};
    ov::genai::utils::SharedOptional<const ov::AnyMap> props(&properties);
    std::optional<ov::genai::LESPredictor> predictor;
    EXPECT_THROW(ov::genai::les::extract_from_properties(props, predictor), ov::Exception);
}

TEST_F(LESPredictorTest, RequiresPreviousWeight) {
    write_safetensors(path, heads, /*previous_weight=*/false);
    EXPECT_THROW(ov::genai::LESPredictor{path}, ov::Exception);
}
