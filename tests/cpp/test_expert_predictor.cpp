// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <set>

#include "expert_predictor/expert_predictor.hpp"
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

void write_safetensors(const std::filesystem::path& path, const std::map<size_t, HeadData>& heads) {
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
        add(prefix + "previous_weight", "F32", {EXPERTS, HIDDEN}, h.previous_weight.data(), h.previous_weight.size() * sizeof(float));
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

std::vector<float> norm_row(const std::vector<float>& x, size_t token, const std::vector<float>& gamma) {
    std::vector<float> r(HIDDEN);
    for (size_t j = 0; j < HIDDEN; ++j) {
        r[j] = x[token * HIDDEN + j] * gamma[j];
    }
    return r;
}

class ExpertPredictorTest : public ::testing::Test {
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
        path = std::filesystem::temp_directory_path() / ("expert_predictor_test_" + std::to_string(::getpid()) + ".safetensors");
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

TEST_F(ExpertPredictorTest, ReadsPredictorFile) {
    ov::genai::ExpertPredictor predictor(path);
    ASSERT_TRUE(predictor);
    const auto& impl = *predictor.get_impl();
    EXPECT_EQ(impl.hidden_size, HIDDEN);
    EXPECT_EQ(impl.num_experts, EXPERTS);
    ASSERT_EQ(impl.heads.size(), 2u);
    EXPECT_TRUE(impl.heads.count(1) && impl.heads.count(2));  // expert_predictors.2 / .3 -> MoE layers 1 / 2
}

TEST_F(ExpertPredictorTest, StatefulUsesPreviousInferenceState) {
    auto model = make_model(false, gammas, model_rng);
    const ov::genai::ExpertPredictor predictor(path);
    const auto info = ov::genai::les::apply(model, predictor, ov::genai::les::GraphMode::STATEFUL);
    ASSERT_EQ(info.layers, (std::vector<size_t>{1, 2}));
    EXPECT_EQ(info.top_k, ROUTER_K);

    auto request = ov::Core().compile_model(model, "CPU").create_infer_request();
    std::mt19937 rng(3);
    std::vector<float> prev_last[NUM_LAYERS];
    for (auto& p : prev_last) {
        p.assign(HIDDEN, 0.0f);  // reset state
    }
    for (size_t seq_len : {3, 1, 1}) {  // prefill, then two decode steps
        auto x = random_vector(seq_len * HIDDEN, rng);
        request.set_tensor("x", ov::Tensor(ov::element::f32, {1, seq_len, HIDDEN}, x.data()));
        request.set_tensor("beam_idx", ov::Tensor(ov::element::i32, {1}, std::vector<int32_t>{0}.data()));
        request.infer();
        for (size_t layer : info.layers) {
            auto predicted = request.get_tensor(ov::genai::les::PREDICTED_EXPERTS_PREFIX + std::to_string(layer));
            ASSERT_EQ(predicted.get_shape(), (ov::Shape{1, seq_len, ROUTER_K}));
            for (size_t t = 0; t < seq_len; ++t) {
                auto cur = norm_row(x, t, gammas[layer - 1]);
                // every token of an inference uses the router input saved at the last token of the previous one
                const auto& prev = prev_last[layer];
                EXPECT_EQ(row_set(predicted, t, ROUTER_K), reference_experts(heads[layer], cur.data(), prev, ROUTER_K))
                    << "layer " << layer << " token " << t << " seq_len " << seq_len;
            }
            prev_last[layer] = norm_row(x, seq_len - 1, gammas[layer]);
        }
    }
}

TEST_F(ExpertPredictorTest, PagedKeepsPreviousTokenPerSequence) {
    auto model = make_model(true, gammas, model_rng);
    const ov::genai::ExpertPredictor predictor(path, /*top_k=*/3);
    const auto info = ov::genai::les::apply(model, predictor, ov::genai::les::GraphMode::PAGED);
    EXPECT_EQ(info.top_k, 3u);
    auto request = ov::Core().compile_model(model, "CPU").create_infer_request();
    ov::genai::les::PagedState state(info);

    std::mt19937 rng(5);
    std::map<uint64_t, std::map<size_t, std::vector<float>>> last;  // sequence -> layer -> router input of last token
    // step 1: sequence 10 prefills 3 tokens, sequence 20 prefills 2 tokens; step 2: both decode, in swapped order;
    // step 3: sequence 20 finished, sequence 30 is new
    const std::vector<std::vector<ov::genai::les::ScheduledSequence>> steps = {
        {{10, 3}, {20, 2}}, {{20, 1}, {10, 1}}, {{10, 1}, {30, 2}}};
    for (const auto& sequences : steps) {
        size_t total = 0;
        std::vector<int32_t> begins{0};
        std::unordered_set<uint64_t> alive;
        for (const auto& s : sequences) {
            total += s.num_tokens;
            begins.push_back(static_cast<int32_t>(total));
            alive.insert(s.id);
        }
        auto x = random_vector(total * HIDDEN, rng);
        request.set_tensor("x", ov::Tensor(ov::element::f32, {total, 1, HIDDEN}, x.data()));
        request.set_tensor("subsequence_begins", ov::Tensor(ov::element::i32, {begins.size()}, begins.data()));
        state.set_inputs(request, sequences);
        request.infer();

        for (size_t layer : info.layers) {
            auto predicted = request.get_tensor(ov::genai::les::PREDICTED_EXPERTS_PREFIX + std::to_string(layer));
            ASSERT_EQ(predicted.get_shape(), (ov::Shape{total, 3}));
            for (size_t s = 0; s < sequences.size(); ++s) {
                for (size_t t = begins[s]; t < size_t(begins[s + 1]); ++t) {
                    auto cur = norm_row(x, t, gammas[layer - 1]);
                    // every token of a sequence uses the router input cached at its last token of the previous step
                    std::vector<float> prev(HIDDEN, 0.0f);
                    if (last.count(sequences[s].id)) {
                        prev = last[sequences[s].id][layer];
                    }
                    EXPECT_EQ(row_set(predicted, t, 3), reference_experts(heads[layer], cur.data(), prev, 3))
                        << "layer " << layer << " sequence " << sequences[s].id << " row " << t;
                }
            }
        }
        state.update(request, sequences, alive);
        for (size_t s = 0; s < sequences.size(); ++s) {
            for (size_t layer : info.layers) {
                last[sequences[s].id][layer] = norm_row(x, begins[s + 1] - 1, gammas[layer]);
            }
        }
    }
}

TEST_F(ExpertPredictorTest, ExtractsPropertyFromPath) {
    ov::AnyMap properties{{"expert_predictor", path.string()}, {"CACHE_DIR", ""}};
    ov::genai::utils::SharedOptional<const ov::AnyMap> props(&properties);
    std::optional<ov::genai::ExpertPredictor> predictor;
    ASSERT_TRUE(ov::genai::les::extract_from_properties(props, predictor));
    ASSERT_TRUE(predictor && *predictor);
    EXPECT_EQ(predictor->get_top_k(), 0u);
    EXPECT_TRUE(predictor->get_use_previous_token());
    EXPECT_EQ(props->count("expert_predictor"), 0u);
    EXPECT_EQ(props->count("CACHE_DIR"), 1u);
}

TEST_F(ExpertPredictorTest, ExtractsPropertyFromObject) {
    ov::AnyMap properties{ov::genai::expert_predictor(ov::genai::ExpertPredictor(path, 5, false))};
    ov::genai::utils::SharedOptional<const ov::AnyMap> props(&properties);
    std::optional<ov::genai::ExpertPredictor> predictor;
    ASSERT_TRUE(ov::genai::les::extract_from_properties(props, predictor));
    EXPECT_EQ(predictor->get_top_k(), 5u);
    EXPECT_FALSE(predictor->get_use_previous_token());
    EXPECT_TRUE(props->empty());
}
