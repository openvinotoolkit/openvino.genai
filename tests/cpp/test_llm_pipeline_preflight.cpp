// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/variable.hpp"
#include "utils.hpp"

namespace {

std::shared_ptr<ov::Model> create_valid_minimal_llm(const std::string& input_name = "input_ids",
                                                    const std::string& output_name = "logits") {
    auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 16});
    input_param->set_friendly_name(input_name);

    auto bias = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 16}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(input_param, bias);
    add->set_friendly_name("add_logits");

    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name(output_name);
    result->get_output_tensor(0).set_names({output_name});

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_param}, "test_valid_llm");
}

std::shared_ptr<ov::Model> create_stateful_llm_with_sinks() {
    auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 16});
    input_param->set_friendly_name("input_ids");

    auto bias = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 16}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(input_param, bias);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("logits");

    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::Shape{1, 8, 16, 64}, ov::element::f32, "kv_cache_variable"});
    auto init_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 8, 16, 64}, {0.0f});
    auto read_val = std::make_shared<ov::op::v6::ReadValue>(init_const, variable);
    auto state_add = std::make_shared<ov::op::v1::Add>(read_val, init_const);
    auto assign_sink = std::make_shared<ov::op::v6::Assign>(state_add, variable);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::SinkVector{assign_sink},
                                             ov::ParameterVector{input_param}, "test_stateful_model");
    return model;
}

}  // anonymous namespace

TEST(LLMPipelinePreflight, ValidMinimalLLM_PassesPreFlight) {
    auto model = create_valid_minimal_llm();
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM));
}

TEST(LLMPipelinePreflight, ValidModel_MultipleOutputs_PassesPreFlight) {
    auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 16});
    input_param->set_friendly_name("input_ids");

    auto bias = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 16}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(input_param, bias);

    auto res_logits = std::make_shared<ov::op::v0::Result>(add);
    res_logits->set_friendly_name("logits");

    auto res_hidden = std::make_shared<ov::op::v0::Result>(add);
    res_hidden->set_friendly_name("hidden_states");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{res_logits, res_hidden},
                                             ov::ParameterVector{input_param}, "multi_out");
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM));
}

TEST(LLMPipelinePreflight, ValidModelWithOptionalInputs_PassesPreFlight) {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 16});
    input_ids->set_friendly_name("input_ids");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 16});
    attention_mask->set_friendly_name("attention_mask");

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 16});
    position_ids->set_friendly_name("position_ids");

    auto sum1 = std::make_shared<ov::op::v1::Add>(input_ids, attention_mask);
    auto sum2 = std::make_shared<ov::op::v1::Add>(sum1, position_ids);
    auto res = std::make_shared<ov::op::v0::Result>(sum2);
    res->set_friendly_name("logits");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{input_ids, attention_mask, position_ids},
                                             "optional_inputs");
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM));
}

TEST(LLMPipelinePreflight, ValidStatefulAssignSinkNodes_DualRootPasses) {
    auto model = create_stateful_llm_with_sinks();
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM));
}

TEST(LLMPipelinePreflight, ValidDraftModelWithHiddenStates_PassesPreFlight) {
    auto model = create_valid_minimal_llm("hidden_states", "logits");
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::DRAFT_MODEL));
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::EAGLE3_DRAFT));
}

TEST(LLMPipelinePreflight, ValidEagle3DraftModel_InputsEmbeds_PassesPreFlight) {
    auto model = create_valid_minimal_llm("inputs_embeds", "logits");
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::EAGLE3_DRAFT));
}

TEST(LLMPipelinePreflight, ValidLogitsNameVariants_LmLogits_Passes) {
    auto model = create_valid_minimal_llm("input_ids", "lm_logits");
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM));
}

TEST(LLMPipelinePreflight, ValidSingleOutput3DRank_InfersLogits_Passes) {
    auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 32000});
    input_param->set_friendly_name("input_ids");

    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 16, 32000}, {0.0f});
    auto add = std::make_shared<ov::op::v1::Add>(input_param, bias);

    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("output_0");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_param}, "rank3");
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM));
}

TEST(LLMPipelinePreflight, MissingInputIds_ThrowsDescriptiveDiagnostic) {
    auto model = create_valid_minimal_llm("wrong_token_name", "logits");
    try {
        ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM);
        FAIL() << "Expected ov::Exception for missing input_ids";
    } catch (const ov::Exception& ex) {
        std::string err = ex.what();
        EXPECT_TRUE(err.find("GAP-04") != std::string::npos);
        EXPECT_TRUE(err.find("input_ids") != std::string::npos);
    }
}

TEST(LLMPipelinePreflight, MissingLogits_ThrowsDescriptiveDiagnostic) {
    auto input_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 16});
    input_param->set_friendly_name("input_ids");

    auto bias = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 16}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(input_param, bias);

    auto res1 = std::make_shared<ov::op::v0::Result>(add);
    res1->set_friendly_name("intermediate_state");
    auto res2 = std::make_shared<ov::op::v0::Result>(add);
    res2->set_friendly_name("embeddings");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{res1, res2},
                                             ov::ParameterVector{input_param}, "no_logits");
    try {
        ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM);
        FAIL() << "Expected ov::Exception for missing logits";
    } catch (const ov::Exception& ex) {
        std::string err = ex.what();
        EXPECT_TRUE(err.find("GAP-04") != std::string::npos);
        EXPECT_TRUE(err.find("logits") != std::string::npos);
    }
}

TEST(LLMPipelinePreflight, DisconnectedParam_ThrowsDescriptiveDiagnostic) {
    auto connected_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 16});
    connected_param->set_friendly_name("input_ids");

    auto disconnected_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 16});
    disconnected_param->set_friendly_name("input_ids_extra");

    auto bias = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 16}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(connected_param, bias);
    auto res = std::make_shared<ov::op::v0::Result>(add);
    res->set_friendly_name("logits");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                             ov::ParameterVector{connected_param, disconnected_param},
                                             "disconnected");
    try {
        ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM);
        FAIL() << "Expected ov::Exception for disconnected parameter";
    } catch (const ov::Exception& ex) {
        std::string err = ex.what();
        EXPECT_TRUE(err.find("GAP-04") != std::string::npos);
        EXPECT_TRUE(err.find("disconnected") != std::string::npos || err.find("input_ids_extra") != std::string::npos);
    }
}

TEST(LLMPipelinePreflight, EmptyModel_ThrowsGracefully) {
    auto model = std::make_shared<ov::Model>(ov::ResultVector{}, ov::ParameterVector{}, "empty_model");
    try {
        ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM);
        FAIL() << "Expected ov::Exception for empty model";
    } catch (const ov::Exception& ex) {
        std::string err = ex.what();
        EXPECT_TRUE(err.find("GAP-04") != std::string::npos);
    }
}

TEST(LLMPipelinePreflight, NullModel_ThrowsImmediately) {
    std::shared_ptr<ov::Model> null_model = nullptr;
    try {
        ov::genai::utils::validate_llm_model_dag(null_model, ov::genai::utils::ModelRole::PRIMARY_LLM);
        FAIL() << "Expected ov::Exception for null model";
    } catch (const ov::Exception& ex) {
        std::string err = ex.what();
        EXPECT_TRUE(err.find("Model must not be null") != std::string::npos);
    }
}

TEST(LLMPipelinePreflight, DraftModelMissingAllAllowedInputs_Throws) {
    auto model = create_valid_minimal_llm("unsupported_input_name", "logits");
    EXPECT_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::DRAFT_MODEL),
                 ov::Exception);
}

TEST(LLMPipelinePreflight, ConcurrentValidation_ThreadSafe) {
    auto model = create_valid_minimal_llm();
    std::vector<std::thread> workers;
    workers.reserve(8);
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&model]() {
            for (int iter = 0; iter < 100; ++iter) {
                EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM));
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

TEST(LLMPipelinePreflight, LargeGraphTopology_PreFlightLatencyBudget) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 64});
    input->set_friendly_name("input_ids");

    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 64}, {0.1f});
    std::shared_ptr<ov::Node> curr = input;
    for (size_t i = 0; i < 1000; ++i) {
        curr = std::make_shared<ov::op::v1::Add>(curr, bias);
    }

    auto result = std::make_shared<ov::op::v0::Result>(curr);
    result->set_friendly_name("logits");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "large_model");

    auto start = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(ov::genai::utils::validate_llm_model_dag(model, ov::genai::utils::ModelRole::PRIMARY_LLM));
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);

    EXPECT_LT(duration.count(), 5000);  // Pre-flight latency under 5ms
}
