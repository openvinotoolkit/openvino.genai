// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/runtime/core.hpp"
#include "utils.hpp"

namespace {

constexpr size_t BATCH = 1;
constexpr size_t HEADS = 4;
constexpr size_t HEAD_DIM = 128;
constexpr size_t SEQ_AXIS = 2;

struct TrimKvCacheTestParam {
    std::string device;
    bool use_attention;
    bool dynamic_initializer;
    ov::element::Type precision;
};

std::string test_name(const testing::TestParamInfo<TrimKvCacheTestParam>& info) {
    std::ostringstream name;
    name << info.param.device
         << "_" << (info.param.precision == ov::element::f16 ? "F16" : "F32")
         << (info.param.dynamic_initializer ? "_DynamicInit" : "_StaticInit")
         << (info.param.use_attention ? "_SDPAState" : "_PlainState");
    return name.str();
}

bool device_available(ov::Core& core, const std::string& device) {
    auto devices = core.get_available_devices();
    return std::find(devices.begin(), devices.end(), device) != devices.end();
}

ov::Shape kv_shape(size_t seq_len) {
    return ov::Shape{BATCH, HEADS, seq_len, HEAD_DIM};
}

std::vector<float> make_kv_data(size_t token_start, size_t seq_len, float offset) {
    std::vector<float> data(BATCH * HEADS * seq_len * HEAD_DIM);
    size_t idx = 0;
    for (size_t batch = 0; batch < BATCH; ++batch) {
        for (size_t head = 0; head < HEADS; ++head) {
            for (size_t token = 0; token < seq_len; ++token) {
                for (size_t dim = 0; dim < HEAD_DIM; ++dim) {
                    data[idx++] = offset + static_cast<float>((token_start + token) * 100 + head * 10 + dim);
                }
            }
        }
    }
    return data;
}

ov::Tensor make_kv_tensor(size_t token_start, size_t seq_len, float offset, ov::element::Type precision) {
    auto data = make_kv_data(token_start, seq_len, offset);
    ov::Tensor tensor(precision, kv_shape(seq_len));
    if (precision == ov::element::f32) {
        std::copy(data.begin(), data.end(), tensor.data<float>());
    } else {
        auto* tensor_data = tensor.data<ov::float16>();
        std::transform(data.begin(), data.end(), tensor_data, [](float value) {
            return ov::float16(value);
        });
    }
    return tensor;
}

ov::Tensor make_integral_tensor(ov::element::Type precision, const ov::Shape& shape, const std::vector<int64_t>& values) {
    ov::Tensor tensor(precision, shape);
    if (precision == ov::element::i32) {
        auto* data = tensor.data<int32_t>();
        std::transform(values.begin(), values.end(), data, [](int64_t value) {
            return static_cast<int32_t>(value);
        });
    } else {
        auto* data = tensor.data<int64_t>();
        std::copy(values.begin(), values.end(), data);
    }
    return tensor;
}

std::shared_ptr<ov::op::v0::Parameter> make_kv_parameter(const std::string& name, ov::element::Type precision) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        precision,
        ov::PartialShape{-1, static_cast<int64_t>(HEADS), -1, static_cast<int64_t>(HEAD_DIM)});
    parameter->set_friendly_name(name);
    parameter->output(0).get_tensor().set_names({name});
    return parameter;
}

std::shared_ptr<ov::op::v0::Parameter> make_beam_idx_parameter() {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    parameter->set_friendly_name("beam_idx");
    parameter->output(0).get_tensor().set_names({"beam_idx"});
    return parameter;
}

std::shared_ptr<ov::op::util::Variable> make_kv_variable(const std::string& name, ov::element::Type precision) {
    return std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
        ov::PartialShape{-1, static_cast<int64_t>(HEADS), -1, static_cast<int64_t>(HEAD_DIM)},
        precision,
        name});
}

ov::Output<ov::Node> make_dynamic_empty_kv_initializer(const ov::Output<ov::Node>& new_value, ov::element::Type precision) {
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(new_value, ov::element::i64);
    auto batch_index = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto batch = std::make_shared<ov::op::v8::Gather>(input_shape, batch_index, gather_axis);
    auto static_tail = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{3},
        std::vector<int64_t>{static_cast<int64_t>(HEADS), 0, static_cast<int64_t>(HEAD_DIM)});
    auto target_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch, static_tail}, 0);
    auto zero = ov::op::v0::Constant::create(precision, ov::Shape{}, std::vector<float>{0.0f});
    return std::make_shared<ov::op::v3::Broadcast>(zero, target_shape, ov::op::BroadcastType::NUMPY);
}

ov::Output<ov::Node> make_static_empty_kv_initializer(ov::element::Type precision) {
    return ov::op::v0::Constant::create(precision, kv_shape(0), std::vector<float>{});
}

struct StatefulKv {
    ov::Output<ov::Node> present;
    std::shared_ptr<ov::op::v6::Assign> assign;
};

StatefulKv make_stateful_kv(const std::string& variable_name,
                            const ov::Output<ov::Node>& new_value,
                            const ov::Output<ov::Node>& beam_idx,
                            ov::element::Type precision,
                            bool dynamic_initializer) {
    auto variable = make_kv_variable(variable_name, precision);
    auto initial_state = dynamic_initializer ? make_dynamic_empty_kv_initializer(new_value, precision)
                                             : make_static_empty_kv_initializer(precision);
    auto read = std::make_shared<ov::op::v6::ReadValue>(initial_state, variable);
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto reordered_past = std::make_shared<ov::op::v8::Gather>(read, beam_idx, gather_axis);
    auto present = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reordered_past, new_value}, SEQ_AXIS);
    auto assign = std::make_shared<ov::op::v6::Assign>(present, variable);
    return {present, assign};
}

std::shared_ptr<ov::Model> make_plain_state_model(ov::element::Type precision, bool dynamic_initializer) {
    auto key = make_kv_parameter("key", precision);
    auto value = make_kv_parameter("value", precision);
    auto beam_idx = make_beam_idx_parameter();
    auto key_state = make_stateful_kv("key_cache.0", key, beam_idx, precision, dynamic_initializer);
    auto value_state = make_stateful_kv("value_cache.0", value, beam_idx, precision, dynamic_initializer);

    auto key_result = std::make_shared<ov::op::v0::Result>(key_state.present);
    auto value_result = std::make_shared<ov::op::v0::Result>(value_state.present);
    return std::make_shared<ov::Model>(
        ov::ResultVector{key_result, value_result},
        ov::SinkVector{key_state.assign, value_state.assign},
        ov::ParameterVector{key, value, beam_idx},
        "trim_kv_cache_plain_state");
}

std::shared_ptr<ov::Model> make_sdpa_state_model(ov::element::Type precision, bool dynamic_initializer) {
    auto query = make_kv_parameter("query", precision);
    auto key = make_kv_parameter("key", precision);
    auto value = make_kv_parameter("value", precision);
    auto beam_idx = make_beam_idx_parameter();
    auto key_state = make_stateful_kv("key_cache.0", key, beam_idx, precision, dynamic_initializer);
    auto value_state = make_stateful_kv("value_cache.0", value, beam_idx, precision, dynamic_initializer);

    auto attention = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
        query,
        key_state.present,
        value_state.present,
        true);
    auto attention_result = std::make_shared<ov::op::v0::Result>(attention);
    return std::make_shared<ov::Model>(
        ov::ResultVector{attention_result},
        ov::SinkVector{key_state.assign, value_state.assign},
        ov::ParameterVector{query, key, value, beam_idx},
        "trim_kv_cache_sdpa_state");
}

std::shared_ptr<ov::Model> make_model(bool use_attention, ov::element::Type precision, bool dynamic_initializer) {
    return use_attention ? make_sdpa_state_model(precision, dynamic_initializer)
                         : make_plain_state_model(precision, dynamic_initializer);
}

void append_tokens(ov::InferRequest& request,
                   bool use_attention,
                   ov::element::Type precision,
                   size_t token_start,
                   size_t seq_len) {
    if (use_attention) {
        request.set_tensor("query", make_kv_tensor(token_start, seq_len, 0.0f, precision));
    }
    request.set_tensor("key", make_kv_tensor(token_start, seq_len, 1000.0f, precision));
    request.set_tensor("value", make_kv_tensor(token_start, seq_len, 2000.0f, precision));
    ov::Tensor beam_idx(ov::element::i32, ov::Shape{BATCH});
    std::fill_n(beam_idx.data<int32_t>(), BATCH, 0);
    request.set_tensor("beam_idx", beam_idx);
    request.infer();
}

void append_lm_tokens(ov::InferRequest& request, size_t token_start, size_t seq_len, size_t total_seq_len) {
    std::vector<int64_t> input_ids(seq_len);
    std::vector<int64_t> attention_mask(total_seq_len, 1);
    std::vector<int64_t> position_ids(seq_len);
    for (size_t idx = 0; idx < seq_len; ++idx) {
        input_ids[idx] = static_cast<int64_t>(100 + token_start + idx);
        position_ids[idx] = static_cast<int64_t>(token_start + idx);
    }

    const auto input_ids_type = request.get_compiled_model().input("input_ids").get_element_type();
    const auto attention_mask_type = request.get_compiled_model().input("attention_mask").get_element_type();
    const auto position_ids_type = request.get_compiled_model().input("position_ids").get_element_type();
    const auto beam_idx_type = request.get_compiled_model().input("beam_idx").get_element_type();

    request.set_tensor("input_ids", make_integral_tensor(input_ids_type, ov::Shape{BATCH, seq_len}, input_ids));
    request.set_tensor("attention_mask",
                       make_integral_tensor(attention_mask_type, ov::Shape{BATCH, total_seq_len}, attention_mask));
    request.set_tensor("position_ids", make_integral_tensor(position_ids_type, ov::Shape{BATCH, seq_len}, position_ids));
    request.set_tensor("beam_idx", make_integral_tensor(beam_idx_type, ov::Shape{BATCH}, std::vector<int64_t>{0}));
    request.infer();
}

std::unordered_map<std::string, ov::Tensor> snapshot_state(ov::InferRequest& request) {
    std::unordered_map<std::string, ov::Tensor> snapshot;
    for (auto& state : request.query_state()) {
        auto state_tensor = state.get_state();
        ov::Tensor host_tensor(state_tensor.get_element_type(), state_tensor.get_shape());
        state_tensor.copy_to(host_tensor);
        snapshot.emplace(state.get_name(), host_tensor);
    }
    return snapshot;
}

std::unordered_map<std::string, ov::Tensor> trim_state_snapshot(
    const std::unordered_map<std::string, ov::Tensor>& snapshot,
    size_t tokens_to_trim,
    size_t seq_length_axis) {
    std::unordered_map<std::string, ov::Tensor> trimmed_snapshot;
    for (const auto& [name, tensor] : snapshot) {
        auto shape = tensor.get_shape();
        OPENVINO_ASSERT(seq_length_axis < shape.size(), "Invalid KV cache sequence axis for state: ", name);
        OPENVINO_ASSERT(shape[seq_length_axis] >= tokens_to_trim,
                        "Cannot trim ",
                        tokens_to_trim,
                        " tokens from state '",
                        name,
                        "' with sequence length ",
                        shape[seq_length_axis]);
        shape[seq_length_axis] -= tokens_to_trim;

        ov::Coordinate begin(tensor.get_shape().size(), 0);
        ov::Coordinate end(shape.begin(), shape.end());
        ov::Tensor trimmed_view(tensor, begin, end);
        ov::Tensor trimmed_tensor(tensor.get_element_type(), shape);
        trimmed_view.copy_to(trimmed_tensor);
        trimmed_snapshot.emplace(name, trimmed_tensor);
    }
    return trimmed_snapshot;
}

float tensor_value_at(const ov::Tensor& tensor, size_t idx) {
    return tensor.get_element_type() == ov::element::f32 ? tensor.data<const float>()[idx]
                                                         : static_cast<float>(tensor.data<const ov::float16>()[idx]);
}

::testing::AssertionResult states_are_same(const std::unordered_map<std::string, ov::Tensor>& actual,
                                           const std::unordered_map<std::string, ov::Tensor>& expected) {
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure() << "State count mismatch: actual=" << actual.size()
                                             << ", expected=" << expected.size();
    }

    for (const auto& [name, expected_tensor] : expected) {
        auto actual_it = actual.find(name);
        if (actual_it == actual.end()) {
            return ::testing::AssertionFailure() << "Missing state: " << name;
        }

        const auto& actual_tensor = actual_it->second;
        if (actual_tensor.get_element_type() != expected_tensor.get_element_type()) {
            return ::testing::AssertionFailure() << "Element type mismatch for state " << name
                                                 << ": actual=" << actual_tensor.get_element_type()
                                                 << ", expected=" << expected_tensor.get_element_type();
        }
        if (actual_tensor.get_shape() != expected_tensor.get_shape()) {
            return ::testing::AssertionFailure() << "Shape mismatch for state " << name
                                                 << ": actual=" << actual_tensor.get_shape()
                                                 << ", expected=" << expected_tensor.get_shape();
        }

        for (size_t idx = 0; idx < expected_tensor.get_size(); ++idx) {
            const float actual_value = tensor_value_at(actual_tensor, idx);
            const float expected_value = tensor_value_at(expected_tensor, idx);
            if (std::abs(actual_value - expected_value) > 1e-3f) {
                return ::testing::AssertionFailure()
                       << "State mismatch: " << name
                       << ", element: " << idx
                       << ", actual=" << actual_value
                       << ", expected=" << expected_value
                       << ", diff=" << std::abs(actual_value - expected_value);
            }
        }
    }

    return ::testing::AssertionSuccess();
}

void expect_same_state(const std::unordered_map<std::string, ov::Tensor>& actual,
                       const std::unordered_map<std::string, ov::Tensor>& expected) {
    ASSERT_TRUE(states_are_same(actual, expected));
}

ov::genai::utils::CacheState make_kv_cache_state(size_t tokens_to_trim) {
    ov::genai::utils::CacheTypes cache_types;
    cache_types.add_kvcache();
    ov::genai::utils::CacheState cache_state(cache_types);
    cache_state.num_tokens_to_trim = tokens_to_trim;
    cache_state.seq_length_axis = SEQ_AXIS;
    return cache_state;
}

void expose_annotated_hidden_state_outputs(const std::shared_ptr<ov::Model>& model) {
    ov::ResultVector hidden_results;
    for (const auto& op : model->get_ordered_ops()) {
        for (const auto& output : op->outputs()) {
            for (const auto& name : output.get_names()) {
                if (name.rfind("ov.hidden_states.decoder_layer_", 0) == 0) {
                    hidden_results.push_back(std::make_shared<ov::op::v0::Result>(output));
                    break;
                }
            }
        }
    }
    if (!hidden_results.empty()) {
        model->add_results(hidden_results);
        model->validate_nodes_and_infer_types();
    }
}

class TrimKvCacheSyntheticStateTest : public ::testing::TestWithParam<TrimKvCacheTestParam> {};

TEST_P(TrimKvCacheSyntheticStateTest, TrimmedStateMatchesFreshReplayPrefix) {
    ov::Core core;
    const auto param = GetParam();
    if (!device_available(core, param.device)) {
        GTEST_SKIP() << param.device << " device is not available";
    }

    auto compiled_model = core.compile_model(make_model(param.use_attention, param.precision, param.dynamic_initializer),
                                             param.device);
    auto live_request = compiled_model.create_infer_request();
    auto replay_request = compiled_model.create_infer_request();

    append_tokens(live_request, param.use_attention, param.precision, 0, 4);
    append_tokens(live_request, param.use_attention, param.precision, 4, 3);

    auto cache_state = make_kv_cache_state(2);
    ov::genai::utils::trim_kv_cache(live_request, cache_state, {});

    append_tokens(replay_request, param.use_attention, param.precision, 0, 4);
    append_tokens(replay_request, param.use_attention, param.precision, 4, 1);

    expect_same_state(snapshot_state(live_request), snapshot_state(replay_request));
}

class TrimKvCacheRealModelTest : public ::testing::TestWithParam<std::string> {};

TEST_P(TrimKvCacheRealModelTest, TrimmedStateMatchesPreTrimPrefix) {
    const char* model_env = std::getenv("TRIM_KV_CACHE_TEST_MODEL");
    if (model_env == nullptr) {
        GTEST_SKIP() << "TRIM_KV_CACHE_TEST_MODEL is not set";
    }

    ov::Core core;
    const std::string device = GetParam();
    if (!device_available(core, device)) {
        GTEST_SKIP() << device << " device is not available";
    }

    std::filesystem::path model_path(model_env);
    if (std::filesystem::is_directory(model_path)) {
        model_path /= "openvino_model.xml";
    }
    ASSERT_TRUE(std::filesystem::exists(model_path)) << model_path;

    auto model = core.read_model(model_path);
    if (std::getenv("TRIM_KV_CACHE_EXPOSE_HIDDEN_STATES") != nullptr) {
        expose_annotated_hidden_state_outputs(model);
    }
    auto compiled_model = core.compile_model(model, device);
    auto live_request = compiled_model.create_infer_request();

    append_lm_tokens(live_request, 0, 4, 4);
    append_lm_tokens(live_request, 4, 3, 7);

    const size_t tokens_to_trim = 2;
    const auto pre_trim_state = snapshot_state(live_request);
    const auto expected_state = trim_state_snapshot(pre_trim_state, tokens_to_trim, SEQ_AXIS);

    auto cache_state = make_kv_cache_state(2);
    ov::genai::utils::trim_kv_cache(live_request, cache_state, {});

    expect_same_state(snapshot_state(live_request), expected_state);
}

TEST_P(TrimKvCacheRealModelTest, BlockValidationStateMatchesAutoregressiveReplayAcrossAcceptedPrefixes) {
    const char* model_env = std::getenv("TRIM_KV_CACHE_TEST_MODEL");
    if (model_env == nullptr) {
        GTEST_SKIP() << "TRIM_KV_CACHE_TEST_MODEL is not set";
    }

    ov::Core core;
    const std::string device = GetParam();
    if (!device_available(core, device)) {
        GTEST_SKIP() << device << " device is not available";
    }

    std::filesystem::path model_path(model_env);
    if (std::filesystem::is_directory(model_path)) {
        model_path /= "openvino_model.xml";
    }
    ASSERT_TRUE(std::filesystem::exists(model_path)) << model_path;

    auto model = core.read_model(model_path);
    if (std::getenv("TRIM_KV_CACHE_EXPOSE_HIDDEN_STATES") != nullptr) {
        expose_annotated_hidden_state_outputs(model);
    }
    auto compiled_model = core.compile_model(model, device);

    constexpr size_t prompt_len = 4;
    constexpr size_t validation_block_len = 5;
    const std::vector<size_t> accepted_prefix_lengths = {1, 2, 3, validation_block_len};

    for (const auto accepted_prefix_len : accepted_prefix_lengths) {
        SCOPED_TRACE("accepted_prefix_len=" + std::to_string(accepted_prefix_len));

        auto block_request = compiled_model.create_infer_request();
        append_lm_tokens(block_request, 0, prompt_len, prompt_len);
        append_lm_tokens(block_request, prompt_len, validation_block_len, prompt_len + validation_block_len);

        const size_t tokens_to_trim = validation_block_len - accepted_prefix_len;
        if (tokens_to_trim > 0) {
            auto cache_state = make_kv_cache_state(tokens_to_trim);
            ov::genai::utils::trim_kv_cache(block_request, cache_state, {});
        }

        auto autoregressive_request = compiled_model.create_infer_request();
        append_lm_tokens(autoregressive_request, 0, prompt_len, prompt_len);
        for (size_t idx = 0; idx < accepted_prefix_len; ++idx) {
            append_lm_tokens(autoregressive_request, prompt_len + idx, 1, prompt_len + idx + 1);
        }

        EXPECT_TRUE(states_are_same(snapshot_state(block_request), snapshot_state(autoregressive_request)));
    }
}

INSTANTIATE_TEST_SUITE_P(
    CPUAndGPU,
    TrimKvCacheSyntheticStateTest,
    ::testing::Values(TrimKvCacheTestParam{"CPU", false, false, ov::element::f32},
                      TrimKvCacheTestParam{"CPU", true, false, ov::element::f32},
                      TrimKvCacheTestParam{"GPU", false, false, ov::element::f32},
                      TrimKvCacheTestParam{"GPU", true, false, ov::element::f32},
                      TrimKvCacheTestParam{"GPU", false, true, ov::element::f16},
                      TrimKvCacheTestParam{"GPU", true, true, ov::element::f16}),
    test_name);

INSTANTIATE_TEST_SUITE_P(CPUAndGPU,
                         TrimKvCacheRealModelTest,
                         ::testing::Values(std::string{"CPU"}, std::string{"GPU"}));

}  // namespace
