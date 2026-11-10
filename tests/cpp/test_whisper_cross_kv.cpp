// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/core.hpp"
#include "whisper/decoder_round_guard.hpp"
#include "whisper/decoder_rows.hpp"
#include "whisper/models/cross_kv_decoder.hpp"
#include "whisper/transformations/cross_kv_inputs.hpp"

namespace {

constexpr size_t layers = 2;
constexpr int64_t hidden = 4;
constexpr int64_t frames = 5;

using Rows = std::vector<std::vector<float>>;

struct DecoderOptions {
    bool extra_encoder_consumer = false;
    bool duplicate_layer_name = false;
    bool duplicate_read_value_name = false;
    bool modified_cross_assign = false;
    bool convert_before_sdpa = false;
    bool initializer_reads_other_input = false;
    bool f16_second_layer_cross_states = false;
    bool wider_second_layer_cross_value = false;
};

std::shared_ptr<ov::op::v0::Parameter> make_input(const std::string& name,
                                                  const ov::element::Type& type,
                                                  const ov::PartialShape& shape) {
    auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);
    input->set_friendly_name(name);
    input->output(0).get_tensor().set_names({name});
    return input;
}

std::shared_ptr<ov::Model> make_whisper_decoder(const DecoderOptions& options = {}) {
    auto input_ids = make_input("input_ids", ov::element::i64, {-1, -1});
    auto encoder = make_input("encoder_hidden_states", ov::element::f32, {-1, -1, hidden});
    auto beam_idx = make_input("beam_idx", ov::element::i32, {-1});
    auto extra = make_input("extra", ov::element::f32, {-1, 1, hidden});

    auto ids = std::make_shared<ov::op::v0::Convert>(input_ids, ov::element::f32);
    auto token_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
    auto embedding = ov::op::v0::Constant::create(ov::element::f32, {1, 1, hidden}, {0.1f, 0.2f, 0.3f, 0.4f});
    auto query =
        std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v0::Unsqueeze>(ids, token_axis), embedding);

    const ov::op::util::VariableInfo state_info{ov::PartialShape{-1, -1, hidden}, ov::element::f32, ""};
    auto batch_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    ov::OutputVector layer_outputs;
    ov::SinkVector sinks;
    for (size_t layer = 0; layer < layers; ++layer) {
        const bool f16_cross = options.f16_second_layer_cross_states && layer == 1;
        const bool wider_value = options.wider_second_layer_cross_value && layer == 1;
        const int64_t value_width = hidden + 2;
        ov::OutputVector cross_kv;
        for (const std::string kind : {"key", "value"}) {
            const bool widened = wider_value && kind == "value";
            auto info = state_info;
            info.variable_id = "cross." + std::to_string(layer) + "." + kind;
            if (f16_cross) {
                info.data_type = ov::element::f16;
            }
            if (widened) {
                info.data_shape = ov::PartialShape{-1, -1, value_width};
            }
            auto variable = std::make_shared<ov::op::util::Variable>(info);
            auto scale = ov::op::v0::Constant::create(ov::element::f32,
                                                      {},
                                                      {static_cast<float>(layer) + (kind == "key" ? 1.5f : 2.5f)});
            ov::Output<ov::Node> initializer = std::make_shared<ov::op::v1::Multiply>(encoder, scale);
            const bool first_state = layer == 0 && kind == "key";
            if (options.initializer_reads_other_input && first_state) {
                initializer = std::make_shared<ov::op::v1::Add>(initializer, extra);
            }
            if (widened) {
                auto widen = ov::op::v0::Constant::create(ov::element::f32,
                                                          {static_cast<size_t>(hidden), static_cast<size_t>(value_width)},
                                                          std::vector<float>(hidden * value_width, 0.1f));
                initializer = std::make_shared<ov::op::v0::MatMul>(initializer, widen);
            }
            if (f16_cross) {
                initializer = std::make_shared<ov::op::v0::Convert>(initializer, ov::element::f16);
            }
            auto read_value = std::make_shared<ov::op::v6::ReadValue>(initializer, variable);
            read_value->set_friendly_name(options.duplicate_read_value_name ? "cross_state" : info.variable_id);
            std::shared_ptr<ov::Node> stored = read_value;
            if (options.modified_cross_assign && first_state) {
                stored = std::make_shared<ov::op::v0::Relu>(read_value);
            }
            sinks.push_back(std::make_shared<ov::op::v6::Assign>(stored, variable));
            ov::Output<ov::Node> consumed = read_value;
            if (options.convert_before_sdpa) {
                consumed = std::make_shared<ov::op::v0::Convert>(
                    std::make_shared<ov::op::v0::Convert>(read_value, ov::element::f16),
                    ov::element::f32);
            }
            cross_kv.push_back(consumed);
        }
        ov::Output<ov::Node> cross_query = query;
        if (f16_cross) {
            cross_query = std::make_shared<ov::op::v0::Convert>(query, ov::element::f16);
        }
        auto cross_sdpa =
            std::make_shared<ov::op::v13::ScaledDotProductAttention>(cross_query, cross_kv[0], cross_kv[1], false);
        const size_t named_layer = options.duplicate_layer_name ? 0 : layer;
        cross_sdpa->set_friendly_name("model.decoder.layers." + std::to_string(named_layer) + ".encoder_attn/sdpa");
        ov::Output<ov::Node> cross = cross_sdpa;
        if (f16_cross) {
            cross = std::make_shared<ov::op::v0::Convert>(cross, ov::element::f32);
        }
        if (wider_value) {
            auto narrow = ov::op::v0::Constant::create(ov::element::f32,
                                                       {static_cast<size_t>(value_width), static_cast<size_t>(hidden)},
                                                       std::vector<float>(value_width * hidden, 0.1f));
            cross = std::make_shared<ov::op::v0::MatMul>(cross, narrow);
        }

        auto info = state_info;
        info.variable_id = "self." + std::to_string(layer);
        auto variable = std::make_shared<ov::op::util::Variable>(info);
        auto zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
        auto past =
            std::make_shared<ov::op::v6::ReadValue>(std::make_shared<ov::op::v1::Multiply>(query, zero), variable);
        auto reordered_past = std::make_shared<ov::op::v8::Gather>(past, beam_idx, batch_axis);
        auto present = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reordered_past, query}, 1);
        sinks.push_back(std::make_shared<ov::op::v6::Assign>(present, variable));
        auto self = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, present, present, false);
        self->set_friendly_name("model.decoder.layers." + std::to_string(layer) + ".self_attn/sdpa");
        layer_outputs.push_back(std::make_shared<ov::op::v1::Add>(cross, self));
    }
    auto logits = std::make_shared<ov::op::v1::Add>(layer_outputs[0], layer_outputs[1]);
    logits->output(0).get_tensor().set_names({"logits"});
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(logits)};
    if (options.extra_encoder_consumer) {
        results.push_back(std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v3::ShapeOf>(encoder)));
    }
    ov::ParameterVector parameters{input_ids, encoder, beam_idx};
    if (options.initializer_reads_other_input) {
        parameters.push_back(extra);
    }
    return std::make_shared<ov::Model>(results, sinks, parameters);
}

ov::Tensor make_encoder(const std::vector<size_t>& audios) {
    ov::Tensor encoder(ov::element::f32, {audios.size(), frames, hidden});
    const size_t row_size = frames * hidden;
    for (size_t row = 0; row < audios.size(); ++row) {
        for (size_t i = 0; i < row_size; ++i) {
            encoder.data<float>()[row * row_size + i] =
                0.3f * static_cast<float>(audios[row]) - 0.2f + 0.01f * static_cast<float>(i % 13);
        }
    }
    return encoder;
}

ov::Tensor make_input_ids(const std::vector<int64_t>& tokens) {
    ov::Tensor input_ids(ov::element::i64, {tokens.size(), 1});
    std::copy(tokens.begin(), tokens.end(), input_ids.data<int64_t>());
    return input_ids;
}

ov::Tensor make_beam_idx(const std::vector<int32_t>& beams) {
    ov::Tensor beam_idx(ov::element::i32, {beams.size()});
    std::copy(beams.begin(), beams.end(), beam_idx.data<int32_t>());
    return beam_idx;
}

std::vector<int64_t> tokens_for(const std::vector<size_t>& audios, size_t step_index) {
    std::vector<int64_t> tokens;
    for (const size_t audio : audios) {
        tokens.push_back(static_cast<int64_t>(1 + (audio * 3 + step_index) % 7));
    }
    return tokens;
}

Rows step(ov::genai::WhisperDecoder& decoder,
          const ov::Tensor& encoder,
          const std::vector<int64_t>& tokens,
          const std::vector<int32_t>& beams) {
    decoder.start_async(encoder, make_input_ids(tokens), make_beam_idx(beams));
    const ov::Tensor logits = decoder.wait();
    const ov::Shape shape = logits.get_shape();
    Rows rows(shape[0]);
    for (size_t row = 0; row < shape[0]; ++row) {
        const float* data = logits.data<const float>() + (row * shape[1] + shape[1] - 1) * shape[2];
        rows[row].assign(data, data + shape[2]);
    }
    return rows;
}

std::vector<float> logits_of(ov::InferRequest& request) {
    const ov::Tensor logits = request.get_tensor("logits");
    return {logits.data<const float>(), logits.data<const float>() + logits.get_size()};
}

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "element " << i;
    }
}

void expect_error(const std::function<void()>& action, const std::string& message) {
    try {
        action();
        ADD_FAILURE() << "Expected an exception containing: " << message;
    } catch (const ov::Exception& error) {
        EXPECT_THAT(error.what(), ::testing::HasSubstr(message));
    }
}

TEST(WhisperCrossKV, SplitDecoderMatchesOriginalDecoder) {
    DecoderOptions repeated_names;
    repeated_names.duplicate_read_value_name = true;
    for (const DecoderOptions& options : {DecoderOptions{}, repeated_names}) {
        SCOPED_TRACE(options.duplicate_read_value_name ? "repeated ReadValue names" : "default");
        const auto model = make_whisper_decoder(options);
        const auto split = ov::genai::split_whisper_cross_kv(model);
        ASSERT_TRUE(split.has_value());
        EXPECT_EQ(
            split->names,
            (std::vector<std::string>{"cross_kv.0.key", "cross_kv.0.value", "cross_kv.1.key", "cross_kv.1.value"}));
        std::vector<std::string> variables;
        for (const auto& variable : split->decoder->get_variables()) {
            variables.push_back(variable->get_info().variable_id);
        }
        std::sort(variables.begin(), variables.end());
        EXPECT_EQ(variables, (std::vector<std::string>{"self.0", "self.1"}));

        ov::Core core;
        const ov::Tensor encoder = make_encoder({1});
        const ov::Tensor input_ids = make_input_ids({3});
        const ov::Tensor beam_idx = make_beam_idx({0});

        auto original = core.compile_model(model, "CPU").create_infer_request();
        original.set_tensor("encoder_hidden_states", encoder);
        original.set_tensor("input_ids", input_ids);
        original.set_tensor("beam_idx", beam_idx);
        original.infer();

        auto projector = core.compile_model(split->projector, "CPU").create_infer_request();
        projector.set_tensor("encoder_hidden_states", encoder);
        projector.infer();
        auto decoder = core.compile_model(split->decoder, "CPU").create_infer_request();
        for (const auto& name : split->names) {
            decoder.set_tensor(name, projector.get_tensor(name));
        }
        decoder.set_tensor("input_ids", input_ids);
        decoder.set_tensor("beam_idx", beam_idx);
        decoder.infer();

        expect_near(logits_of(decoder), logits_of(original), 1e-5f);
    }
}

TEST(WhisperCrossKV, SplitFallsBackForUnsupportedDecoderStructures) {
    const std::vector<std::pair<std::string, bool DecoderOptions::*>> unsupported{
        {"encoder hidden states have another consumer", &DecoderOptions::extra_encoder_consumer},
        {"two states claim the same layer", &DecoderOptions::duplicate_layer_name},
        {"cross-attention state is not stored unchanged", &DecoderOptions::modified_cross_assign},
        {"state reaches attention through other ops", &DecoderOptions::convert_before_sdpa},
        {"state initializer reads another input", &DecoderOptions::initializer_reads_other_input},
    };
    for (const auto& [name, option] : unsupported) {
        SCOPED_TRACE(name);
        DecoderOptions options;
        options.*option = true;
        std::optional<ov::genai::WhisperCrossKVModels> split;
        ASSERT_NO_THROW(split = ov::genai::split_whisper_cross_kv(make_whisper_decoder(options)));
        EXPECT_FALSE(split.has_value());
    }
}

class SerializedDecoder {
public:
    explicit SerializedDecoder(const DecoderOptions& options)
        : m_path(std::filesystem::temp_directory_path() /
                 ("whisper_cross_kv_outputs_" +
                  std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        std::filesystem::create_directories(m_path);
        ov::serialize(make_whisper_decoder(options),
                      (m_path / "openvino_decoder_model.xml").string(),
                      (m_path / "openvino_decoder_model.bin").string());
    }

    ~SerializedDecoder() {
        std::filesystem::remove_all(m_path);
    }

    const std::filesystem::path& path() const {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

std::shared_ptr<ov::genai::WhisperDecoder> decoder_from_path(const std::filesystem::path& models_path) {
    return ov::genai::WhisperDecoder::from_path(models_path,
                                                "CPU",
                                                {},
                                                ov::PartialShape{-1, frames, hidden},
                                                false,
                                                ov::element::f32);
}

TEST(WhisperCrossKV, HeterogeneousCrossKVOutputsFallBackToStatefulDecoder) {
    const std::vector<std::pair<std::string, bool DecoderOptions::*>> heterogeneous{
        {"cross-attention states have different element types", &DecoderOptions::f16_second_layer_cross_states},
        {"cross-attention states have different shapes", &DecoderOptions::wider_second_layer_cross_value},
    };
    for (const auto& [name, option] : heterogeneous) {
        SCOPED_TRACE(name);
        DecoderOptions options;
        options.*option = true;
        ASSERT_TRUE(ov::genai::split_whisper_cross_kv(make_whisper_decoder(options)).has_value());

        const SerializedDecoder serialized{options};
        EXPECT_EQ(ov::genai::WhisperCrossKVDecoder::create(serialized.path(),
                                                           "CPU",
                                                           {},
                                                           ov::PartialShape{-1, frames, hidden},
                                                           false),
                  nullptr);

        auto decoder = decoder_from_path(serialized.path());
        ASSERT_NE(decoder, nullptr);
        EXPECT_EQ(std::dynamic_pointer_cast<ov::genai::WhisperCrossKVDecoder>(decoder), nullptr);
        const Rows rows = step(*decoder, make_encoder({0}), tokens_for({0}, 0), {0});
        ASSERT_EQ(rows.size(), 1);
        EXPECT_EQ(rows.front().size(), static_cast<size_t>(hidden));
    }
}

TEST(WhisperCrossKV, UniformCrossKVOutputsSelectCrossKVDecoder) {
    const SerializedDecoder serialized{DecoderOptions{}};
    auto decoder = decoder_from_path(serialized.path());
    ASSERT_NE(std::dynamic_pointer_cast<ov::genai::WhisperCrossKVDecoder>(decoder), nullptr);
    const Rows rows = step(*decoder, make_encoder({0}), tokens_for({0}, 0), {0});
    ASSERT_EQ(rows.size(), 1);
    EXPECT_EQ(rows.front().size(), static_cast<size_t>(hidden));
}

TEST(WhisperCrossKV, RowPlanKeepsEachSurvivorOnceAndMovesOnlyFromBeyondNewWidth) {
    for (size_t width = 1; width <= 10; ++width) {
        for (uint32_t mask = 0; mask < (1u << width); ++mask) {
            std::vector<bool> survivors(width);
            std::vector<int32_t> expected_rows;
            for (size_t row = 0; row < width; ++row) {
                survivors[row] = (mask >> row) & 1u;
                if (survivors[row]) {
                    expected_rows.push_back(static_cast<int32_t>(row));
                }
            }
            const std::vector<int32_t> plan = ov::genai::plan_surviving_decoder_rows(survivors);
            std::vector<int32_t> sorted_plan = plan;
            std::sort(sorted_plan.begin(), sorted_plan.end());
            ASSERT_EQ(sorted_plan, expected_rows) << "mask " << mask;
            for (size_t row = 0; row < plan.size(); ++row) {
                if (survivors[row]) {
                    ASSERT_EQ(plan[row], static_cast<int32_t>(row)) << "mask " << mask;
                } else {
                    ASSERT_GE(plan[row], static_cast<int32_t>(plan.size())) << "mask " << mask;
                }
            }
        }
    }
}

TEST(WhisperCrossKV, RoundGuardKeepsOriginalExceptionWhenCleanupFails) {
    class FailingDecoder : public ov::genai::WhisperDecoder {
    public:
        void start_async(const ov::Tensor&, const ov::Tensor&, const ov::Tensor&) override {}
        ov::Tensor wait() override {
            throw std::runtime_error("wait failed");
        }
        void reset_state() override {
            throw std::runtime_error("reset failed");
        }
    };

    ov::genai::Sampler sampler;
    try {
        ov::genai::DecoderRoundGuard guard{std::make_shared<FailingDecoder>(), sampler};
        guard.track_request(0);
        throw std::logic_error("original failure");
    } catch (const std::logic_error& error) {
        EXPECT_STREQ(error.what(), "original failure");
    }
}

class WhisperCrossKVDecoderTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        models_path =
            std::filesystem::temp_directory_path() /
            ("whisper_cross_kv_decoder_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(models_path);
        ov::serialize(make_whisper_decoder(),
                      (models_path / "openvino_decoder_model.xml").string(),
                      (models_path / "openvino_decoder_model.bin").string());
    }

    static void TearDownTestSuite() {
        std::filesystem::remove_all(models_path);
    }

    static std::shared_ptr<ov::genai::WhisperCrossKVDecoder> make_decoder() {
        auto decoder = ov::genai::WhisperCrossKVDecoder::create(models_path,
                                                                "CPU",
                                                                {},
                                                                ov::PartialShape{-1, frames, hidden},
                                                                false);
        EXPECT_NE(decoder, nullptr);
        return decoder;
    }

    static Rows decode_alone(size_t audio, size_t steps) {
        auto decoder = make_decoder();
        const ov::Tensor encoder = make_encoder({audio});
        Rows logits;
        for (size_t step_index = 0; step_index < steps; ++step_index) {
            logits.push_back(step(*decoder, encoder, tokens_for({audio}, step_index), {0}).front());
        }
        return logits;
    }

    static inline std::filesystem::path models_path;
};

TEST_F(WhisperCrossKVDecoderTest, RetainedRowsKeepSelfAndCrossAttentionAligned) {
    const std::vector<Rows> alone{decode_alone(0, 5), decode_alone(1, 5), decode_alone(2, 5)};
    auto expect_rows = [&](const Rows& rows, const std::vector<size_t>& audios, size_t step_index) {
        ASSERT_EQ(rows.size(), audios.size());
        for (size_t row = 0; row < rows.size(); ++row) {
            SCOPED_TRACE("step " + std::to_string(step_index) + ", row " + std::to_string(row));
            expect_near(rows[row], alone[audios[row]][step_index], 1e-3f);
        }
    };

    auto decoder = make_decoder();
    const ov::Tensor encoder = make_encoder({0, 1, 2});
    expect_rows(step(*decoder, encoder, tokens_for({0, 1, 2}, 0), {0, 1, 2}), {0, 1, 2}, 0);

    decoder->retain_rows({2, 1});
    expect_rows(step(*decoder, encoder, tokens_for({2, 1}, 1), {2, 1}), {2, 1}, 1);
    expect_rows(step(*decoder, encoder, tokens_for({2, 1}, 2), {0, 1}), {2, 1}, 2);

    decoder->retain_rows({1});
    expect_rows(step(*decoder, encoder, tokens_for({1}, 3), {1}), {1}, 3);
    expect_rows(step(*decoder, encoder, tokens_for({1}, 4), {0}), {1}, 4);
}

TEST_F(WhisperCrossKVDecoderTest, RejectsRowOperationsThatWouldMisalignState) {
    auto decoder = make_decoder();
    const ov::Tensor encoder = make_encoder({0, 1, 2});
    auto run = [&](const std::vector<int64_t>& tokens, const std::vector<int32_t>& beams) {
        return [&decoder, &encoder, tokens, beams] {
            step(*decoder, encoder, tokens, beams);
        };
    };
    auto retain = [&](const std::vector<int32_t>& source_rows) {
        return [&decoder, source_rows] {
            decoder->retain_rows(source_rows);
        };
    };

    expect_error(run(tokens_for({0, 1, 2}, 0), {1, 0, 2}), "keep their order");
    expect_error(run(tokens_for({0, 1, 2}, 0), {0, 1}), "beam_idx of shape [3]");
    step(*decoder, encoder, tokens_for({0, 1, 2}, 0), {0, 1, 2});

    expect_error(retain({2, 2}), "retained more than once");
    expect_error(retain({0, 3}), "cannot be filled from row 3");
    expect_error(retain({1, 0}), "at or beyond the new width");

    decoder->retain_rows({0, 2});
    expect_error(retain({0}), "already retained");
    expect_error(run(tokens_for({0, 2}, 1), {0, 1}), "source row passed to retain_rows");
    step(*decoder, encoder, tokens_for({0, 2}, 1), {0, 2});

    decoder->retain_rows({1});
    expect_error(run(tokens_for({0, 1, 2}, 2), {1, 1, 1}), "per-row cross-attention rows");
    expect_error(run(tokens_for({2}, 2), {0}), "source row passed to retain_rows");

    auto shared = make_decoder();
    step(*shared, make_encoder({1}), {3}, {0});
    expect_error(
        [&] {
            shared->retain_rows({0});
        },
        "shared by all decoder rows");
}

TEST_F(WhisperCrossKVDecoderTest, ResetWaitsForInferenceInFlight) {
    auto decoder = make_decoder();
    decoder->start_async(make_encoder({0, 1}), make_input_ids(tokens_for({0, 1}, 0)), make_beam_idx({0, 1}));
    decoder->reset_state();

    const Rows rows = step(*decoder, make_encoder({2}), tokens_for({2}, 0), {0});
    expect_near(rows.front(), decode_alone(2, 1).front(), 1e-3f);
}

}  // namespace
