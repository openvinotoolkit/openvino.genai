// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset15.hpp>

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_fused_predictor.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"

using namespace ov::genai;
using namespace ov::genai::qwen3_tts;

namespace {

// Builds a synthetic fused code predictor IR conforming to the locked contract:
//   Inputs:
//     past_hidden:   [-1, 1, H] (f32)
//     first_code:    [-1, 1] (i64)
//     tts_pad_embed: [1, 1, H] (f32)
//     (optional) uniform_noise: [-1, G-1, V_sub] (f32)
//     (optional) temperature:   [1] (f32)
//   Outputs:
//     codes:      [-1, G] (i64)   — [first_code, first_code + 1 + noise_shift, ...]
//     next_embed: [-1, 1, H] (f32) — past_hidden + float(first_code) + tts_pad_embed
std::shared_ptr<ov::Model> make_synthetic_fused_predictor(size_t num_groups,
                                                          size_t hidden_size,
                                                          bool with_sampling,
                                                          size_t sub_vocab = 16) {
    auto past_hidden =
        std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, static_cast<int64_t>(hidden_size)});
    past_hidden->set_friendly_name("past_hidden");
    past_hidden->output(0).get_tensor().set_names({"past_hidden"});

    auto first_code = std::make_shared<ov::opset15::Parameter>(ov::element::i64, ov::PartialShape{-1, 1});
    first_code->set_friendly_name("first_code");
    first_code->output(0).get_tensor().set_names({"first_code"});

    auto tts_pad =
        std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::PartialShape{1, 1, static_cast<int64_t>(hidden_size)});
    tts_pad->set_friendly_name("tts_pad_embed");
    tts_pad->output(0).get_tensor().set_names({"tts_pad_embed"});

    ov::ParameterVector params{past_hidden, first_code, tts_pad};

    // Offsets [0, 1, ..., G-1] shaped [1, G]
    std::vector<int64_t> offsets(num_groups);
    for (size_t g = 0; g < num_groups; ++g) {
        offsets[g] = static_cast<int64_t>(g);
    }
    auto offset_const = ov::opset15::Constant::create(ov::element::i64, ov::Shape{1, num_groups}, offsets);
    std::shared_ptr<ov::Node> codes_node = std::make_shared<ov::opset15::Add>(first_code, offset_const);

    if (with_sampling) {
        auto uniform_noise = std::make_shared<ov::opset15::Parameter>(
            ov::element::f32,
            ov::PartialShape{-1, static_cast<int64_t>(num_groups - 1), static_cast<int64_t>(sub_vocab)});
        uniform_noise->set_friendly_name("uniform_noise");
        uniform_noise->output(0).get_tensor().set_names({"uniform_noise"});

        auto temperature = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::PartialShape{1});
        temperature->set_friendly_name("temperature");
        temperature->output(0).get_tensor().set_names({"temperature"});

        params.push_back(uniform_noise);
        params.push_back(temperature);

        // Mirror optimum-intel's `temp = max(temperature, 1e-6); scores = logits / temp + gumbel`:
        // - Penalize overflow if `1 / max(temperature, 1e-6) > 1000` (catches GPU FP16 temp=0 bug!).
        // - Add `floor(mean(|uniform_noise - 0.5|, axis=2) * 40 / max(temperature, 1e-6))` to residual groups:
        //   when greedy (`uniform_noise == 0.5f` and `temperature == 1.0f`), shift is identically 0;
        //   when sampled (`uniform_noise ~ U(0,1)`), shift > 0.
        auto eps = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1}, {1e-6f});
        auto safe_temp = std::make_shared<ov::opset15::Maximum>(temperature, eps);
        auto one_f = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
        auto inv_temp = std::make_shared<ov::opset15::Divide>(one_f, safe_temp);
        auto overflow_thresh = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1}, {1000.0f});
        auto zero_f = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1}, {0.0f});
        auto overflow_penalty = std::make_shared<ov::opset15::Maximum>(
            std::make_shared<ov::opset15::Subtract>(inv_temp, overflow_thresh), zero_f);

        auto half = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1}, {0.5f});
        auto dev = std::make_shared<ov::opset15::Abs>(std::make_shared<ov::opset15::Subtract>(uniform_noise, half));
        auto axis2 = ov::opset15::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto noise_dev_mean = std::make_shared<ov::opset15::ReduceMean>(dev, axis2, false);  // [B, G-1]
        auto forty = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1}, {40.0f});
        auto sample_shift = std::make_shared<ov::opset15::Multiply>(
            std::make_shared<ov::opset15::Multiply>(noise_dev_mean, forty), inv_temp);
        auto total_shift = std::make_shared<ov::opset15::Add>(sample_shift, overflow_penalty);
        auto shift_i64 = std::make_shared<ov::opset15::Convert>(total_shift, ov::element::i64);  // [B, G-1]

        // Prepend 0 for group 0 so first_code is unchanged.
        auto zero_col = std::make_shared<ov::opset15::Multiply>(
            first_code, ov::opset15::Constant::create(ov::element::i64, ov::Shape{1, 1}, {0}));
        auto full_shift = std::make_shared<ov::opset15::Concat>(ov::OutputVector{zero_col, shift_i64}, 1);
        codes_node = std::make_shared<ov::opset15::Add>(codes_node, full_shift);
    }

    auto codes_result = std::make_shared<ov::opset15::Result>(codes_node);
    codes_result->output(0).get_tensor().set_names({"codes"});

    // next_embed = past_hidden + unsqueeze(float(first_code), 2) + tts_pad_embed
    auto fc_f32 = std::make_shared<ov::opset15::Convert>(first_code, ov::element::f32);
    auto unsq_axis = ov::opset15::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto fc_3d = std::make_shared<ov::opset15::Unsqueeze>(fc_f32, unsq_axis);  // [B, 1, 1]
    auto sum1 = std::make_shared<ov::opset15::Add>(past_hidden, fc_3d);
    auto next_embed = std::make_shared<ov::opset15::Add>(sum1, tts_pad);
    auto next_embed_result = std::make_shared<ov::opset15::Result>(next_embed);
    next_embed_result->output(0).get_tensor().set_names({"next_embed"});

    return std::make_shared<ov::Model>(ov::ResultVector{codes_result, next_embed_result}, params, "synthetic_fused_cp");
}

}  // namespace

TEST(TestQwen3TTSFusedAndMultiDevice, multi_device_routing_precedence_and_property_stripping) {
    // 1. Shorthand decoder_device & encoder_device for GPU Talker + NPU Decoder + CPU Encoders.
    ov::AnyMap props{
        decoder_device("NPU"),
        encoder_device("CPU"),
        {"CACHE_DIR", "/tmp/cache"},
        ov::device::properties(ov::AnyMap{
            {"NPU", ov::AnyMap{{"NPU_COMPILER_TYPE", "DRIVER"}}},
            {"GPU", ov::AnyMap{{"INFERENCE_PRECISION_HINT", "f16"}}},
        }),
    };

    auto [talker_dev, talker_props] = resolve_component_target("GPU", props, roles::TALKER);
    EXPECT_EQ(talker_dev, "GPU");
    EXPECT_EQ(talker_props.count("decoder_device"), 0u);
    EXPECT_EQ(talker_props.count("encoder_device"), 0u);
    EXPECT_EQ(talker_props.count("device_mapping"), 0u);
    EXPECT_EQ(talker_props.at("INFERENCE_PRECISION_HINT").as<std::string>(), "f16");
    EXPECT_EQ(talker_props.at("CACHE_DIR").as<std::string>(), "/tmp/cache");

    auto [dec_dev, dec_props] = resolve_component_target("GPU", props, roles::SPEECH_TOKENIZER_DECODER);
    EXPECT_EQ(dec_dev, "NPU");
    EXPECT_EQ(dec_props.at("NPU_COMPILER_TYPE").as<std::string>(), "DRIVER");
    EXPECT_EQ(dec_props.count("decoder_device"), 0u);

    auto [enc_dev, enc_props] = resolve_component_target("GPU", props, roles::SPEECH_TOKENIZER_ENCODER);
    EXPECT_EQ(enc_dev, "CPU");
    EXPECT_EQ(enc_props.count("encoder_device"), 0u);

    auto [spk_dev, spk_props] = resolve_component_target("GPU", props, roles::SPEAKER_ENCODER);
    EXPECT_EQ(spk_dev, "CPU");

    // 2. Explicit device_mapping overrides decoder_device, and code_predictor_model aliases code_predictor_fused.
    std::map<std::string, std::string> mapping{
        {roles::SPEECH_TOKENIZER_DECODER, "CPU"},
        {roles::CODE_PREDICTOR, "NPU"},
    };
    ov::AnyMap props_override{
        decoder_device("GPU"),
        device_mapping(mapping),
    };
    auto [dec_dev2, dec_props2] = resolve_component_target("GPU", props_override, roles::SPEECH_TOKENIZER_DECODER);
    EXPECT_EQ(dec_dev2, "CPU");
    EXPECT_EQ(dec_props2.count("device_mapping"), 0u);
    EXPECT_EQ(dec_props2.count("decoder_device"), 0u);

    auto [fused_dev, fused_props] = resolve_component_target("GPU", props_override, roles::CODE_PREDICTOR_FUSED);
    EXPECT_EQ(fused_dev, "NPU");

    // 3. Python-style ov::AnyMap device_mapping also works.
    ov::AnyMap py_props{
        {"device_mapping", ov::AnyMap{{roles::CODE_PREDICTOR_FUSED, std::string("CPU")}}},
    };
    auto [fused_dev_py, _] = resolve_component_target("GPU", py_props, roles::CODE_PREDICTOR_FUSED);
    EXPECT_EQ(fused_dev_py, "CPU");

    // 4. Inherited GPU-only and LLM-only top-level hints must not leak to NPU/CPU codec/encoder components,
    //    while remaining intact on GPU Talker and when explicitly set under per-device/per-model properties.
    ov::AnyMap hetero_props{
        decoder_device("NPU"),
        encoder_device("CPU"),
        {"CACHE_DIR", "/tmp/cache"},
        ov::hint::kv_cache_precision(ov::element::u8),
        ov::hint::dynamic_quantization_group_size(32),
        {"ACTIVATIONS_SCALE_FACTOR", 8.0f},
        {"GPU_HOST_TASK_PRIORITY", "HIGH"},
        ov::device::properties(ov::AnyMap{
            {"NPU", ov::AnyMap{{"NPU_COMPILER_TYPE", "DRIVER"}}},
        }),
    };

    auto [ht_talker_dev, ht_talker_props] = resolve_component_target("GPU", hetero_props, roles::TALKER);
    EXPECT_EQ(ht_talker_dev, "GPU");
    EXPECT_EQ(ht_talker_props.count(ov::hint::kv_cache_precision.name()), 1u);
    EXPECT_EQ(ht_talker_props.count(ov::hint::dynamic_quantization_group_size.name()), 1u);
    EXPECT_EQ(ht_talker_props.count("ACTIVATIONS_SCALE_FACTOR"), 1u);
    EXPECT_EQ(ht_talker_props.count("GPU_HOST_TASK_PRIORITY"), 1u);

    auto [ht_dec_dev, ht_dec_props] = resolve_component_target("GPU", hetero_props, roles::SPEECH_TOKENIZER_DECODER);
    EXPECT_EQ(ht_dec_dev, "NPU");
    EXPECT_EQ(ht_dec_props.count("ACTIVATIONS_SCALE_FACTOR"), 0u);
    EXPECT_EQ(ht_dec_props.count("GPU_HOST_TASK_PRIORITY"), 0u);
    EXPECT_EQ(ht_dec_props.count(ov::hint::kv_cache_precision.name()), 0u);
    EXPECT_EQ(ht_dec_props.count(ov::hint::dynamic_quantization_group_size.name()), 0u);
    EXPECT_EQ(ht_dec_props.at("NPU_COMPILER_TYPE").as<std::string>(), "DRIVER");
    EXPECT_EQ(ht_dec_props.at("CACHE_DIR").as<std::string>(), "/tmp/cache");

    auto [ht_enc_dev, ht_enc_props] = resolve_component_target("GPU", hetero_props, roles::SPEECH_TOKENIZER_ENCODER);
    EXPECT_EQ(ht_enc_dev, "CPU");
    EXPECT_EQ(ht_enc_props.count("ACTIVATIONS_SCALE_FACTOR"), 0u);
    EXPECT_EQ(ht_enc_props.count("GPU_HOST_TASK_PRIORITY"), 0u);
    EXPECT_EQ(ht_enc_props.count(ov::hint::kv_cache_precision.name()), 0u);
    EXPECT_EQ(ht_enc_props.count(ov::hint::dynamic_quantization_group_size.name()), 0u);
}

TEST(TestQwen3TTSFusedAndMultiDevice, fused_predictor_single_and_batch_greedy_and_sampled) {
    constexpr size_t G = 16;
    constexpr size_t H = 8;
    auto model = make_synthetic_fused_predictor(G, H, /*with_sampling=*/true, /*sub_vocab=*/16);

    Qwen3TTSFusedCodePredictor predictor;
    ASSERT_TRUE(predictor.initialize_from_model(model, "CPU", {}, G, H));
    EXPECT_TRUE(predictor.is_available());
    EXPECT_TRUE(predictor.has_tts_pad_embed());
    EXPECT_TRUE(predictor.supports_sampling());
    EXPECT_EQ(predictor.sub_vocab_size(), 16u);

    std::unordered_map<std::string, double> perf_ms;
    std::unordered_map<std::string, int64_t> perf_calls;
    std::mt19937 rng(42);

    ov::Tensor past_hidden(ov::element::f32, ov::Shape{1, 1, H});
    ov::Tensor tts_pad(ov::element::f32, ov::Shape{1, 1, H});
    for (size_t h = 0; h < H; ++h) {
        past_hidden.data<float>()[h] = static_cast<float>(h) * 0.5f;
        tts_pad.data<float>()[h] = 10.0f + static_cast<float>(h);
    }

    // 1. Greedy single-row with in-graph tts_pad_embed
    auto greedy = predictor.predict_single(
        past_hidden, /*first_code=*/100, tts_pad, /*additive_is_tts_pad=*/true,
        /*do_sample=*/false, /*temperature=*/0.9f, rng, /*allow_device_zero_copy=*/false, perf_ms, perf_calls);
    ASSERT_EQ(greedy.codes.size(), G);
    for (size_t g = 0; g < G; ++g) {
        EXPECT_EQ(greedy.codes[g], static_cast<int64_t>(100 + g));
    }
    ASSERT_EQ(greedy.next_embed.get_shape(), (ov::Shape{1, 1, H}));
    for (size_t h = 0; h < H; ++h) {
        EXPECT_NEAR(greedy.next_embed.data<const float>()[h],
                    past_hidden.data<const float>()[h] + 100.0f + tts_pad.data<const float>()[h],
                    1e-4f);
    }

    // 2. Host additive override (trailing_text_hidden mode: zeros passed to tts_pad_embed, override added on host)
    ov::Tensor trailing_step(ov::element::f32, ov::Shape{1, 1, H});
    for (size_t h = 0; h < H; ++h) {
        trailing_step.data<float>()[h] = -5.0f;
    }
    auto stream_text = predictor.predict_single(
        past_hidden, /*first_code=*/20, trailing_step, /*additive_is_tts_pad=*/false,
        /*do_sample=*/false, /*temperature=*/0.0f, rng, /*allow_device_zero_copy=*/false, perf_ms, perf_calls);
    for (size_t h = 0; h < H; ++h) {
        EXPECT_NEAR(stream_text.next_embed.data<const float>()[h],
                    past_hidden.data<const float>()[h] + 20.0f - 5.0f,
                    1e-4f);
    }

    // 3. Sampled single-row: temperature > 0 shifts residual groups via uniform_noise while preserving group 0.
    auto sampled = predictor.predict_single(
        past_hidden, /*first_code=*/50, tts_pad, /*additive_is_tts_pad=*/true,
        /*do_sample=*/true, /*temperature=*/1.0f, rng, /*allow_device_zero_copy=*/false, perf_ms, perf_calls);
    EXPECT_EQ(sampled.codes[0], 50);
    bool any_shifted = false;
    for (size_t g = 1; g < G; ++g) {
        if (sampled.codes[g] > static_cast<int64_t>(50 + g)) {
            any_shifted = true;
            break;
        }
    }
    EXPECT_TRUE(any_shifted);

    // 4. Batched prediction (B=3) with per-row additive overrides
    constexpr size_t B = 3;
    ov::Tensor ph_batch(ov::element::f32, ov::Shape{B, 1, H});
    std::vector<int64_t> first_codes = {10, 20, 30};
    std::vector<std::vector<float>> row_adds(B, std::vector<float>(H));
    std::vector<const float*> row_add_ptrs(B);
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            ph_batch.data<float>()[b * H + h] = static_cast<float>(b + 1) + 0.25f * static_cast<float>(h);
            row_adds[b][h] = static_cast<float>((b + 1) * 100);
        }
        row_add_ptrs[b] = row_adds[b].data();
    }

    auto batch_res = predictor.predict_batch(
        ph_batch, first_codes, /*shared_tts_pad=*/ov::Tensor{}, row_add_ptrs,
        /*do_sample=*/false, /*temperature=*/0.0f, rng, perf_ms, perf_calls);
    ASSERT_EQ(batch_res.codes.size(), B);
    ASSERT_EQ(batch_res.next_embeds.get_shape(), (ov::Shape{B, 1, H}));
    for (size_t b = 0; b < B; ++b) {
        ASSERT_EQ(batch_res.codes[b].size(), G);
        for (size_t g = 0; g < G; ++g) {
            EXPECT_EQ(batch_res.codes[b][g], first_codes[b] + static_cast<int64_t>(g));
        }
        for (size_t h = 0; h < H; ++h) {
            const float expected = ph_batch.data<const float>()[b * H + h] +
                                   static_cast<float>(first_codes[b]) +
                                   row_adds[b][h];
            EXPECT_NEAR(batch_res.next_embeds.data<const float>()[b * H + h], expected, 1e-4f);
        }
    }
}
