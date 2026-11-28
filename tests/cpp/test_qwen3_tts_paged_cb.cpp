// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset15.hpp>

#include "continuous_batching/pipeline_impl.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"

using namespace ov::genai;

namespace {

constexpr size_t kHiddenSize = 8;
constexpr size_t kVocabSize = 8;
constexpr int64_t kEosTokenId = 2;

// Synthetic PagedAttention-shaped embedding model with 3-row M-RoPE `position_ids`,
// `sampled_tokens_indices` gather, `key_cache.0`/`value_cache.0`, and `last_hidden_state` export.
//
// Token selection rule:
//   - `logits[i, v] = -|inputs_embeds[sampled_i, 0] - float(v)|` (peak at nearest integer vocab id!)
//   - `last_hidden_state[t, 0, h] = inputs_embeds[t, h] * 2.0 + 1.0`
// This lets the test drive exact deterministic trajectories via PostSampleEmbedCallback!
std::shared_ptr<ov::Model> make_synthetic_pa_embed_model(size_t block_size = 32) {
    ov::ParameterVector params;
    auto add_param = [&](const std::string& name, ov::element::Type et, const ov::PartialShape& ps) {
        auto p = std::make_shared<ov::opset15::Parameter>(et, ps);
        p->set_friendly_name(name);
        p->output(0).get_tensor().set_names({name});
        params.push_back(p);
        return p;
    };

    auto inputs_embeds = add_param("inputs_embeds", ov::element::f32, ov::PartialShape{-1, static_cast<int64_t>(kHiddenSize)});
    auto position_ids = add_param("position_ids", ov::element::i64, ov::PartialShape{3, -1});
    auto past_lens = add_param("past_lens", ov::element::i32, ov::PartialShape{-1});
    auto subseq_begins = add_param("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
    auto block_indices = add_param("block_indices", ov::element::i32, ov::PartialShape{-1});
    auto block_indices_begins = add_param("block_indices_begins", ov::element::i32, ov::PartialShape{-1});
    auto max_context_len = add_param("max_context_len", ov::element::i32, ov::PartialShape{});
    auto sampled_indices = add_param("sampled_tokens_indices", ov::element::i64, ov::PartialShape{-1});
    auto key_cache = add_param("key_cache.0", ov::element::f32, ov::PartialShape{-1, 2, static_cast<int64_t>(block_size), 4});
    auto val_cache = add_param("value_cache.0", ov::element::f32, ov::PartialShape{-1, 2, static_cast<int64_t>(block_size), 4});
    (void)position_ids;
    (void)past_lens;
    (void)subseq_begins;
    (void)block_indices;
    (void)block_indices_begins;
    (void)max_context_len;
    (void)key_cache;
    (void)val_cache;

    // last_hidden_state = unsqueeze(inputs_embeds * 2.0 + 1.0, axis=1) -> [total_tokens, 1, H]
    auto two = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f});
    auto one = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    auto hs_2d = std::make_shared<ov::opset15::Add>(std::make_shared<ov::opset15::Multiply>(inputs_embeds, two), one);
    auto axis1 = ov::opset15::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto hs_3d = std::make_shared<ov::opset15::Unsqueeze>(hs_2d, axis1);
    auto hs_res = std::make_shared<ov::opset15::Result>(hs_3d);
    hs_res->output(0).get_tensor().set_names({"last_hidden_state"});

    // Gather sampled rows from inputs_embeds along axis 0 -> [num_sampled, H]
    auto axis0 = ov::opset15::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto gathered = std::make_shared<ov::opset15::Gather>(inputs_embeds, sampled_indices, axis0);
    // Take channel 0 -> [num_sampled, 1]
    auto col0_idx = ov::opset15::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto col0 = std::make_shared<ov::opset15::Gather>(gathered, col0_idx, axis1);

    std::vector<float> vocab_vals(kVocabSize);
    for (size_t v = 0; v < kVocabSize; ++v) {
        vocab_vals[v] = static_cast<float>(v);
    }
    auto vocab_const = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1, kVocabSize}, vocab_vals);
    auto diff = std::make_shared<ov::opset15::Subtract>(col0, vocab_const);
    auto abs_diff = std::make_shared<ov::opset15::Abs>(diff);
    auto neg_one = ov::opset15::Constant::create(ov::element::f32, ov::Shape{1}, {-1.0f});
    auto logits_2d = std::make_shared<ov::opset15::Multiply>(abs_diff, neg_one);
    // PagedAttention logits output has rank 3: [1, num_sampled, V]
    auto logits = std::make_shared<ov::opset15::Unsqueeze>(logits_2d, axis0);
    auto logits_res = std::make_shared<ov::opset15::Result>(logits);
    logits_res->output(0).get_tensor().set_names({"logits"});

    return std::make_shared<ov::Model>(ov::ResultVector{logits_res, hs_res}, params, "synthetic_pa_qwen3_talker");
}

ov::Tensor make_prompt_embeds(size_t prompt_len, float last_token_target_id, float prefix_seed = 0.0f) {
    ov::Tensor t(ov::element::f32, ov::Shape{1, prompt_len, kHiddenSize});
    float* ptr = t.data<float>();
    for (size_t pos = 0; pos < prompt_len; ++pos) {
        const float val = (pos + 1 == prompt_len) ? last_token_target_id : (prefix_seed + static_cast<float>(pos) * 0.01f);
        for (size_t h = 0; h < kHiddenSize; ++h) {
            ptr[pos * kHiddenSize + h] = (h == 0) ? val : static_cast<float>(h) * 0.1f;
        }
    }
    return t;
}

ov::Tensor make_single_embed(float target_id) {
    ov::Tensor t(ov::element::f32, ov::Shape{1, 1, kHiddenSize});
    float* ptr = t.data<float>();
    ptr[0] = target_id;
    for (size_t h = 1; h < kHiddenSize; ++h) {
        ptr[h] = static_cast<float>(h) * 0.1f;
    }
    return t;
}

}  // namespace

TEST(TestQwen3TTSPagedCB, post_sample_embed_callback_concurrent_batch_chunked_prefill_and_suppression) {
    auto model = make_synthetic_pa_embed_model(/*block_size=*/32);

    SchedulerConfig sched_cfg;
    sched_cfg.enable_prefix_caching = true;
    sched_cfg.dynamic_split_fuse = true;
    sched_cfg.max_num_batched_tokens = 4;  // Force chunked prefill on 7-token and 5-token prompts!
    sched_cfg.num_kv_blocks = 16;

    GenerationConfig default_cfg;
    default_cfg.do_sample = false;
    default_cfg.eos_token_id = kEosTokenId;
    default_cfg.stop_token_ids = {kEosTokenId};
    default_cfg.max_new_tokens = 16;

    // Suppress token id 3: when input channel 0 == 3.1f, nearest unsuppressed token must be 4 (or 5 if 3 & 4 suppressed).
    std::set<int64_t> suppressed_ids = {3, 4};

    // Trajectory per request:
    //   req 0: prompt targets 3.9 -> suppressed {3,4}, nearest unsuppressed is 5 (|3.9-5|=1.1 < |3.9-2|=1.9)!
    //          then callback feeds 6.0 -> samples 6 -> feeds 2.0 (EOS) -> samples 2 (done).
    //   req 1: prompt targets 1.0 -> samples 1 -> callback feeds 7.0 -> samples 7 -> feeds 0.0 -> samples 0 -> feeds 2.0 (EOS).
    std::unordered_map<uint64_t, std::vector<float>> feedback_targets{
        {0, {6.0f, 2.0f}},
        {1, {7.0f, 0.0f, 2.0f}},
    };
    std::unordered_map<uint64_t, std::vector<int64_t>> observed_tokens;
    std::unordered_map<uint64_t, std::vector<float>> observed_hs0;

    PostSampleEmbedCallback cb = [&](const std::vector<StepSampledSequence>& batch) -> std::vector<StepNextEmbed> {
        std::vector<StepNextEmbed> out(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& item = batch[i];
            observed_tokens[item.request_id].push_back(item.sampled_token_id);
            EXPECT_TRUE(static_cast<bool>(item.last_hidden_state));
            EXPECT_EQ(item.last_hidden_state.get_shape(), (ov::Shape{1, 1, kHiddenSize}));
            if (item.last_hidden_state && item.last_hidden_state.get_size() > 0) {
                observed_hs0[item.request_id].push_back(item.last_hidden_state.data<const float>()[0]);
            }

            if (!item.is_finished) {
                const auto& seq_targets = feedback_targets.at(item.request_id);
                const float next_val =
                    (item.generated_index < seq_targets.size()) ? seq_targets[item.generated_index] : static_cast<float>(kEosTokenId);
                out[i].next_embed = make_single_embed(next_val);
                out[i].next_position_ids =
                    qwen3_tts::make_position_ids_decode(item.prompt_len + item.generated_index);
            }
        }
        return out;
    };

    EmbeddingContinuousBatchingPipeline cb_impl(
        model, Tokenizer{}, sched_cfg, "CPU", {}, default_cfg, cb, suppressed_ids, /*enable_hidden_state_export=*/true);

    // Submit 2 concurrent requests of different prompt lengths (7 and 5) with 3-row M-RoPE position_ids.
    auto h0 = cb_impl.add_request_with_embeddings(
        /*request_id=*/0, make_prompt_embeds(7, 3.9f), qwen3_tts::make_position_ids_prefill(7), default_cfg);
    auto h1 = cb_impl.add_request_with_embeddings(
        /*request_id=*/1, make_prompt_embeds(5, 1.0f), qwen3_tts::make_position_ids_prefill(5), default_cfg);

    size_t step_count = 0;
    while (cb_impl.has_non_finished_requests()) {
        cb_impl.step();
        ++step_count;
        ASSERT_LT(step_count, 50u);
    }

    // Verify chunked prefill actually happened (7 + 5 = 12 prompt tokens at max_num_batched_tokens=4 => >= 3 prefill steps + decode steps).
    EXPECT_GE(step_count, 5u);

    // Verify suppressed tokens {3, 4} were never sampled and exact expected trajectories ran.
    ASSERT_EQ(observed_tokens[0], (std::vector<int64_t>{5, 6, 2}));
    ASSERT_EQ(observed_tokens[1], (std::vector<int64_t>{1, 7, 0, 2}));

    // Verify last_hidden_state export formula (inputs_embeds[0] * 2 + 1) matched at every step!
    ASSERT_EQ(observed_hs0[0].size(), 3u);
    EXPECT_NEAR(observed_hs0[0][0], 3.9f * 2.0f + 1.0f, 1e-4f);
    EXPECT_NEAR(observed_hs0[0][1], 6.0f * 2.0f + 1.0f, 1e-4f);
    EXPECT_NEAR(observed_hs0[0][2], 2.0f * 2.0f + 1.0f, 1e-4f);

    ASSERT_EQ(observed_hs0[1].size(), 4u);
    EXPECT_NEAR(observed_hs0[1][0], 1.0f * 2.0f + 1.0f, 1e-4f);
    EXPECT_NEAR(observed_hs0[1][1], 7.0f * 2.0f + 1.0f, 1e-4f);
    EXPECT_NEAR(observed_hs0[1][2], 0.0f * 2.0f + 1.0f, 1e-4f);
    EXPECT_NEAR(observed_hs0[1][3], 2.0f * 2.0f + 1.0f, 1e-4f);

    (void)h0;
    (void)h1;
}

TEST(TestQwen3TTSPagedCB, prefix_caching_reuses_shared_embedding_blocks_and_early_stop_works) {
    auto model = make_synthetic_pa_embed_model(/*block_size=*/32);

    SchedulerConfig sched_cfg;
    sched_cfg.enable_prefix_caching = true;
    sched_cfg.dynamic_split_fuse = true;
    sched_cfg.max_num_batched_tokens = 128;
    sched_cfg.num_kv_blocks = 32;

    GenerationConfig default_cfg;
    default_cfg.do_sample = false;
    default_cfg.eos_token_id = kEosTokenId;
    default_cfg.stop_token_ids = {kEosTokenId};
    default_cfg.max_new_tokens = 20;

    size_t stop_after_frames = 2;
    size_t req2_frames = 0;

    PostSampleEmbedCallback cb = [&](const std::vector<StepSampledSequence>& batch) -> std::vector<StepNextEmbed> {
        std::vector<StepNextEmbed> out(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& item = batch[i];
            if (item.request_id == 0) {
                // Request 0 stops immediately after 1 frame via EOS.
                if (!item.is_finished) {
                    out[i].next_embed = make_single_embed(static_cast<float>(kEosTokenId));
                }
            } else {
                // Request 1 tests early stop via `stop_sequence = true` after 2 frames (even though token != EOS).
                ++req2_frames;
                if (req2_frames >= stop_after_frames) {
                    out[i].stop_sequence = true;
                } else {
                    out[i].next_embed = make_single_embed(5.0f);
                }
            }
        }
        return out;
    };

    EmbeddingContinuousBatchingPipeline cb_impl(
        model, Tokenizer{}, sched_cfg, "CPU", {}, default_cfg, cb, {}, /*enable_hidden_state_export=*/true);

    // 1. Run request 0 with a 40-token prompt (> 1 full block of 32 tokens) so block 0 is published to prefix cache.
    auto prompt0 = make_prompt_embeds(40, 1.0f, /*prefix_seed=*/0.25f);
    auto h0 = cb_impl.add_request_with_embeddings(0, prompt0, qwen3_tts::make_position_ids_prefill(40), default_cfg);
    while (cb_impl.has_non_finished_requests()) {
        cb_impl.step();
    }

    // 2. Submit request 1 sharing the exact first 39 embeddings (including the entire 32-token block 0!).
    auto prompt1 = make_prompt_embeds(40, 5.0f, /*prefix_seed=*/0.25f);
    auto h1 = cb_impl.add_request_with_embeddings(1, prompt1, qwen3_tts::make_position_ids_prefill(40), default_cfg);

    auto awaiting = cb_impl.get_awaiting_requests();
    ASSERT_EQ(awaiting.size(), 1u);
    EXPECT_GE(awaiting[0]->get_num_prefix_cache_hit_tokens(), 32u);

    // 3. Run request 1 and verify early stop terminates immediately after `stop_after_frames` (2 frames, << max_new_tokens=20).
    size_t steps = 0;
    while (cb_impl.has_non_finished_requests()) {
        cb_impl.step();
        ++steps;
        ASSERT_LT(steps, 10u);
    }
    EXPECT_EQ(req2_frames, stop_after_frames);
    (void)h0;
    (void)h1;
}
