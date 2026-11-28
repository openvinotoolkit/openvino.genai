// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/tile.hpp>

#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/speech_generation/audio_streamer_base.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_stream_decoder.hpp"

using namespace ov::genai;
using namespace ov::genai::qwen3_tts;

namespace {

// Builds a synthetic static streaming codec decoder:
// input:  audio_codes [1, Q, C + T] (i64)
// output: waveform    [1, 1, T * upsample] (f32) where each emitted frame t fills `upsample` copies of
//         the group-0 code at frame (C + t) (simulating baked left-context trim!).
std::shared_ptr<ov::Model> make_synthetic_static_stream_decoder(size_t num_quantizers,
                                                                size_t left_ctx,
                                                                size_t chunk_frames,
                                                                size_t upsample) {
    const size_t total_frames = left_ctx + chunk_frames;
    auto codes = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{1, static_cast<int64_t>(num_quantizers), static_cast<int64_t>(total_frames)});
    codes->set_friendly_name("audio_codes");
    codes->output(0).get_tensor().set_names({"audio_codes"});

    // Slice group 0 and frames [left_ctx, left_ctx + chunk_frames) -> [1, 1, chunk_frames]
    auto start = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 0, static_cast<int64_t>(left_ctx)});
    auto stop = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, static_cast<int64_t>(total_frames)});
    auto step = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
    auto sliced = std::make_shared<ov::op::v8::Slice>(codes, start, stop, step);
    auto as_f32 = std::make_shared<ov::op::v0::Convert>(sliced, ov::element::f32);

    // Reshape [1, 1, chunk_frames, 1] -> Tile [1, 1, 1, upsample] -> Reshape [1, 1, chunk_frames * upsample]
    auto r4 = std::make_shared<ov::op::v1::Reshape>(
        as_f32,
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, static_cast<int64_t>(chunk_frames), 1}),
        false);
    auto tiled = std::make_shared<ov::op::v0::Tile>(
        r4,
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, static_cast<int64_t>(upsample)}));
    auto wav = std::make_shared<ov::op::v1::Reshape>(
        tiled,
        ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, static_cast<int64_t>(chunk_frames * upsample)}),
        false);
    wav->set_friendly_name("waveform");
    wav->output(0).get_tensor().set_names({"waveform"});

    return std::make_shared<ov::Model>(ov::OutputVector{wav}, ov::ParameterVector{codes});
}

class RecordingAudioStreamer : public AudioStreamerBase {
public:
    StreamingStatus status_after_chunks = StreamingStatus::RUNNING;
    size_t stop_after_chunk_count = SIZE_MAX;
    bool throw_on_write = false;

    std::vector<size_t> chunk_sizes;
    std::vector<float> all_samples;
    std::atomic<int> end_calls{0};

    StreamingStatus write(const ov::Tensor& audio_chunk) override {
        if (throw_on_write) {
            throw std::runtime_error("synthetic streamer write failure");
        }
        const auto shape = audio_chunk.get_shape();
        EXPECT_EQ(shape.size(), 3u);
        EXPECT_EQ(shape[0], 1u);
        EXPECT_EQ(shape[1], 1u);
        const float* ptr = audio_chunk.data<const float>();
        chunk_sizes.push_back(shape[2]);
        all_samples.insert(all_samples.end(), ptr, ptr + shape[2]);
        if (chunk_sizes.size() >= stop_after_chunk_count) {
            return status_after_chunks;
        }
        return StreamingStatus::RUNNING;
    }

    void end() override {
        end_calls.fetch_add(1);
    }
};

}  // namespace

TEST(TestQwen3TTSStreaming, audio_streamer_aliases_omni_speech_streamer) {
    static_assert(std::is_same_v<AudioStreamerBase, OmniSpeechStreamerBase>,
                  "OmniSpeechStreamerBase must alias AudioStreamerBase");
    static_assert(std::is_same_v<AudioStreamerVariant, OmniSpeechStreamerVariant>,
                  "OmniSpeechStreamerVariant must alias AudioStreamerVariant");
    auto prop = audio_streamer(std::monostate{});
    EXPECT_EQ(prop.first, "audio_streamer");
}

TEST(TestQwen3TTSStreaming, discover_stream_decoders_from_openvino_config_and_filenames) {
    const auto tmp_dir = std::filesystem::temp_directory_path() / "ov_genai_qwen3_stream_discover_test";
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);

    // Create placeholder files for discovery
    std::ofstream(tmp_dir / "openvino_codec_decoder_stream_c0_t8.xml") << "<net/>";
    std::ofstream(tmp_dir / "openvino_codec_decoder_stream_c25_t12.xml") << "<net/>";
    std::ofstream(tmp_dir / "openvino_codec_decoder_stream_c25_t24.xml") << "<net/>";

    // 1. Filename fallback discovery (no openvino_config.json yet)
    auto specs = discover_stream_decoder_specs(tmp_dir);
    ASSERT_EQ(specs.size(), 3u);
    EXPECT_EQ(specs[0].left_context_frames, 0u);
    EXPECT_EQ(specs[0].chunk_frames, 8u);
    EXPECT_EQ(specs[1].left_context_frames, 25u);
    EXPECT_EQ(specs[1].chunk_frames, 12u);
    EXPECT_EQ(specs[2].left_context_frames, 25u);
    EXPECT_EQ(specs[2].chunk_frames, 24u);

    // 2. Discovery via openvino_config.json streaming_codec_decoders
    {
        std::ofstream cfg(tmp_dir / "openvino_config.json");
        cfg << R"({
            "streaming_codec_decoders": [
                {"file": "openvino_codec_decoder_stream_c0_t8.xml", "left_context": 0, "chunk_frames": 8},
                {"file": "openvino_codec_decoder_stream_c25_t24.xml", "left_context": 25, "chunk_frames": 24}
            ]
        })";
    }
    specs = discover_stream_decoder_specs(tmp_dir);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].chunk_frames, 8u);
    EXPECT_EQ(specs[1].chunk_frames, 24u);

    std::filesystem::remove_all(tmp_dir);
}

TEST(TestQwen3TTSStreaming, two_tier_static_stream_decoder_sync_and_async_parity) {
    constexpr size_t Q = 16;
    constexpr size_t UPSAMPLE = 4;  // small upsample for fast unit test
    ov::Core core;

    Qwen3TTSStreamDecoderPool pool;
    pool.set_metadata(Q, UPSAMPLE);
    pool.register_compiled_tier(
        0, 8, core.compile_model(make_synthetic_static_stream_decoder(Q, 0, 8, UPSAMPLE), "CPU").create_infer_request());
    pool.register_compiled_tier(
        25, 12, core.compile_model(make_synthetic_static_stream_decoder(Q, 25, 12, UPSAMPLE), "CPU").create_infer_request());

    for (bool async_mode : {false, true}) {
        SpeechGenerationConfig cfg;
        cfg.first_chunk_frames = 8;
        cfg.steady_chunk_frames = 12;
        cfg.first_context_frames = 0;
        cfg.steady_context_frames = 25;
        cfg.async_audio_decode = async_mode;

        auto streamer = std::make_shared<RecordingAudioStreamer>();
        AudioStreamerVariant variant = streamer;

        // 25 frames total -> chunk 0: 8 frames, chunk 1: 12 frames, final short tail: 5 frames (padded to 12, sliced to 5)
        Qwen3TTSStreamSession session(pool, nullptr, cfg, variant, std::chrono::steady_clock::now());
        for (size_t f = 0; f < 25; ++f) {
            std::vector<int64_t> codes(Q, static_cast<int64_t>(f + 1));
            ASSERT_TRUE(session.push_frame(codes.data(), Q));
        }
        auto res = session.finish();

        EXPECT_EQ(streamer->end_calls.load(), 1);
        ASSERT_EQ(streamer->chunk_sizes, (std::vector<size_t>{8 * UPSAMPLE, 12 * UPSAMPLE, 5 * UPSAMPLE}));
        ASSERT_EQ(res.waveform.size(), 25 * UPSAMPLE);
        EXPECT_EQ(streamer->all_samples, res.waveform);
        EXPECT_TRUE(res.has_ttfa);
        EXPECT_GE(res.ttfa_us, 0.0);
        EXPECT_GE(res.total_decode_us, 0.0);
        // Verify exact frame values survived context padding, right-edge padding, and trim.
        for (size_t f = 0; f < 25; ++f) {
            for (size_t s = 0; s < UPSAMPLE; ++s) {
                EXPECT_FLOAT_EQ(res.waveform[f * UPSAMPLE + s], static_cast<float>(f + 1));
            }
        }
    }
}

TEST(TestQwen3TTSStreaming, early_stop_cancel_and_worker_error_propagation) {
    constexpr size_t Q = 16;
    constexpr size_t UPSAMPLE = 4;
    ov::Core core;

    Qwen3TTSStreamDecoderPool pool;
    pool.set_metadata(Q, UPSAMPLE);
    pool.register_compiled_tier(
        0, 4, core.compile_model(make_synthetic_static_stream_decoder(Q, 0, 4, UPSAMPLE), "CPU").create_infer_request());

    // Early STOP keeps already-generated audio and calls end() exactly once.
    {
        SpeechGenerationConfig cfg;
        cfg.first_chunk_frames = 4;
        cfg.steady_chunk_frames = 4;
        cfg.first_context_frames = 0;
        cfg.steady_context_frames = 0;
        cfg.async_audio_decode = false;

        auto streamer = std::make_shared<RecordingAudioStreamer>();
        streamer->stop_after_chunk_count = 1;
        streamer->status_after_chunks = StreamingStatus::STOP;
        AudioStreamerVariant variant = streamer;

        Qwen3TTSStreamSession session(pool, nullptr, cfg, variant, std::chrono::steady_clock::now());
        size_t pushed = 0;
        for (size_t f = 0; f < 12; ++f) {
            std::vector<int64_t> codes(Q, static_cast<int64_t>(f + 1));
            ++pushed;
            if (!session.push_frame(codes.data(), Q)) {
                break;
            }
        }
        auto res = session.finish();
        EXPECT_EQ(pushed, 4u);
        EXPECT_TRUE(res.stopped_early);
        EXPECT_FALSE(res.cancelled);
        EXPECT_EQ(res.waveform.size(), 4 * UPSAMPLE);
        EXPECT_EQ(streamer->end_calls.load(), 1);
    }

    // Async worker exception propagates cleanly on finish() and still calls end().
    {
        SpeechGenerationConfig cfg;
        cfg.first_chunk_frames = 4;
        cfg.steady_chunk_frames = 4;
        cfg.first_context_frames = 0;
        cfg.steady_context_frames = 0;
        cfg.async_audio_decode = true;

        auto streamer = std::make_shared<RecordingAudioStreamer>();
        streamer->throw_on_write = true;
        AudioStreamerVariant variant = streamer;

        Qwen3TTSStreamSession session(pool, nullptr, cfg, variant, std::chrono::steady_clock::now());
        for (size_t f = 0; f < 4; ++f) {
            std::vector<int64_t> codes(Q, 1);
            try {
                session.push_frame(codes.data(), Q);
            } catch (const std::runtime_error&) {
                break;
            }
        }
        EXPECT_THROW(session.finish(), std::runtime_error);
        EXPECT_EQ(streamer->end_calls.load(), 1);
    }
}

TEST(TestQwen3TTSStreaming, perf_metrics_computes_ttfa_rtf_and_safe_zeros) {
    SpeechGenerationPerfMetrics metrics;
    metrics.output_sample_rate = 24000;
    metrics.num_generated_samples = 48000;  // 2.0 seconds of audio
    metrics.num_generated_frames = 25;
    metrics.raw_metrics.generate_durations.emplace_back(1000000.0f);  // 1000 ms -> RTF = 0.5
    metrics.raw_ttfa_durations.emplace_back(250000.0f);               // 250 ms
    metrics.raw_codegen_durations.emplace_back(700000.0f);            // 700 ms
    metrics.raw_audio_decode_durations.emplace_back(300000.0f);       // 300 ms
    metrics.evaluate_statistics();

    EXPECT_NEAR(metrics.generate_duration.mean, 1000.0f, 1e-2f);
    EXPECT_NEAR(metrics.ttfa.mean, 250.0f, 1e-2f);
    EXPECT_NEAR(metrics.codegen_duration.mean, 700.0f, 1e-2f);
    EXPECT_NEAR(metrics.audio_decode_duration.mean, 300.0f, 1e-2f);
    EXPECT_NEAR(metrics.rtf, 0.5f, 1e-3f);
    EXPECT_NEAR(metrics.throughput.mean, 48000.0f, 1.0f);

    SpeechGenerationPerfMetrics empty;
    empty.evaluate_statistics();
    EXPECT_EQ(empty.rtf, 0.0f);
    EXPECT_EQ(empty.throughput.mean, 0.0f);
}
