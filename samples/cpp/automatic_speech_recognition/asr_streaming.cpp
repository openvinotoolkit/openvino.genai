// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Streaming ASR from a WAV file using ASRStreamingSession.
// Pushes the file in fixed-size segments without injecting artificial delays.
//
// Usage:
//   asr_streaming <MODEL_DIR> <WAV_FILE> [DEVICE] [CHUNK_SEC] [STEP_MS]
//
//   MODEL_DIR   — path to an OpenVINO Qwen3-ASR model directory
//   WAV_FILE    — mono 16 kHz PCM WAV file to transcribe
//   DEVICE      — OpenVINO device string (default: CPU)
//   CHUNK_SEC   — decode interval in seconds (default: 2.0)
//   STEP_MS     — chunk push interval in ms (default: 500)
//   --window-chunks N          — sliding window size in chunks, 0 = unbounded (default: 6)
//   --window-rollback-chunks N — unfixed rollback chunks within the window (default: 2)
//   --unbounded-prefix         — experimental, Qwen3-ASR only: keep audio bounded (per
//                                --window-chunks) but stop evicting the text prefix.
//   --repetition-penalty F     — generation repetition penalty, 1.0 = disabled (default: 1.0)
//   --no-repeat-ngram-size N   — forbid repeating any N-gram, 0 = disabled (default: 0)

#include <chrono>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <windows.h>
#endif

#include "audio_utils.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

int main(int argc, char* argv[]) try {
#ifdef _WIN32
    // The model emits UTF-8 text; without this the Windows console renders non-ASCII bytes
    // (e.g. em-dashes) as mojibake under its default OEM codepage.
    SetConsoleOutputCP(CP_UTF8);
#endif

    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <WAV_FILE> [DEVICE] [CHUNK_SEC] [STEP_MS] "
                                 "[--device DEVICE] [--chunk-sec SEC] [--step-ms MS] "
                                 "[--window-chunks N] [--window-rollback-chunks N] "
                                 "[--unbounded-prefix] "
                                 "[--repetition-penalty F] [--no-repeat-ngram-size N]");
    }

    const std::filesystem::path models_path = argv[1];
    const std::string wav_file = argv[2];
    std::string device = "CPU";
    float chunk_sec = 2.0f;
    int step_ms = 500;
    size_t window_chunks = 6;
    size_t window_rollback_chunks = 2;
    bool unbounded_prefix = false;
    float repetition_penalty = 1.0f;
    size_t no_repeat_ngram_size = 0;

    int positional_index = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--device") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --device");
            }
            device = argv[++i];
            continue;
        }

        if (arg == "--chunk-sec") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --chunk-sec");
            }
            chunk_sec = std::stof(argv[++i]);
            continue;
        }

        if (arg == "--step-ms") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --step-ms");
            }
            step_ms = std::stoi(argv[++i]);
            continue;
        }

        if (arg == "--window-chunks") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --window-chunks");
            }
            window_chunks = static_cast<size_t>(std::stoul(argv[++i]));
            continue;
        }

        if (arg == "--window-rollback-chunks") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --window-rollback-chunks");
            }
            window_rollback_chunks = static_cast<size_t>(std::stoul(argv[++i]));
            continue;
        }

        if (arg == "--unbounded-prefix") {
            unbounded_prefix = true;
            continue;
        }

        if (arg == "--repetition-penalty") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --repetition-penalty");
            }
            repetition_penalty = std::stof(argv[++i]);
            continue;
        }

        if (arg == "--no-repeat-ngram-size") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --no-repeat-ngram-size");
            }
            no_repeat_ngram_size = static_cast<size_t>(std::stoul(argv[++i]));
            continue;
        }

        if (positional_index >= 3) {
            throw std::runtime_error("Unexpected argument: " + arg);
        }

        switch (positional_index++) {
            case 0:
                device = arg;
                break;
            case 1:
                chunk_sec = std::stof(arg);
                break;
            case 2:
                step_ms = std::stoi(arg);
                break;
        }
    }

    std::cout << "Loading model from: " << models_path << " on " << device << "\n";
    ov::genai::ASRPipeline pipeline(models_path, device);

    ov::genai::ASRGenerationConfig gen_config = pipeline.get_generation_config();
    gen_config.max_new_tokens = 32;  // keep chunk decodes fast
    gen_config.repetition_penalty = repetition_penalty;
    if (no_repeat_ngram_size > 0) {
        gen_config.no_repeat_ngram_size = no_repeat_ngram_size;
    }

    ov::genai::ASRStreamingConfig streaming_config;
    streaming_config.chunk_size_sec = chunk_sec;
    streaming_config.warmup_chunks = 2;
    streaming_config.context_rollback_tokens = 5;
    streaming_config.window_chunk_num = window_chunks;
    streaming_config.window_rollback_chunk_num = window_rollback_chunks;
    streaming_config.unbounded_prefix = unbounded_prefix;

    std::cout << "Loading audio: " << wav_file << "\n";
    const ov::genai::RawSpeechInput wav = utils::audio::read_wav(wav_file);
    const size_t total_samples = wav.size();
    const float total_sec = static_cast<float>(total_samples) / 16000.0f;
    std::cout << std::fixed << std::setprecision(2)
              << "  Duration: " << total_sec << " s  (" << total_samples << " samples @ 16 kHz)\n\n";

    auto session = ov::genai::ASRStreamingSession(pipeline, streaming_config, gen_config);

    const size_t step_samples = static_cast<size_t>(step_ms * 16000 / 1000);
    const auto wall_start = std::chrono::steady_clock::now();

    int chunk_num = 0;
    for (size_t pos = 0; pos < total_samples; pos += step_samples) {
        const size_t end = std::min(pos + step_samples, total_samples);
        const std::vector<float> segment(wav.begin() + pos, wav.begin() + end);
        if (const auto result = session.push_chunk(segment)) {
            std::cout << chunk_num << " [partial] (" << result->language << ") +" << result->new_committed_text
                      << " [" << result->partial_text << "]\n";

            chunk_num++;
        }
    }

    const auto final_result = session.finish();
    if (!final_result.new_committed_text.empty()) {
        std::cout << chunk_num << "[partial] (" << final_result.language << ") +" << final_result.new_committed_text
                  << " []\n";
    }

    const auto wall_sec =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - wall_start).count();
    std::cout << "\n[final ] (" << final_result.language << ") " << final_result.committed_text << "\n";
    std::cout << std::fixed << std::setprecision(2)
              << "\nTotal wall time: " << wall_sec << " s  (RTF " << wall_sec / total_sec << "x)\n";
    return 0;

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}
