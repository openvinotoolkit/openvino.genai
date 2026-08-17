// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Streaming ASR from a WAV file using ASRPipeline::create_streaming_session().
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

#include <chrono>
#include <iomanip>
#include <iostream>

#include "audio_utils.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <WAV_FILE> [DEVICE] [CHUNK_SEC] [STEP_MS] "
                                 "[--simulate-live] [--device DEVICE] [--chunk-sec SEC] [--step-ms MS]");
    }

    const std::filesystem::path models_path = argv[1];
    const std::string wav_file = argv[2];
    std::string device = "CPU";
    float chunk_sec = 2.0f;
    int step_ms = 500;

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

    ov::genai::ASRStreamingConfig streaming_config;
    streaming_config.chunk_size_sec = chunk_sec;
    streaming_config.warmup_chunks = 2;
    streaming_config.context_rollback_tokens = 5;

    std::cout << "Loading audio: " << wav_file << "\n";
    const ov::genai::RawSpeechInput wav = utils::audio::read_wav(wav_file);
    const size_t total_samples = wav.size();
    const float total_sec = static_cast<float>(total_samples) / 16000.0f;
    std::cout << std::fixed << std::setprecision(2)
              << "  Duration: " << total_sec << " s  (" << total_samples << " samples @ 16 kHz)\n\n";

    auto on_partial = [&](ov::genai::ASRPartialResult result) {
        std::cout << "[partial] (" << result.language << ") +" << result.committed_text
                  << " [" << result.partial_text << "]\n";
    };

    auto session = pipeline.create_streaming_session(streaming_config, gen_config, on_partial);

    const size_t step_samples = static_cast<size_t>(step_ms * 16000 / 1000);
    const auto wall_start = std::chrono::steady_clock::now();

    for (size_t pos = 0; pos < total_samples; pos += step_samples) {
        const size_t end = std::min(pos + step_samples, total_samples);
        const std::vector<float> segment(wav.begin() + pos, wav.begin() + end);
        session.push_chunk(segment);
    }

    auto result = session.finish();

    const auto wall_sec =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - wall_start).count();
    std::cout << "\n[final ] (" << result.languages[0] << ") " << result.texts[0] << "\n";
    std::cout << std::fixed << std::setprecision(2)
              << "\nTotal wall time: " << wall_sec << " s  (RTF " << wall_sec / total_sec << "x)\n";
    return 0;

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}
