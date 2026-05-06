// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <vector>

#include "load_image.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

namespace {

bool streamer(const std::string& text) {
    std::cout << text << std::flush;
    return false;  // Don't stop generation
}

void print_usage(const char* prog_name) {
    std::cout << "Qwen3-Omni multimodal chat\n\n"
              << "Usage: " << prog_name
              << " <model_dir> [--image <path>] [--enable-speech] [--speaker <name>] [--stream-audio] "
                 "[--audio-chunk-frames <N>]\n\n"
              << "  model_dir              Path to OpenVINO model directory\n"
              << "  --image <path>         Input image file path\n"
              << "  --enable-speech        Enable speech output generation\n"
              << "  --speaker <name>       Speaker name for speech (e.g., f245, m02)\n"
              << "  --stream-audio         Enable audio streaming (receive chunks during generation)\n"
              << "  --audio-chunk-frames N Number of codec frames per chunk (default 5, ~400ms each)\n"
              << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_dir = argv[1];
    std::string image_path;
    bool enable_speech = false;
    bool stream_audio = false;
    std::string speaker;
    size_t audio_chunk_frames = 5;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--enable-speech") {
            enable_speech = true;
        } else if (arg == "--stream-audio") {
            stream_audio = true;
        } else if (arg == "--audio-chunk-frames" && i + 1 < argc) {
            audio_chunk_frames = std::stoull(argv[++i]);
        } else if (arg == "--speaker" && i + 1 < argc) {
            speaker = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "Loading model from: " << model_dir << std::endl;
    ov::genai::VLMPipeline pipe(model_dir, "CPU");

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 256;
    if (enable_speech) {
        config.return_audio = true;
        config.speaker = speaker;
        if (stream_audio) {
            config.audio_chunk_frames = audio_chunk_frames;
        }
    }

    size_t chunk_count = 0;
    size_t total_samples = 0;
    auto audio_callback = [&](ov::Tensor audio_chunk) -> ov::genai::StreamingStatus {
        chunk_count++;
        total_samples += audio_chunk.get_size();
        auto duration_ms = static_cast<float>(total_samples) / 24.0f;
        std::cout << "\r  [audio: " << chunk_count << " chunks, " << duration_ms << "ms]" << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    std::vector<ov::Tensor> images;
    if (!image_path.empty()) {
        std::cout << "Loading image: " << image_path << std::endl;
        auto [rgb, _] = utils::load_image(image_path);
        images.push_back(std::move(rgb));
    }

    pipe.start_chat();
    std::cout << "\nQwen3-Omni Chat (type 'quit' to exit)\n"
              << "----------------------------------------\n";

    std::string prompt;
    bool first_turn = true;
    while (true) {
        std::cout << "\nYou: ";
        if (!std::getline(std::cin, prompt))
            break;
        if (prompt == "quit" || prompt == "exit" || prompt == "q")
            break;
        if (prompt.empty())
            continue;

        std::cout << "Assistant: ";
        chunk_count = 0;
        total_samples = 0;

        ov::genai::VLMDecodedResults result;
        auto current_images = (first_turn && !images.empty()) ? images : std::vector<ov::Tensor>{};

        ov::AnyMap generate_params;
        generate_params["images"] = current_images;
        generate_params["generation_config"] = config;
        generate_params["streamer"] = std::function<bool(std::string)>(streamer);
        if (enable_speech && stream_audio) {
            generate_params["audio_streamer"] = std::function<ov::genai::StreamingStatus(ov::Tensor)>(audio_callback);
        }
        result = pipe.generate(prompt, generate_params);
        std::cout << std::endl;

        if (enable_speech && stream_audio && chunk_count > 0) {
            std::cout << "\n[Streamed " << chunk_count << " audio chunks, " << total_samples << " samples]"
                      << std::endl;
        } else if (enable_speech && !result.speech_outputs.empty()) {
            std::cout << "[Speech output generated: " << result.speech_outputs[0].get_size() << " samples]"
                      << std::endl;
        }

        first_turn = false;
    }

    pipe.finish_chat();
    std::cout << "\nChat ended." << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
