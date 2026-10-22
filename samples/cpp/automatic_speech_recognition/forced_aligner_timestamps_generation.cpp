// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../speech_generation/audio_utils.hpp"
#include "openvino/genai/automatic_speech_recognition/forced_aligner.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <ASR_MODEL_DIR> <FORCED_ALIGNER_MODEL_DIR> \"<WAV_FILE_PATH>\""
                                 " [DEVICE] [--language LANGUAGE]");
    }

    std::filesystem::path asr_models_path = argv[1];
    std::filesystem::path forced_aligner_models_path = argv[2];
    std::string wav_file_path = argv[3];
    std::string device = "CPU";
    std::string language;
    bool device_specified = false;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--language") {
            if (++i >= argc) {
                throw std::runtime_error("Missing value for --language");
            }
            language = argv[i];
        } else if (!device_specified && arg.rfind("-", 0) != 0) {
            device = arg;
            device_specified = true;
        } else {
            throw std::runtime_error("Unexpected argument: " + arg);
        }
    }

    ov::genai::ASRPipeline pipeline(asr_models_path, device);
    ov::genai::ASRForcedAligner aligner(forced_aligner_models_path, device);

    // ASR pipelines expect normalized 16 kHz audio.
    auto raw_speech = utils::audio::read_wav(wav_file_path);

    // Transcribe audio using the ASR model.
    auto result = pipeline.generate(raw_speech);
    const auto& transcript = result.texts[0];

    std::cout << "Transcription: " << transcript << "\n";

    // Use the detected language when available, otherwise use the specified language.
    if (language.empty() && !result.languages.empty()) {
        language = result.languages[0];
    }

    if (language.empty()) {
        throw std::runtime_error("Specify --language when the ASR model does not report the language");
    }

    // Align the transcription with the audio to get word timestamps.
    auto words = aligner.align(raw_speech, transcript, ov::genai::language(language));

    std::cout << "\nWord timestamps:\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& word : words) {
        std::cout << "[" << word.start_ts << ", " << word.end_ts << "]: " << word.text << "\n";
    }
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}