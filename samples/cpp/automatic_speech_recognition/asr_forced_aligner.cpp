// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "audio_utils.hpp"
#include "openvino/genai/automatic_speech_recognition/forced_aligner.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 4 || argc > 6) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> \"<WAV_FILE_PATH>\" \"<TRANSCRIPT>\" [DEVICE] [LANGUAGE]");
    }

    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    std::string transcript = argv[3];
    std::string device = (argc >= 5) ? argv[4] : "CPU";
    std::string language = (argc >= 6) ? argv[5] : "english";

    ov::genai::ASRForcedAligner aligner(models_path, device);

    // Forced aligner expects normalized audio with a sample rate of 16kHz
    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);
    auto words = aligner.align(raw_speech, transcript, ov::genai::language(language));

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
