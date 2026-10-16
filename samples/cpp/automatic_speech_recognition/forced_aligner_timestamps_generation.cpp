// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "../speech_generation/audio_utils.hpp"
#include "openvino/genai/automatic_speech_recognition/forced_aligner.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 4 || argc > 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <ASR_MODEL_DIR> <FORCED_ALIGNER_MODEL_DIR> \"<WAV_FILE_PATH>\" [DEVICE]");
    }

    std::filesystem::path asr_models_path = argv[1];
    std::filesystem::path forced_aligner_models_path = argv[2];
    std::string wav_file_path = argv[3];
    std::string device = (argc == 5) ? argv[4] : "CPU";

    auto aligner = std::make_shared<ov::genai::ASRForcedAligner>(forced_aligner_models_path, device);
    ov::genai::ASRPipeline pipeline(asr_models_path, device, ov::genai::forced_aligner(aligner));

    // ASR pipelines expect normalized 16 kHz audio.
    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

    // Leave language unset to let Qwen3-ASR detect it from the audio.
    auto result = pipeline.generate(raw_speech, ov::genai::word_timestamps(true));

    std::cout << "Detected language: " << result.languages[0] << "\n";
    std::cout << "Transcription: " << result.texts[0] << "\n\n";

    std::cout << "Word timestamps:\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& word : (*result.words)[0]) {
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
