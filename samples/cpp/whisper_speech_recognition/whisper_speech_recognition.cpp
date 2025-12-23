// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

auto get_config_for_cache() {
    ov::AnyMap config;
    config.insert({ov::cache_dir("whisper_cache")});
    return config;
}

void test_word_level_timestamps(const std::filesystem::path& models_path,
                                const std::string& samples_path,
                                const std::string& references_path) {
    // count files in samples_path
    size_t num_files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(samples_path)) {
        if (entry.path().extension() == ".wav") {
            num_files++;
        }
    }

    // load references json
    std::ifstream references_file(references_path);
    nlohmann::json references_json;
    references_file >> references_json;

    auto pipe = ov::genai::WhisperPipeline(models_path, "CPU", ov::genai::word_timestamps(true));

    constexpr float WORD_TS_ACCURACY = 0.01f;

    std::cout << "Testing word-level timestamps for " << num_files << " samples...\n";
    const auto start_time = std::chrono::high_resolution_clock::now();

    // iterate over files in samples_path
    for (size_t i = 0; i < num_files; ++i) {
        std::string wav_file_path = samples_path + "/sample_" + std::to_string(i) + ".wav";
        const ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

        auto result = pipe.generate(raw_speech, ov::genai::word_timestamps(true));

        const auto& reference = references_json[std::to_string(i)];
        const bool transcription_match = result.texts[0] == reference["transcription"];
        if (!transcription_match) {
            throw std::runtime_error("Transcription does not match reference for sample " + std::to_string(i));
        }
        for (size_t j = 0; j < result.words->size(); ++j) {
            const auto& word_info = (*result.words)[j];
            const auto& ref_word_info = reference["words"][j];
            const bool word_match = word_info.word == ref_word_info["word"];

            const bool start_ts_close =
                std::abs(word_info.start_ts - ref_word_info["start_ts"].get<double>()) < WORD_TS_ACCURACY;
            const bool end_ts_close =
                std::abs(word_info.end_ts - ref_word_info["end_ts"].get<double>()) < WORD_TS_ACCURACY;

            if (!word_match || !start_ts_close || !end_ts_close) {
                throw std::runtime_error("Word timing does not match reference for sample " + std::to_string(i) +
                                         ", word " + std::to_string(j));
            }
        }
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "All word-level timestamps tests passed in " << duration << " ms!\n";
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\" <DEVICE>");
    }

    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    std::string device = (argc == 4) ? argv[3] : "CPU";  // Default to CPU if no device is provided

    ov::AnyMap ov_config;
    if (device == "NPU" ||
        device.find("GPU") != std::string::npos) {  // need to handle cases like "GPU", "GPU.0" and "GPU.1"
        // Cache compiled models on disk for GPU and NPU to save time on the
        // next run. It's not beneficial for CPU.
        ov_config = get_config_for_cache();
    }

    ov_config.insert({ov::genai::word_timestamps.name(), true});

    ov::genai::WhisperPipeline pipeline(models_path, device, ov_config);

    ov::genai::WhisperGenerationConfig config = pipeline.get_generation_config();
    // 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>";  // can switch to <|zh|> for Chinese language
    config.task = "transcribe";
    config.return_timestamps = false;
    config.word_timestamps = true;

    // Pipeline expects normalized audio with Sample Rate of 16kHz
    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);
    auto result = pipeline.generate(raw_speech, config);

    std::cout << result << "\n";

    std::cout << std::fixed << std::setprecision(2);
    // for (auto& chunk : *result.chunks) {
    //     std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text <<
    //     "\n";
    // }

    test_word_level_timestamps(models_path,
                               "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/"
                               "whisper/librispeech_asr_dummy_wav_samples",
                               "/home/asuvorov/projects/openvino.genai/tests/python_tests/data/whisper/"
                               "librispeech_asr_dummy_word_timestamps_reference_tiny.json");

    if (result.words) {
        std::cout << "Word-level timestamps:\n";
        for (const auto& word_info : *result.words) {
            std::cout << "  " << word_info.word << "  " << word_info.start_ts << " - " << word_info.end_ts << "\n";
        }
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
