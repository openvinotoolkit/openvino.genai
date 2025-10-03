// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

auto get_config_for_cache() {
    ov::AnyMap config;
    config.insert({ov::cache_dir("whisper_cache")});
    return config;
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\" <DEVICE>");
    }

    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    std::string device = (argc == 4) ? argv[3] : "CPU";  // Default to CPU if no device is provided

    ov::AnyMap ov_config;
    if (device == "NPU" || device.find("GPU") != std::string::npos) {  // need to handle cases like "GPU", "GPU.0" and "GPU.1"
        // Cache compiled models on disk for GPU and NPU to save time on the
        // next run. It's not beneficial for CPU.
        ov_config = get_config_for_cache();
    }

    ov::genai::WhisperPipeline pipeline(models_path, device, ov_config);

    ov::genai::WhisperGenerationConfig config = pipeline.get_generation_config();
    // 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>";  // can switch to <|zh|> for Chinese language
    config.task = "transcribe";
    config.return_timestamps = true;

    // Pipeline expects normalized audio with Sample Rate of 16kHz
    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);
    auto result = pipeline.generate(raw_speech, config);

    std::cout << result << "\n";

    std::cout << std::fixed << std::setprecision(2);
    for (auto& chunk : *result.chunks) {
        std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text << "\n";
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
