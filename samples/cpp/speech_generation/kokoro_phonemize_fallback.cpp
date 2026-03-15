// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <optional>
#include <string>

int main(int argc, char* argv[]) try {
    OPENVINO_ASSERT(
        argc >= 3,
        "Usage: ",
        argv[0],
        " <MODEL_DIR> \"<PROMPT>\" [--voice <VOICE_ID>] [--language <en-us|en-gb>] "
        "[--phonemize_fallback_model_dir <DIR>] [--output <FILE.wav>] [--device <DEVICE>]");

    const std::string models_path = argv[1];
    const std::string prompt = argv[2];

    std::string device = "CPU";
    std::string voice = "af_heart";
    std::string language = "en-us";
    std::optional<std::string> phonemize_fallback_model_dir;
    std::string output_file_name = "output_audio.wav";

    int idx = 3;
    while (idx < argc) {
        const std::string option = argv[idx++];
        OPENVINO_ASSERT(idx < argc, "Missing value for option ", option);
        const std::string value = argv[idx++];

        if (option == "--voice") {
            voice = value;
        } else if (option == "--language") {
            language = value;
        } else if (option == "--phonemize_fallback_model_dir") {
            phonemize_fallback_model_dir = value;
        } else if (option == "--output") {
            output_file_name = value;
        } else if (option == "--device") {
            device = value;
        } else {
            OPENVINO_THROW("Unknown option: ", option);
        }
    }

    OPENVINO_ASSERT(language == "en-us" || language == "en-gb",
                    "kokoro_phonemize_fallback sample supports only --language en-us or en-gb");

    ov::genai::Text2SpeechPipeline pipe(models_path, device);
    auto config = pipe.get_generation_config();
    config.voice = voice;
    config.language = language;
    config.phonemize_fallback_model_dir = phonemize_fallback_model_dir;
    pipe.set_generation_config(config);

    auto result = pipe.generate(prompt, ov::Tensor());
    OPENVINO_ASSERT(result.speeches.size() == 1, "Expected exactly one decoded waveform");

    const auto waveform_size = result.speeches[0].get_size();
    const auto* waveform_ptr = result.speeches[0].data<const float>();
    const auto bits_per_sample = result.speeches[0].get_element_type().bitwidth();
    utils::audio::save_to_wav(waveform_ptr,
                              waveform_size,
                              output_file_name,
                              bits_per_sample,
                              result.output_sample_rate);

    std::cout << "[Info] Saved: " << output_file_name << std::endl;
    if (phonemize_fallback_model_dir.has_value()) {
        std::cout << "[Info] Phonemize fallback: OpenVINO model at '" << *phonemize_fallback_model_dir << "'"
                  << std::endl;
    } else {
        std::cout << "[Info] Phonemize fallback: espeak-ng" << std::endl;
    }

    return EXIT_SUCCESS;
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
