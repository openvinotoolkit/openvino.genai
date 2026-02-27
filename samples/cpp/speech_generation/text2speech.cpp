// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <optional>

int main(int argc, char* argv[]) try {
    OPENVINO_ASSERT(
        argc >= 3,
        "Usage: ",
        argv[0],
        " <MODEL_DIR> \"<PROMPT>\" [<SPEAKER_EMBEDDING_BIN_FILE>] [--speech_model_type <speecht5_tts|kokoro>] [--voice <VOICE_ID>] [--language <en-us|en-gb>] [--speed <FLOAT>] [--sample_rate <INT>]");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";

    std::optional<std::string> speaker_embedding_path;
    std::string speech_model_type;
    std::string voice;
    std::string language;
    float speed = 1.0f;
    uint32_t sample_rate = 0;

    int arg_idx = 3;
    if (arg_idx < argc && std::string(argv[arg_idx]).rfind("--", 0) != 0) {
        speaker_embedding_path = argv[arg_idx++];
    }

    while (arg_idx < argc) {
        const std::string option = argv[arg_idx++];
        OPENVINO_ASSERT(arg_idx < argc, "Missing value for option ", option);
        const std::string value = argv[arg_idx++];

        if (option == "--speech_model_type") {
            speech_model_type = value;
        } else if (option == "--voice") {
            voice = value;
        } else if (option == "--language") {
            language = value;
        } else if (option == "--speed") {
            speed = std::stof(value);
        } else if (option == "--sample_rate") {
            sample_rate = static_cast<uint32_t>(std::stoul(value));
        } else {
            OPENVINO_THROW("Unknown option: ", option);
        }
    }

    ov::genai::Text2SpeechPipeline pipe(models_path, device);
    ov::AnyMap properties;
    if (!speech_model_type.empty()) {
        properties["speech_model_type"] = speech_model_type;
    }
    if (!voice.empty()) {
        properties["voice"] = voice;
    }
    if (!language.empty()) {
        properties["language"] = language;
    }
    if (speed != 1.0f) {
        properties["speed"] = speed;
    }

    ov::genai::Text2SpeechDecodedResults gen_speech;
    if (speaker_embedding_path.has_value()) {
        auto speaker_embedding = utils::audio::read_speaker_embedding(*speaker_embedding_path);
        gen_speech = pipe.generate(prompt, speaker_embedding, properties);
    } else {
        gen_speech = pipe.generate(prompt, ov::Tensor(), properties);
    }

    OPENVINO_ASSERT(gen_speech.speeches.size() == 1, "Expected exactly one decoded waveform");

    std::string output_file_name = "output_audio.wav";
    auto waveform_size = gen_speech.speeches[0].get_size();
    auto waveform_ptr = gen_speech.speeches[0].data<const float>();
    auto bits_per_sample = gen_speech.speeches[0].get_element_type().bitwidth();
    const uint32_t output_sample_rate = sample_rate > 0 ? sample_rate : (speech_model_type == "kokoro" ? 24000 : 16000);
    utils::audio::save_to_wav(waveform_ptr, waveform_size, output_file_name, bits_per_sample, output_sample_rate);
    std::cout << "[Info] Text successfully converted to audio file \"" << output_file_name << "\"." << std::endl;

    auto& perf_metrics = gen_speech.perf_metrics;
    if (perf_metrics.m_evaluated) {
        std::cout << "\n\n=== Performance Summary ===" << std::endl;
        std::cout << "Throughput              : " << perf_metrics.throughput.mean << " samples/sec." << std::endl;
        std::cout << "Total Generation Time   : " << perf_metrics.generate_duration.mean / 1000.0f << " sec."
                  << std::endl;
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
