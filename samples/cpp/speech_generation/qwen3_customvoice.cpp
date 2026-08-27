// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Qwen3-TTS CustomVoice sample.
//
// CustomVoice models synthesize speech using one of the model's built-in
// speaker identities. You select a voice with `--speaker` and, optionally,
// steer the delivery (tone, emotion, pace) with a natural-language `--instruct`
// description. No reference audio or speaker embedding is required.

#include "audio_utils.hpp"
#include "qwen3_cli_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <string>

int main(int argc, char* argv[]) try {
    const auto args = qwen3_cli::normalized_argv(argc, argv);
    OPENVINO_ASSERT(args.size() >= 4,
                    "Usage: ",
                    args[0],
                    " <MODEL_DIR> \"<PROMPT>\" --speaker <NAME>"
                    " [--language <LANG>] [--instruct \"<STYLE>\"] [--device <DEVICE>]");

    const std::string models_path = args[1];
    const std::string prompt = args[2];
    std::string device = "CPU";
    std::string speaker;
    std::string language;
    std::string instruct;

    for (int arg_idx = 3; arg_idx < static_cast<int>(args.size());) {
        const std::string option = args[arg_idx++];
        OPENVINO_ASSERT(arg_idx < static_cast<int>(args.size()), "Missing value for option ", option);
        const std::string value = args[arg_idx++];

        if (option == "--speaker") {
            speaker = value;
        } else if (option == "--language") {
            language = value;
        } else if (option == "--instruct") {
            instruct = value;
        } else if (option == "--device") {
            device = value;
        } else {
            OPENVINO_THROW("Unknown option: ", option);
        }
    }

    OPENVINO_ASSERT(!speaker.empty(), "Qwen3-TTS CustomVoice requires --speaker <NAME>.");

    ov::genai::Text2SpeechPipeline pipe(models_path, device);

    ov::AnyMap properties;
    properties["speaker"] = speaker;
    if (!language.empty()) {
        properties["language"] = language;
    }
    if (!instruct.empty()) {
        properties["instruct"] = instruct;
    }

    // CustomVoice does not use an external speaker embedding.
    ov::genai::Text2SpeechDecodedResults gen_speech = pipe.generate(prompt, ov::Tensor(), properties);

    OPENVINO_ASSERT(gen_speech.speeches.size() == 1, "Expected exactly one decoded waveform");

    const std::string output_file_name = "output_audio.wav";
    const auto waveform_size = gen_speech.speeches[0].get_size();
    const auto waveform_ptr = gen_speech.speeches[0].data<const float>();
    const auto bits_per_sample = gen_speech.speeches[0].get_element_type().bitwidth();
    utils::audio::save_to_wav(waveform_ptr,
                              waveform_size,
                              output_file_name,
                              bits_per_sample,
                              gen_speech.output_sample_rate);
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
