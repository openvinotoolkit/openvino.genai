// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Qwen3-TTS VoiceDesign sample.
//
// VoiceDesign models create a brand-new voice from a natural-language
// description passed via `--instruct` (for example, "A calm middle-aged male
// voice with a slight British accent"). Unlike CustomVoice, there is no speaker
// list; the `--instruct` prompt is the primary control. VoiceDesign does not
// accept a `--speaker` name or an external speaker embedding.

#include "audio_utils.hpp"
#include "qwen3_cli_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <string>

int main(int argc, char* argv[]) try {
    const auto args = qwen3_cli::normalized_argv(argc, argv);
    OPENVINO_ASSERT(args.size() >= 4,
                    "Usage: ",
                    args[0],
                    " <MODEL_DIR> \"<PROMPT>\" --instruct \"<VOICE_DESCRIPTION>\""
                    " [--language <LANG>] [--device <DEVICE>]");

    const std::string models_path = args[1];
    const std::string prompt = args[2];
    std::string device = "CPU";
    std::string instruct;
    std::string language;

    for (int arg_idx = 3; arg_idx < static_cast<int>(args.size());) {
        const std::string option = args[arg_idx++];
        OPENVINO_ASSERT(arg_idx < static_cast<int>(args.size()), "Missing value for option ", option);
        const std::string value = args[arg_idx++];

        if (option == "--instruct") {
            instruct = value;
        } else if (option == "--language") {
            language = value;
        } else if (option == "--device") {
            device = value;
        } else {
            OPENVINO_THROW("Unknown option: ", option);
        }
    }

    OPENVINO_ASSERT(!instruct.empty(),
                    "Qwen3-TTS VoiceDesign requires --instruct \"<VOICE_DESCRIPTION>\".");

    ov::genai::Text2SpeechPipeline pipe(models_path, device);

    ov::AnyMap properties;
    properties["instruct"] = instruct;
    if (!language.empty()) {
        properties["language"] = language;
    }

    // VoiceDesign does not use an external speaker embedding.
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
