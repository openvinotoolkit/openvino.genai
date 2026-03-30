// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

#ifdef _WIN32
std::vector<std::string> windows_utf8_argv(int argc, char* argv[]) {
    std::vector<std::string> args;
    int wide_argc = 0;
    LPWSTR* wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (wide_argv == nullptr || wide_argc <= 0) {
        args.reserve(static_cast<size_t>(argc));
        for (int i = 0; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }
        return args;
    }

    args.reserve(static_cast<size_t>(wide_argc));
    for (int i = 0; i < wide_argc; ++i) {
        const wchar_t* warg = wide_argv[i];
        const int needed = WideCharToMultiByte(CP_UTF8, 0, warg, -1, nullptr, 0, nullptr, nullptr);
        OPENVINO_ASSERT(needed > 0, "Failed to convert command-line argument to UTF-8");
        std::string utf8(static_cast<size_t>(needed), '\0');
        const int written = WideCharToMultiByte(CP_UTF8,
                                                0,
                                                warg,
                                                -1,
                                                utf8.data(),
                                                needed,
                                                nullptr,
                                                nullptr);
        OPENVINO_ASSERT(written == needed, "Failed to convert command-line argument to UTF-8");
        utf8.pop_back();
        args.push_back(std::move(utf8));
    }

    LocalFree(wide_argv);
    return args;
}
#endif

std::vector<std::string> normalized_argv(int argc, char* argv[]) {
#ifdef _WIN32
    return windows_utf8_argv(argc, argv);
#else
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return args;
#endif
}

} // namespace

int main(int argc, char* argv[]) try {
    const auto args = normalized_argv(argc, argv);
    OPENVINO_ASSERT(
        args.size() >= 3,
        "Usage: ",
        args[0],
        " <MODEL_DIR> \"<PROMPT>\" [<SPEAKER_EMBEDDING_BIN_FILE>] [--language <en-us|en-gb|es|fr-fr|hi|it|pt-br>] [--speed <FLOAT>] [--sample_rate <INT>]");

    const std::string models_path = args[1], prompt = args[2];
    const std::string device = "CPU";

    std::optional<std::string> speaker_embedding_path;
    std::string language;
    float speed = 1.0f;
    uint32_t sample_rate = 0;

    int arg_idx = 3;
    if (arg_idx < static_cast<int>(args.size()) && args[arg_idx].rfind("--", 0) != 0) {
        speaker_embedding_path = args[arg_idx++];
    }

    while (arg_idx < static_cast<int>(args.size())) {
        const std::string option = args[arg_idx++];
        OPENVINO_ASSERT(arg_idx < static_cast<int>(args.size()), "Missing value for option ", option);
        const std::string value = args[arg_idx++];

        if (option == "--language") {
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
    if (!language.empty()) {
        properties["language"] = language;
    }
    if (speed != 1.0f) {
        properties["speed"] = speed;
    }

    ov::genai::Text2SpeechDecodedResults gen_speech;
    if (speaker_embedding_path.has_value()) {
        auto speaker_embedding = utils::audio::read_speaker_embedding(*speaker_embedding_path,
                                                                      pipe.get_speaker_embedding_shape());
        gen_speech = pipe.generate(prompt, speaker_embedding, properties);
    } else {
        gen_speech = pipe.generate(prompt, ov::Tensor(), properties);
    }

    OPENVINO_ASSERT(gen_speech.speeches.size() == 1, "Expected exactly one decoded waveform");

    std::string output_file_name = "output_audio.wav";
    auto waveform_size = gen_speech.speeches[0].get_size();
    auto waveform_ptr = gen_speech.speeches[0].data<const float>();
    auto bits_per_sample = gen_speech.speeches[0].get_element_type().bitwidth();
    const uint32_t output_sample_rate = sample_rate > 0 ? sample_rate : gen_speech.output_sample_rate;
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
