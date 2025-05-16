// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

int main(int argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3 || argc == 4,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> \"<PROMPT>\" [<SPEAKER_EMBEDDING_BIN_FILE>]");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";

    ov::genai::Text2SpeechPipeline pipe(models_path, device);

    ov::genai::Text2SpeechDecodedResults gen_speech;
    if (argc == 4) {
        const std::string speaker_embedding_path = argv[3];
        auto speaker_embedding = utils::audio::read_speaker_embedding(speaker_embedding_path);
        gen_speech = pipe.generate(prompt, speaker_embedding);
    } else {
        gen_speech = pipe.generate(prompt);
    }

    OPENVINO_ASSERT(gen_speech.speeches.size() == 1, "Expected exactly one decoded waveform");

    std::string output_file_name = "output_audio.wav";
    auto waveform_size = gen_speech.speeches[0].get_size();
    auto waveform_ptr = gen_speech.speeches[0].data<const float>();
    auto bits_per_sample = gen_speech.speeches[0].get_element_type().bitwidth();
    utils::audio::save_to_wav(waveform_ptr, waveform_size, output_file_name, bits_per_sample);
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
