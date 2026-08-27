// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Qwen3-TTS Base (voice clone) sample.
//
// Base models clone a voice from a short reference recording. There are two
// cloning modes, selected automatically from the inputs you provide:
//
//   1. x-vector mode (fast, identity only)
//        Provide reference audio (or a pre-saved speaker embedding). Only the
//        speaker embedding is used. `ref_text` is NOT required.
//
//   2. ICL mode (in-context learning, higher fidelity)
//        Provide reference audio AND its transcript (`--ref_text`). The pipeline
//        additionally conditions on the reference speech codes, which improves
//        similarity at the cost of extra compute.
//
// Reusing a reference prompt
// --------------------------
// Extracting the speaker embedding and reference codes from audio is the
// expensive part of cloning. To avoid recomputing them for every generation,
// clone once from reference audio and save the artifacts returned on the result
// (`speaker_embedding` and `voice_clone_ref_codec_ids`) with
// `--save_speaker_embedding_file_path` / `--save_ref_codec_ids_file_path`. Later
// runs can pass them back via `--speaker_embedding_file_path` and
// `--ref_codec_ids_file_path` to skip the reference-audio encoder entirely.

#include "audio_utils.hpp"
#include "qwen3_cli_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace {

std::string shape_to_string(const ov::Shape& shape) {
    std::ostringstream os;
    os << "{";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "}";
    return os.str();
}

// Reference codes (int64) are persisted with a tiny self-describing binary format
// so this sample can save and reload them without any external dependency:
//
//   [int64 rank][int64 dim_0] ... [int64 dim_{rank-1}][int64 payload ...]
//
// This is intentionally simple; it is only meant for round-tripping artifacts
// between runs of these samples.
ov::Tensor read_reference_codes(const std::filesystem::path& path) {
    std::ifstream in{path, std::ios::binary};
    OPENVINO_ASSERT(in.good(), "Failed to open reference codes file: ", path.string());

    int64_t rank = 0;
    in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    OPENVINO_ASSERT(in.good() && rank > 0 && rank <= 8, "Invalid reference codes file: ", path.string());

    ov::Shape shape;
    size_t elem_count = 1;
    for (int64_t i = 0; i < rank; ++i) {
        int64_t dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        OPENVINO_ASSERT(in.good() && dim > 0, "Invalid reference codes shape in: ", path.string());
        shape.push_back(static_cast<size_t>(dim));
        elem_count *= static_cast<size_t>(dim);
    }

    ov::Tensor tensor{ov::element::i64, shape};
    in.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(elem_count * sizeof(int64_t)));
    OPENVINO_ASSERT(static_cast<size_t>(in.gcount()) == elem_count * sizeof(int64_t),
                    "Failed to read full reference codes payload from: ",
                    path.string());
    return tensor;
}

void write_reference_codes(const std::filesystem::path& path, const ov::Tensor& tensor) {
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::i64, "Reference codes tensor must be int64");
    const ov::Shape shape = tensor.get_shape();

    std::ofstream out{path, std::ios::binary};
    OPENVINO_ASSERT(out.good(), "Failed to open output file: ", path.string());

    const int64_t rank = static_cast<int64_t>(shape.size());
    out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    for (size_t dim : shape) {
        const int64_t dim64 = static_cast<int64_t>(dim);
        out.write(reinterpret_cast<const char*>(&dim64), sizeof(dim64));
    }
    out.write(reinterpret_cast<const char*>(tensor.data<const int64_t>()),
              static_cast<std::streamsize>(tensor.get_size() * sizeof(int64_t)));
    OPENVINO_ASSERT(out.good(), "Failed to write reference codes to: ", path.string());
}

// Saves a float32 tensor as a flat binary file, matching the layout expected by
// utils::audio::read_speaker_embedding.
void write_f32_bin(const std::filesystem::path& bin_path, const ov::Tensor& tensor) {
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32,
                    "Speaker embedding tensor must be f32 to save as .bin");
    std::ofstream out{bin_path, std::ios::binary};
    OPENVINO_ASSERT(out.good(), "Failed to open output file: ", bin_path.string());
    out.write(reinterpret_cast<const char*>(tensor.data<const float>()),
              static_cast<std::streamsize>(tensor.get_size() * sizeof(float)));
    OPENVINO_ASSERT(out.good(), "Failed to write speaker embedding to: ", bin_path.string());
}

}  // namespace

int main(int argc, char* argv[]) try {
    const auto args = qwen3_cli::normalized_argv(argc, argv);
    OPENVINO_ASSERT(args.size() >= 4,
                    "Usage: ",
                    args[0],
                    " <MODEL_DIR> \"<PROMPT>\""
                    " (--ref_audio_wav_path <PATH.wav> | --speaker_embedding_file_path <PATH.bin>)"
                    " [--ref_text \"<TRANSCRIPT>\"] [--ref_codec_ids_file_path <PATH.bin>]"
                    " [--save_speaker_embedding_file_path <PATH.bin>] [--save_ref_codec_ids_file_path <PATH.bin>]"
                    " [--language <LANG>] [--device <DEVICE>]");

    const std::string models_path = args[1];
    const std::string prompt = args[2];
    std::string device = "CPU";
    std::string language;
    std::string ref_text;
    std::optional<std::string> ref_audio_wav_path;
    std::optional<std::string> speaker_embedding_path;
    std::optional<std::string> ref_codec_ids_path;
    std::optional<std::string> save_speaker_embedding_path;
    std::optional<std::string> save_ref_codec_ids_path;

    for (int arg_idx = 3; arg_idx < static_cast<int>(args.size());) {
        const std::string option = args[arg_idx++];
        OPENVINO_ASSERT(arg_idx < static_cast<int>(args.size()), "Missing value for option ", option);
        const std::string value = args[arg_idx++];

        if (option == "--ref_audio_wav_path") {
            ref_audio_wav_path = value;
        } else if (option == "--speaker_embedding_file_path") {
            speaker_embedding_path = value;
        } else if (option == "--ref_text") {
            ref_text = value;
        } else if (option == "--ref_codec_ids_file_path") {
            ref_codec_ids_path = value;
        } else if (option == "--save_speaker_embedding_file_path") {
            save_speaker_embedding_path = value;
        } else if (option == "--save_ref_codec_ids_file_path") {
            save_ref_codec_ids_path = value;
        } else if (option == "--language") {
            language = value;
        } else if (option == "--device") {
            device = value;
        } else {
            OPENVINO_THROW("Unknown option: ", option);
        }
    }

    // A Base clone always needs a voice source: either reference audio (from
    // which the pipeline extracts the identity) or a pre-saved speaker embedding.
    OPENVINO_ASSERT(ref_audio_wav_path.has_value() || speaker_embedding_path.has_value(),
                    "Qwen3-TTS Base requires --ref_audio_wav_path <PATH.wav> or "
                    "--speaker_embedding_file_path <PATH.bin>.");

    // Pre-saved reference codes are only meaningful in ICL mode, which requires
    // the matching reference transcript.
    if (ref_codec_ids_path.has_value()) {
        OPENVINO_ASSERT(!ref_text.empty(),
                        "--ref_text is required when --ref_codec_ids_file_path is provided (ICL mode).");
    }

    ov::genai::Text2SpeechPipeline pipe(models_path, device);
    const ov::Shape expected_speaker_shape = pipe.get_speaker_embedding_shape();

    ov::AnyMap properties;
    if (!language.empty()) {
        properties["language"] = language;
    }

    // Reference audio: the pipeline internally derives the speaker embedding and,
    // in ICL mode, the reference codes from this waveform.
    if (ref_audio_wav_path.has_value()) {
        properties["voice_clone_ref_audio"] = utils::audio::read_wav_mono_f32(*ref_audio_wav_path, 24000);
    }

    // ICL mode is enabled by providing the reference transcript. Reference codes
    // are either supplied directly (reuse) or extracted from the reference audio.
    if (!ref_text.empty()) {
        properties["voice_clone_ref_text"] = ref_text;
    }
    if (ref_codec_ids_path.has_value()) {
        properties["voice_clone_ref_codec_ids"] = read_reference_codes(*ref_codec_ids_path);
    }

    const bool icl_mode = !ref_text.empty();
    std::cout << "[Info] Qwen3-TTS Base voice clone (" << (icl_mode ? "ICL" : "x-vector") << " mode)."
              << " Expected speaker embedding shape: " << shape_to_string(expected_speaker_shape) << std::endl;

    // A pre-saved speaker embedding is passed as the pipeline's speaker_embedding
    // argument; otherwise the pipeline extracts it from the reference audio.
    ov::genai::Text2SpeechDecodedResults gen_speech;
    if (speaker_embedding_path.has_value()) {
        auto speaker_embedding = utils::audio::read_speaker_embedding(*speaker_embedding_path, expected_speaker_shape);
        gen_speech = pipe.generate(prompt, speaker_embedding, properties);
    } else {
        gen_speech = pipe.generate(prompt, ov::Tensor(), properties);
    }

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

    // Persist the resolved clone artifacts for reuse. Passing them back on a later run
    // (via --speaker_embedding_file_path / --ref_codec_ids_file_path) skips the
    // reference-audio encoding step.
    if (save_speaker_embedding_path.has_value()) {
        OPENVINO_ASSERT(static_cast<bool>(gen_speech.speaker_embedding),
                        "No speaker embedding was produced to save. Provide --ref_audio_wav_path.");
        write_f32_bin(*save_speaker_embedding_path, gen_speech.speaker_embedding);
        std::cout << "[Info] Saved speaker embedding to \"" << *save_speaker_embedding_path << "\"." << std::endl;
    }
    if (save_ref_codec_ids_path.has_value()) {
        OPENVINO_ASSERT(static_cast<bool>(gen_speech.voice_clone_ref_codec_ids),
                        "No reference codes were produced to save. ICL mode (--ref_text) with "
                        "--ref_audio_wav_path is required.");
        write_reference_codes(*save_ref_codec_ids_path, gen_speech.voice_clone_ref_codec_ids);
        std::cout << "[Info] Saved reference codes to \"" << *save_ref_codec_ids_path << "\"." << std::endl;
    }

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
