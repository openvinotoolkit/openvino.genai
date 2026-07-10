// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <optional>
#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

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

ov::Tensor read_i64_npy_tensor(const std::filesystem::path& npy_path) {
    std::ifstream fstream{npy_path, std::ios::binary};
    OPENVINO_ASSERT(fstream.good(), "Failed to open ref_code file: ", npy_path.string());

    fstream.seekg(0, std::ios_base::end);
    OPENVINO_ASSERT(fstream.good(), "Failed to read ref_code file size: ", npy_path.string());
    const auto full_file_size = static_cast<std::size_t>(fstream.tellg());
    fstream.seekg(0, std::ios_base::beg);

    std::string magic(6, ' ');
    fstream.read(&magic[0], magic.size());
    OPENVINO_ASSERT(magic == "\x93NUMPY", "Invalid NPY header in ref_code file: ", npy_path.string());

    fstream.ignore(2);
    uint16_t header_size = 0;
    fstream.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    std::string header(header_size, ' ');
    fstream.read(&header[0], header.size());

    const std::string fortran_key = "'fortran_order':";
    const int fortran_idx = static_cast<int>(header.find(fortran_key));
    OPENVINO_ASSERT(fortran_idx != -1, "NPY missing fortran_order field: ", npy_path.string());
    int from = static_cast<int>(header.find_last_of(' ', fortran_idx + static_cast<int>(fortran_key.size()))) + 1;
    int to = static_cast<int>(header.find(',', static_cast<size_t>(from)));
    const std::string fortran_value = header.substr(static_cast<size_t>(from), static_cast<size_t>(to - from));
    OPENVINO_ASSERT(fortran_value == "False", "ref_code NPY must be C-contiguous (fortran_order=False)");

    const std::string shape_key = "'shape':";
    const int shape_idx = static_cast<int>(header.find(shape_key));
    OPENVINO_ASSERT(shape_idx != -1, "NPY missing shape field: ", npy_path.string());
    from = static_cast<int>(header.find('(', shape_idx + static_cast<int>(shape_key.size()))) + 1;
    to = static_cast<int>(header.find(')', static_cast<size_t>(from)));

    std::string shape_data = header.substr(static_cast<size_t>(from), static_cast<size_t>(to - from));
    ov::Shape shape;
    if (!shape_data.empty()) {
        shape_data.erase(std::remove(shape_data.begin(), shape_data.end(), ','), shape_data.end());
        std::istringstream shape_stream(shape_data);
        size_t dim = 0;
        while (shape_stream >> dim) {
            shape.push_back(dim);
        }
    }
    OPENVINO_ASSERT(!shape.empty(), "ref_code NPY shape must be non-empty");

    const std::string descr_key = "'descr':";
    const int descr_idx = static_cast<int>(header.find(descr_key));
    OPENVINO_ASSERT(descr_idx != -1, "NPY missing descr field: ", npy_path.string());
    from = static_cast<int>(header.find('\'', descr_idx + static_cast<int>(descr_key.size()))) + 1;
    to = static_cast<int>(header.find('\'', static_cast<size_t>(from)));
    const std::string dtype = header.substr(static_cast<size_t>(from), static_cast<size_t>(to - from));
    OPENVINO_ASSERT(dtype == "<i8", "ref_code NPY dtype must be int64 ('<i8'), got: ", dtype);

    const size_t payload_size = full_file_size - static_cast<size_t>(fstream.tellg());
    size_t elem_count = 1;
    for (size_t d : shape) {
        elem_count *= d;
    }
    OPENVINO_ASSERT(payload_size == elem_count * sizeof(int64_t),
                    "ref_code NPY payload size mismatch, expected ",
                    elem_count * sizeof(int64_t),
                    " bytes, got ",
                    payload_size,
                    " bytes");

    ov::Tensor tensor{ov::element::i64, shape};
    fstream.read(reinterpret_cast<char*>(tensor.data()), payload_size);
    OPENVINO_ASSERT(static_cast<size_t>(fstream.gcount()) == payload_size, "Failed to read full ref_code payload");
    return tensor;
}

#ifdef _WIN32
std::vector<std::string> windows_utf8_argv(int argc, char* argv[]) {
    struct LocalFreeDeleter {
        void operator()(LPWSTR* ptr) const noexcept {
            if (ptr != nullptr) {
                LocalFree(ptr);
            }
        }
    };

    std::vector<std::string> args;
    int wide_argc = 0;
    std::unique_ptr<LPWSTR, LocalFreeDeleter> wide_argv(CommandLineToArgvW(GetCommandLineW(), &wide_argc));
    if (wide_argv == nullptr || wide_argc <= 0) {
        args.reserve(static_cast<size_t>(argc));
        for (int i = 0; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }
        return args;
    }

    args.reserve(static_cast<size_t>(wide_argc));
    for (int i = 0; i < wide_argc; ++i) {
        const wchar_t* warg = wide_argv.get()[i];
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
        " <MODEL_DIR> \"<PROMPT>\" [<SPEAKER_EMBEDDING_BIN_FILE>] [--speaker_embedding_file_path <PATH>] [--voice_clone_ref_audio_wav_path <PATH.wav>] [--language <LANG>] [--speaker <NAME>] [--instruct <TEXT>] [--speed <FLOAT>] [--non_streaming_mode <true|false>] [--subtalker_dosample <true|false>] [--subtalker_top_k <INT>] [--subtalker_top_p <FLOAT>] [--subtalker_temperature <FLOAT>] [--do_sample <true|false>] [--top_k <INT>] [--top_p <FLOAT>] [--temperature <FLOAT>] [--repetition_penalty <FLOAT>] [--seed <INT>] [--max_new_tokens <INT>] [--voice_clone_ref_text <TEXT>] [--voice_clone_ref_codec_ids_file_path <PATH.npy>] [--device <DEVICE>]");

    const std::string models_path = args[1], prompt = args[2];
    std::string device = "CPU";

    std::optional<std::string> speaker_embedding_path;
    std::string language;
    std::string speaker;
    std::string instruct;
    float speed = 1.0f;
    bool non_streaming_mode = true;
    bool subtalker_dosample = true;
    int subtalker_top_k = 50;
    float subtalker_top_p = 1.0f;
    float subtalker_temperature = 0.9f;
    bool do_sample = true;
    int top_k = 50;
    float top_p = 1.0f;
    float temperature = 0.9f;
    float repetition_penalty = 1.05f;
    uint32_t seed = 0;
    int max_new_tokens = 4096;  // Match Python default instead of C++ default (2048)
    std::string voice_clone_ref_text;
    std::optional<std::string> qwen_ref_audio_wav_path;
    std::optional<std::string> qwen_ref_code_file_path;

    auto parse_bool = [](const std::string& value) {
        if (value == "1" || value == "true" || value == "TRUE" || value == "True") {
            return true;
        }
        if (value == "0" || value == "false" || value == "FALSE" || value == "False") {
            return false;
        }
        OPENVINO_THROW("Expected boolean value true/false/1/0, got: ", value);
    };

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
        } else if (option == "--speaker_embedding_file_path") {
            OPENVINO_ASSERT(!speaker_embedding_path.has_value(),
                            "Speaker embedding path is already provided. "
                            "Use either positional SPEAKER_EMBEDDING_BIN_FILE or --speaker_embedding_file_path.");
            speaker_embedding_path = value;
        } else if (option == "--speaker") {
            speaker = value;
        } else if (option == "--instruct") {
            instruct = value;
        } else if (option == "--speed") {
            speed = std::stof(value);
        } else if (option == "--non_streaming_mode") {
            non_streaming_mode = parse_bool(value);
        } else if (option == "--subtalker_dosample") {
            subtalker_dosample = parse_bool(value);
        } else if (option == "--subtalker_top_k") {
            subtalker_top_k = std::stoi(value);
        } else if (option == "--subtalker_top_p") {
            subtalker_top_p = std::stof(value);
        } else if (option == "--subtalker_temperature") {
            subtalker_temperature = std::stof(value);
        } else if (option == "--do_sample") {
            do_sample = parse_bool(value);
        } else if (option == "--top_k") {
            top_k = std::stoi(value);
        } else if (option == "--top_p") {
            top_p = std::stof(value);
        } else if (option == "--temperature") {
            temperature = std::stof(value);
        } else if (option == "--repetition_penalty") {
            repetition_penalty = std::stof(value);
        } else if (option == "--seed") {
            seed = static_cast<uint32_t>(std::stoul(value));
        } else if (option == "--max_new_tokens") {
            max_new_tokens = std::stoi(value);
        } else if (option == "--voice_clone_ref_text") {
            voice_clone_ref_text = value;
        } else if (option == "--voice_clone_ref_audio_wav_path") {
            qwen_ref_audio_wav_path = value;
        } else if (option == "--voice_clone_ref_codec_ids_file_path") {
            qwen_ref_code_file_path = value;
        } else if (option == "--device") {
            device = value;
        } else {
            OPENVINO_THROW("Unknown option: ", option);
        }
    }

    ov::AnyMap ov_properties;
    if( device == "NPU") {
        ov_properties["CACHE_DIR"] = "ov_cache";
    }

    ov::genai::Text2SpeechPipeline pipe(models_path, device, ov_properties);
    const ov::Shape expected_speaker_shape = pipe.get_speaker_embedding_shape();

    // Qwen3 Base expects x-vector style embedding with shape {1, 1, D}.
    const bool expects_qwen3_base_speaker_embedding =
        expected_speaker_shape.size() == 3 && expected_speaker_shape[0] == 1 && expected_speaker_shape[1] == 1;
    if (expects_qwen3_base_speaker_embedding && !speaker_embedding_path.has_value() && !qwen_ref_audio_wav_path.has_value()) {
        OPENVINO_THROW("This model expects a speaker embedding tensor with shape ",
                       shape_to_string(expected_speaker_shape),
                       ". Provide SPEAKER_EMBEDDING_BIN_FILE, --speaker_embedding_file_path <PATH>, or --voice_clone_ref_audio_wav_path <PATH.wav>.");
    }

    ov::AnyMap properties;
    if (!language.empty()) {
        properties["language"] = language;
    }
    if (!speaker.empty()) {
        properties["speaker"] = speaker;
    }
    if (!instruct.empty()) {
        properties["instruct"] = instruct;
    }
    if (speed != 1.0f) {
        properties["speed"] = speed;
    }
    properties["non_streaming_mode"] = non_streaming_mode;
    properties["subtalker_dosample"] = subtalker_dosample;
    properties["subtalker_top_k"] = subtalker_top_k;
    properties["subtalker_top_p"] = subtalker_top_p;
    properties["subtalker_temperature"] = subtalker_temperature;
    properties["do_sample"] = do_sample;
    properties["top_k"] = top_k;
    properties["top_p"] = top_p;
    properties["temperature"] = temperature;
    properties["repetition_penalty"] = repetition_penalty;
    if (seed != 0) {
        properties["rng_seed"] = static_cast<size_t>(seed);
    }
    properties["max_new_tokens"] = max_new_tokens;

    if (qwen_ref_code_file_path.has_value() && voice_clone_ref_text.empty()) {
        OPENVINO_THROW("--voice_clone_ref_text is required when --voice_clone_ref_codec_ids_file_path is provided.");
    }

    if (!voice_clone_ref_text.empty()) {
        properties["voice_clone_ref_text"] = voice_clone_ref_text;
    }
    if (qwen_ref_audio_wav_path.has_value()) {
        properties["voice_clone_ref_audio"] = utils::audio::read_wav_mono_f32(*qwen_ref_audio_wav_path, 24000);
    }
    if (qwen_ref_code_file_path.has_value()) {
        properties["voice_clone_ref_codec_ids"] = read_i64_npy_tensor(*qwen_ref_code_file_path);
    }

    std::cout << "[QWEN_DEBUG] sample args: device=" << device
              << " language='" << language
              << "' speaker='" << speaker
              << "' instruct_len=" << instruct.size()
              << " non_streaming_mode=" << (non_streaming_mode ? "true" : "false")
              << " subtalker_dosample=" << (subtalker_dosample ? "true" : "false")
              << " subtalker_top_k=" << subtalker_top_k
              << " subtalker_top_p=" << subtalker_top_p
              << " subtalker_temperature=" << subtalker_temperature
              << " do_sample=" << (do_sample ? "true" : "false")
              << " top_k=" << top_k
              << " top_p=" << top_p
              << " temperature=" << temperature
              << " repetition_penalty=" << repetition_penalty
              << " seed=" << seed
              << " max_new_tokens=" << max_new_tokens
              << " voice_clone_ref_text_len=" << voice_clone_ref_text.size()
              << " voice_clone_ref_audio_wav_path='" << (qwen_ref_audio_wav_path.has_value() ? *qwen_ref_audio_wav_path : "") << "'"
              << " voice_clone_ref_codec_ids_file_path='" << (qwen_ref_code_file_path.has_value() ? *qwen_ref_code_file_path : "") << "'"
              << " expected_speaker_shape=" << shape_to_string(expected_speaker_shape)
              << std::endl;

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
    const uint32_t output_sample_rate = gen_speech.output_sample_rate;
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
