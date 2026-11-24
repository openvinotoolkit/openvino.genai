// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"
#include "qwen3_cli_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string to_upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

// Options shared by all Qwen3-TTS variants.
struct CommonOptions {
    std::optional<std::string> attention_backend;  // PA (continuous batching) or SDPA
    bool stream = false;                           // --stream: enable audio output streaming
    std::optional<size_t> stream_chunk_frames;
    std::optional<size_t> stream_first_chunk_frames;
    std::optional<size_t> stream_left_context_frames;

    // Options that take no value.
    static bool is_flag(const std::string& option) {
        return option == "--stream";
    }

    // Consumes `option` if it is a common option. Returns false when it is not one.
    // `value` is empty for flags.
    bool try_parse(const std::string& option, const std::string& value) {
        if (option == "--stream") {
            stream = true;
        } else if (option == "--attention_backend") {
            attention_backend = to_upper(value);
        } else if (option == "--stream_chunk_frames") {
            stream_chunk_frames = static_cast<size_t>(std::stoull(value));
        } else if (option == "--stream_first_chunk_frames") {
            stream_first_chunk_frames = static_cast<size_t>(std::stoull(value));
        } else if (option == "--stream_left_context_frames") {
            stream_left_context_frames = static_cast<size_t>(std::stoull(value));
        } else {
            return false;
        }
        return true;
    }

    // True when --stream was given. The chunk options only tune streaming, so passing one without --stream is
    // rejected instead of being silently ignored.
    bool streaming_requested() const {
        OPENVINO_ASSERT(stream || (!stream_chunk_frames.has_value() && !stream_first_chunk_frames.has_value() &&
                                   !stream_left_context_frames.has_value()),
                        "--stream_chunk_frames, --stream_first_chunk_frames and --stream_left_context_frames "
                        "require --stream.");
        return stream;
    }
};

ov::AnyMap get_pipeline_config_for_device(const std::string& device,
                                          const std::optional<std::string>& attention_backend) {
    const std::string upper_device = to_upper(device);
    ov::AnyMap config;
    if (attention_backend.has_value()) {
        config["ATTENTION_BACKEND"] = *attention_backend;
    }

    if (upper_device.find("NPU") != std::string::npos || upper_device.find("GPU") != std::string::npos) {
        config["CACHE_DIR"] = std::string("qwen3_tts_cache_dir");
    }

    return config;
}

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

void write_f32_bin(const std::filesystem::path& bin_path, const ov::Tensor& tensor) {
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32,
                    "Speaker embedding tensor must be f32 to save as .bin");
    std::ofstream out{bin_path, std::ios::binary};
    OPENVINO_ASSERT(out.good(), "Failed to open output file: ", bin_path.string());
    out.write(reinterpret_cast<const char*>(tensor.data<const float>()),
              static_cast<std::streamsize>(tensor.get_size() * sizeof(float)));
    OPENVINO_ASSERT(out.good(), "Failed to write speaker embedding to: ", bin_path.string());
}

// Timing of audio chunks delivered to the speech streamer, measured from mark_start() (call it right
// before generate()).
struct StreamStats {
    static constexpr double kSampleRate = 24000.0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double first_chunk_ms = -1.0;
    size_t first_chunk_samples = 0;
    size_t chunks = 0;
    size_t samples = 0;

    void mark_start() {
        start = std::chrono::steady_clock::now();
    }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    void on_chunk(size_t chunk_samples) {
        if (chunks++ == 0) {
            first_chunk_ms = elapsed_ms();
            first_chunk_samples = chunk_samples;
        }
        samples += chunk_samples;
    }
    void print_summary() const {
        const double total_ms = elapsed_ms();
        std::cout << "[Stream] Time to first audio: " << first_chunk_ms << " ms (first chunk holds "
                  << first_chunk_samples / kSampleRate * 1000.0 << " ms of audio)" << std::endl;
        std::cout << "[Stream] " << chunks << " chunks, " << samples / kSampleRate << " s of audio streamed, "
                  << "generate() took " << total_ms << " ms" << std::endl;
    }
};

// Enables audio output streaming: audio chunks are delivered to a callback while generation is running.
// The callback only reports arrival times here; the full waveform is still returned by generate().
// Returns the stats object: call mark_start() right before generate() and print_summary() after it.
std::shared_ptr<StreamStats> enable_streaming(ov::AnyMap& properties, const CommonOptions& options) {
    if (options.stream_chunk_frames.has_value()) {
        properties["audio_chunk_frames"] = *options.stream_chunk_frames;
    }
    if (options.stream_first_chunk_frames.has_value()) {
        properties["audio_first_chunk_frames"] = *options.stream_first_chunk_frames;
    }
    if (options.stream_left_context_frames.has_value()) {
        properties["audio_chunk_left_context_frames"] = *options.stream_left_context_frames;
    }
    auto stats = std::make_shared<StreamStats>();
    std::function<ov::genai::StreamingStatus(const ov::Tensor&)> callback = [stats](const ov::Tensor& chunk) {
        std::cout << "[Stream] chunk " << stats->chunks << ": " << chunk.get_size() << " samples at "
                  << stats->elapsed_ms() << " ms" << std::endl;
        stats->on_chunk(chunk.get_size());
        return ov::genai::StreamingStatus::RUNNING;
    };
    properties["speech_streamer"] = callback;
    return stats;
}

void write_audio_and_perf(const ov::genai::Text2SpeechDecodedResults& gen_speech, const std::string& output_file_name) {
    OPENVINO_ASSERT(gen_speech.speeches.size() == 1, "Expected exactly one decoded waveform");

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
}

int run_base(const std::vector<std::string>& args) {
    OPENVINO_ASSERT(args.size() >= 4,
                    "Usage: qwen3_tts base <MODEL_DIR> \"<PROMPT>\""
                    " (--ref_audio_wav_path <PATH.wav> | --speaker_embedding_file_path <PATH.bin>)"
                    " [--ref_text \"<TRANSCRIPT>\"] [--ref_codec_ids_file_path <PATH.bin>]"
                    " [--save_speaker_embedding_file_path <PATH.bin>] [--save_ref_codec_ids_file_path <PATH.bin>]"
                    " [--language <LANG>] [--device <DEVICE>] [--max_new_tokens <N>] [--seed <N>] [--output_wav_path <PATH.wav>]"
                    " [--attention_backend <PA|SDPA>] [--stream]"
                    " [--stream_chunk_frames <N>] [--stream_first_chunk_frames <N>]"
                    " [--stream_left_context_frames <N>]");

    const std::string models_path = args[2];
    const std::string prompt = args[3];
    std::string device = "CPU";
    std::string output_wav_path = "output_audio.wav";
    std::string language;
    std::string ref_text;
    std::optional<int64_t> max_new_tokens;
    std::optional<std::string> ref_audio_wav_path;
    std::optional<std::string> speaker_embedding_path;
    std::optional<std::string> ref_codec_ids_path;
    std::optional<std::string> save_speaker_embedding_path;
    std::optional<std::string> save_ref_codec_ids_path;
    std::optional<size_t> seed;
    CommonOptions common;

    for (int arg_idx = 4; arg_idx < static_cast<int>(args.size());) {
        const std::string option = args[arg_idx++];
        if (CommonOptions::is_flag(option)) {
            common.try_parse(option, "");
            continue;
        }
        OPENVINO_ASSERT(arg_idx < static_cast<int>(args.size()), "Missing value for option ", option);
        const std::string value = args[arg_idx++];

        if (common.try_parse(option, value)) {
            // handled by the shared options
        } else if (option == "--ref_audio_wav_path") {
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
        } else if (option == "--max_new_tokens") {
            max_new_tokens = std::stoll(value);
            OPENVINO_ASSERT(*max_new_tokens > 0, "--max_new_tokens must be > 0");
        } else if (option == "--seed") {
            seed = static_cast<size_t>(std::stoull(value));
        } else if (option == "--output_wav_path") {
            output_wav_path = value;
        } else {
            OPENVINO_THROW("Unknown option: ", option);
        }
    }

    OPENVINO_ASSERT(ref_audio_wav_path.has_value() || speaker_embedding_path.has_value(),
                    "Qwen3-TTS Base requires --ref_audio_wav_path <PATH.wav> or "
                    "--speaker_embedding_file_path <PATH.bin>.");

    if (ref_codec_ids_path.has_value()) {
        OPENVINO_ASSERT(!ref_text.empty(),
                        "--ref_text is required when --ref_codec_ids_file_path is provided (ICL mode).");
    }

    ov::genai::Text2SpeechPipeline pipe(models_path, device, get_pipeline_config_for_device(device, common.attention_backend));
    const ov::Shape expected_speaker_shape = pipe.get_speaker_embedding_shape();

    ov::AnyMap properties;
    if (!language.empty()) {
        properties["language"] = language;
    }
    if (ref_audio_wav_path.has_value()) {
        properties["ref_audio"] = utils::audio::read_wav_as_tensor(*ref_audio_wav_path, 24000);
    }
    if (!ref_text.empty()) {
        properties["ref_text"] = ref_text;
    }
    if (ref_codec_ids_path.has_value()) {
        properties["ref_codec_ids"] = read_reference_codes(*ref_codec_ids_path);
    }
    if (max_new_tokens.has_value()) {
        properties["max_new_tokens"] = *max_new_tokens;
    }
    if (seed.has_value()) {
        properties["rng_seed"] = *seed;
    }
    std::shared_ptr<StreamStats> stream_stats;
    if (common.streaming_requested()) {
        stream_stats = enable_streaming(properties, common);
    }

    const bool icl_mode = !ref_text.empty();
    std::cout << "[Info] Qwen3-TTS Base voice clone (" << (icl_mode ? "ICL" : "x-vector") << " mode)." << std::endl;

    ov::genai::Text2SpeechDecodedResults gen_speech;
    if (speaker_embedding_path.has_value()) {
        auto speaker_embedding = utils::audio::read_speaker_embedding(*speaker_embedding_path, expected_speaker_shape);
        if (stream_stats) {
            stream_stats->mark_start();
        }
        gen_speech = pipe.generate(prompt, speaker_embedding, properties);
    } else {
        if (stream_stats) {
            stream_stats->mark_start();
        }
        gen_speech = pipe.generate(prompt, ov::Tensor(), properties);
    }
    if (stream_stats) {
        stream_stats->print_summary();
    }

    write_audio_and_perf(gen_speech, output_wav_path);

    if (save_speaker_embedding_path.has_value()) {
        OPENVINO_ASSERT(static_cast<bool>(gen_speech.speaker_embedding),
                        "No speaker embedding was produced to save. Provide --ref_audio_wav_path.");
        write_f32_bin(*save_speaker_embedding_path, gen_speech.speaker_embedding);
        std::cout << "[Info] Saved speaker embedding to \"" << *save_speaker_embedding_path << "\"." << std::endl;
    }
    if (save_ref_codec_ids_path.has_value()) {
        OPENVINO_ASSERT(static_cast<bool>(gen_speech.ref_codec_ids),
                        "No reference codes were produced to save. ICL mode (--ref_text) with "
                        "--ref_audio_wav_path is required.");
        write_reference_codes(*save_ref_codec_ids_path, gen_speech.ref_codec_ids);
        std::cout << "[Info] Saved reference codes to \"" << *save_ref_codec_ids_path << "\"." << std::endl;
    }

    return EXIT_SUCCESS;
}

int run_customvoice(const std::vector<std::string>& args) {
    OPENVINO_ASSERT(args.size() >= 4,
                    "Usage: qwen3_tts customvoice <MODEL_DIR> \"<PROMPT>\" --speaker <NAME>"
                    " [--language <LANG>] [--instruct \"<STYLE>\"] [--device <DEVICE>]"
                    " [--max_new_tokens <N>] [--seed <N>] [--output_wav_path <PATH.wav>]"
                    " [--attention_backend <PA|SDPA>] [--stream]"
                    " [--stream_chunk_frames <N>] [--stream_first_chunk_frames <N>]"
                    " [--stream_left_context_frames <N>]");

    const std::string models_path = args[2];
    const std::string prompt = args[3];
    std::string device = "CPU";
    std::string output_wav_path = "output_audio.wav";
    std::string speaker;
    std::string language;
    std::string instruct;
    std::optional<int64_t> max_new_tokens;
    std::optional<size_t> seed;
    CommonOptions common;

    for (int arg_idx = 4; arg_idx < static_cast<int>(args.size());) {
        const std::string option = args[arg_idx++];
        if (CommonOptions::is_flag(option)) {
            common.try_parse(option, "");
            continue;
        }
        OPENVINO_ASSERT(arg_idx < static_cast<int>(args.size()), "Missing value for option ", option);
        const std::string value = args[arg_idx++];

        if (common.try_parse(option, value)) {
            // handled by the shared options
        } else if (option == "--speaker") {
            speaker = value;
        } else if (option == "--language") {
            language = value;
        } else if (option == "--instruct") {
            instruct = value;
        } else if (option == "--device") {
            device = value;
        } else if (option == "--max_new_tokens") {
            max_new_tokens = std::stoll(value);
            OPENVINO_ASSERT(*max_new_tokens > 0, "--max_new_tokens must be > 0");
        } else if (option == "--seed") {
            seed = static_cast<size_t>(std::stoull(value));
        } else if (option == "--output_wav_path") {
            output_wav_path = value;
        } else {
            OPENVINO_THROW("Unknown option: ", option);
        }
    }

    OPENVINO_ASSERT(!speaker.empty(), "Qwen3-TTS CustomVoice requires --speaker <NAME>.");

    ov::genai::Text2SpeechPipeline pipe(models_path, device, get_pipeline_config_for_device(device, common.attention_backend));

    ov::AnyMap properties;
    properties["speaker"] = speaker;
    if (!language.empty()) {
        properties["language"] = language;
    }
    if (!instruct.empty()) {
        properties["instruct"] = instruct;
    }
    if (max_new_tokens.has_value()) {
        properties["max_new_tokens"] = *max_new_tokens;
    }
    if (seed.has_value()) {
        properties["rng_seed"] = *seed;
    }
    std::shared_ptr<StreamStats> stream_stats;
    if (common.streaming_requested()) {
        stream_stats = enable_streaming(properties, common);
    }

    if (stream_stats) {
        stream_stats->mark_start();
    }
    ov::genai::Text2SpeechDecodedResults gen_speech = pipe.generate(prompt, ov::Tensor(), properties);
    if (stream_stats) {
        stream_stats->print_summary();
    }
    write_audio_and_perf(gen_speech, output_wav_path);
    return EXIT_SUCCESS;
}

int run_voice_design(const std::vector<std::string>& args) {
    OPENVINO_ASSERT(args.size() >= 4,
                    "Usage: qwen3_tts voice-design <MODEL_DIR> \"<PROMPT>\" --instruct \"<VOICE_DESCRIPTION>\""
                    " [--language <LANG>] [--device <DEVICE>] [--max_new_tokens <N>] [--seed <N>] [--output_wav_path <PATH.wav>]"
                    " [--attention_backend <PA|SDPA>] [--stream]"
                    " [--stream_chunk_frames <N>] [--stream_first_chunk_frames <N>]"
                    " [--stream_left_context_frames <N>]");

    const std::string models_path = args[2];
    const std::string prompt = args[3];
    std::string device = "CPU";
    std::string output_wav_path = "output_audio.wav";
    std::string instruct;
    std::string language;
    std::optional<int64_t> max_new_tokens;
    std::optional<size_t> seed;
    CommonOptions common;

    for (int arg_idx = 4; arg_idx < static_cast<int>(args.size());) {
        const std::string option = args[arg_idx++];
        if (CommonOptions::is_flag(option)) {
            common.try_parse(option, "");
            continue;
        }
        OPENVINO_ASSERT(arg_idx < static_cast<int>(args.size()), "Missing value for option ", option);
        const std::string value = args[arg_idx++];

        if (common.try_parse(option, value)) {
            // handled by the shared options
        } else if (option == "--instruct") {
            instruct = value;
        } else if (option == "--language") {
            language = value;
        } else if (option == "--device") {
            device = value;
        } else if (option == "--max_new_tokens") {
            max_new_tokens = std::stoll(value);
            OPENVINO_ASSERT(*max_new_tokens > 0, "--max_new_tokens must be > 0");
        } else if (option == "--seed") {
            seed = static_cast<size_t>(std::stoull(value));
        } else if (option == "--output_wav_path") {
            output_wav_path = value;
        } else {
            OPENVINO_THROW("Unknown option: ", option);
        }
    }

    OPENVINO_ASSERT(!instruct.empty(),
                    "Qwen3-TTS VoiceDesign requires --instruct \"<VOICE_DESCRIPTION>\".");

    ov::genai::Text2SpeechPipeline pipe(models_path, device, get_pipeline_config_for_device(device, common.attention_backend));

    ov::AnyMap properties;
    properties["instruct"] = instruct;
    if (!language.empty()) {
        properties["language"] = language;
    }
    if (max_new_tokens.has_value()) {
        properties["max_new_tokens"] = *max_new_tokens;
    }
    if (seed.has_value()) {
        properties["rng_seed"] = *seed;
    }
    std::shared_ptr<StreamStats> stream_stats;
    if (common.streaming_requested()) {
        stream_stats = enable_streaming(properties, common);
    }

    if (stream_stats) {
        stream_stats->mark_start();
    }
    ov::genai::Text2SpeechDecodedResults gen_speech = pipe.generate(prompt, ov::Tensor(), properties);
    if (stream_stats) {
        stream_stats->print_summary();
    }
    write_audio_and_perf(gen_speech, output_wav_path);
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char* argv[]) try {
    const auto args = qwen3_cli::normalized_argv(argc, argv);
    OPENVINO_ASSERT(args.size() >= 2,
                    "Usage: ",
                    args[0],
                    " <base|customvoice|voice-design> <MODEL_DIR> \"<PROMPT>\" [OPTIONS]");

    const std::string variant = args[1];
    if (variant == "base") {
        return run_base(args);
    }
    if (variant == "customvoice") {
        return run_customvoice(args);
    }
    if (variant == "voice-design") {
        return run_voice_design(args);
    }

    OPENVINO_THROW("Unknown Qwen3-TTS variant '", variant, "'. Expected one of: base, customvoice, voice-design.");
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
