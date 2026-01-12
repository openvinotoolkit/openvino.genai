// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

auto get_config_for_cache() {
    ov::AnyMap config;
    config.insert({ov::cache_dir("whisper_cache")});
    return config;
}

std::vector<ov::Tensor> read_vector_of_tensors_from_np(const std::filesystem::path& filename) {
    std::vector<ov::Tensor> tensors;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        OPENVINO_THROW("Failed to open file: " + filename.string());
    }

    // Read and validate NumPy header magic bytes
    uint8_t magic[6];
    file.read(reinterpret_cast<char*>(magic), 6);
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' || magic[3] != 'M' || magic[4] != 'P' ||
        magic[5] != 'Y') {
        std::cerr << "Invalid NumPy file format" << std::endl;
        OPENVINO_THROW("Failed to open file: " + filename.string());
    }

    // Read version
    uint8_t major_version, minor_version;
    file.read(reinterpret_cast<char*>(&major_version), 1);
    file.read(reinterpret_cast<char*>(&minor_version), 1);

    // Read header length (little-endian uint16)
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    // Read header string
    std::string header_str(header_len, '\0');
    file.read(&header_str[0], header_len);

    // Parse shape from header
    // Expected format: {'descr': '<f4', 'fortran_order': False, 'shape': (head_size, dim1, dim2, ...), }
    size_t shape_start = header_str.find("'shape': (");
    if (shape_start == std::string::npos) {
        OPENVINO_THROW("Failed to parse shape from header");
    }
    shape_start += 10;  // length of "'shape': ("
    size_t shape_end = header_str.find(')', shape_start);
    std::string shape_str = header_str.substr(shape_start, shape_end - shape_start);

    // Parse dimensions
    std::vector<size_t> dims;
    std::istringstream shape_stream(shape_str);
    std::string dim_str;
    while (std::getline(shape_stream, dim_str, ',')) {
        // Trim whitespace
        dim_str.erase(0, dim_str.find_first_not_of(" \t"));
        dim_str.erase(dim_str.find_last_not_of(" \t") + 1);
        if (!dim_str.empty()) {
            dims.push_back(std::stoull(dim_str));
        }
    }

    if (dims.size() < 2) {
        OPENVINO_THROW("Invalid shape dimensions");
    }

    // First dimension is head_size, remaining are tensor shape
    size_t head_size = dims[0];
    ov::Shape tensor_shape(dims.begin() + 1, dims.end());

    // Calculate total size per tensor
    size_t total_size = 1;
    for (const auto& dim : tensor_shape) {
        total_size *= dim;
    }

    // Read tensors
    tensors.clear();
    for (size_t i = 0; i < head_size; ++i) {
        ov::Tensor tensor{ov::element::f32, tensor_shape};
        float* data = tensor.data<float>();
        file.read(reinterpret_cast<char*>(data), total_size * sizeof(float));

        if (!file) {
            OPENVINO_THROW("Failed to read tensor data from file: " + filename.string());
        }

        tensors.push_back(tensor);
    }

    file.close();
    std::cout << "Loaded " << head_size << " tensors with shape " << tensor_shape.to_string() << " from " << filename
              << std::endl;
    return tensors;
}

bool tensors_close(const std::vector<ov::Tensor>& tensors_a,
                   const std::vector<ov::Tensor>& tensors_b,
                   const float atol = 1e-6f) {
    if (tensors_a.size() != tensors_b.size()) {
        std::cerr << "Tensor vectors have different sizes: " << tensors_a.size() << " vs " << tensors_b.size()
                  << std::endl;
        return false;
    }

    for (size_t i = 0; i < tensors_a.size(); ++i) {
        const ov::Tensor& a = tensors_a[i];
        const ov::Tensor& b = tensors_b[i];

        if (a.get_shape() != b.get_shape()) {
            std::cerr << "Tensors at index " << i << " have different shapes: " << a.get_shape().to_string() << " vs "
                      << b.get_shape().to_string() << std::endl;
            return false;
        }

        const size_t total_size = a.get_size();
        const float* data_a = a.data<float>();
        const float* data_b = b.data<float>();

        for (size_t j = 0; j < total_size; ++j) {
            if (std::abs(data_a[j] - data_b[j]) > atol) {
                std::cerr << "Tensors differ at index " << i << ", element " << j << ": " << data_a[j] << " vs "
                          << data_b[j] << std::endl;
                return false;
            }
        }
    }

    return true;
}

void test_word_level_timestamps(const std::filesystem::path& models_path,
                                const std::string& samples_path,
                                const std::string& references_path) {
    // count files in samples_path
    size_t num_files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(samples_path)) {
        if (entry.path().extension() == ".wav") {
            num_files++;
        }
    }

    // load references json
    std::ifstream references_file(references_path);
    nlohmann::json references_json;
    references_file >> references_json;

    auto pipe = ov::genai::WhisperPipeline(models_path, "CPU", ov::genai::word_timestamps(true));

    constexpr float WORD_TS_ACCURACY = 0.01f;

    std::cout << "Testing word-level timestamps for " << num_files << " samples...\n";
    const auto start_time = std::chrono::high_resolution_clock::now();

    // iterate over files in samples_path
    for (size_t i = 0; i < num_files; ++i) {
        std::string wav_file_path = samples_path + "/sample_" + std::to_string(i) + ".wav";
        const ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

        auto result = pipe.generate(raw_speech, ov::genai::word_timestamps(true));

        const auto& reference = references_json[std::to_string(i)];
        const bool transcription_match = result.texts[0] == reference["transcription"];
        if (!transcription_match) {
            throw std::runtime_error("Transcription does not match reference for sample " + std::to_string(i));
        }
        for (size_t j = 0; j < result.words->size(); ++j) {
            const auto& word_info = (*result.words)[j];
            const auto& ref_word_info = reference["words"][j];
            const bool word_match = word_info.word == ref_word_info["word"];

            const bool start_ts_close =
                std::abs(word_info.start_ts - ref_word_info["start_ts"].get<double>()) < WORD_TS_ACCURACY;
            const bool end_ts_close =
                std::abs(word_info.end_ts - ref_word_info["end_ts"].get<double>()) < WORD_TS_ACCURACY;

            if (!word_match || !start_ts_close || !end_ts_close) {
                throw std::runtime_error("Word timing does not match reference for sample " + std::to_string(i) +
                                         ", word " + std::to_string(j));
            }
        }
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "All word-level timestamps tests passed in " << duration << " ms!\n";
}

void test_encoder_attention_qks(const std::filesystem::path& models_path, const std::string& samples_path) {
    auto pipe = ov::genai::WhisperPipeline(models_path, "CPU", ov::genai::word_timestamps(true));

    constexpr float WORD_TS_ACCURACY = 0.01f;

    std::cout << "Testing encoder_attention_qks...\n";

    std::string wav_file_path = samples_path + "/sample_0.wav";
    const ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

    const auto start_time = std::chrono::high_resolution_clock::now();
    auto config = pipe.get_generation_config();
    config.word_timestamps = true;
    config.save_attention_weights = true;
    auto result = pipe.generate(raw_speech, config);
    const auto end_time = std::chrono::high_resolution_clock::now();

    const std::filesystem::path reference_path = "/home/asuvorov/projects/openvino.genai/.vscode/tasks/"
                                                 "word_level_timestamps/data/reference/encoder_attention_qks.npy";
    const auto reference_tensors = read_vector_of_tensors_from_np(reference_path);
    const std::filesystem::path current_path = "/home/asuvorov/projects/openvino.genai/.vscode/tasks/"
                                               "word_level_timestamps/data/current/encoder_attention_qks.npy";
    const auto current_tensors = read_vector_of_tensors_from_np(current_path);

    const bool tensors_are_close = tensors_close(reference_tensors, current_tensors, WORD_TS_ACCURACY);
    if (!tensors_are_close) {
        throw std::runtime_error("Encoder attention QK tensors do not match reference");
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Encoder attention QK tensors test passed in " << duration << " ms!\n";
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<WAV_FILE_PATH>\" <DEVICE>");
    }

    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    std::string device = (argc == 4) ? argv[3] : "CPU";  // Default to CPU if no device is provided

    ov::AnyMap ov_config;
    if (device == "NPU" ||
        device.find("GPU") != std::string::npos) {  // need to handle cases like "GPU", "GPU.0" and "GPU.1"
        // Cache compiled models on disk for GPU and NPU to save time on the
        // next run. It's not beneficial for CPU.
        ov_config = get_config_for_cache();
    }

    ov_config.insert({ov::genai::word_timestamps.name(), true});

    ov::genai::WhisperPipeline pipeline(models_path, device, ov_config);

    ov::genai::WhisperGenerationConfig config = pipeline.get_generation_config();
    // 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>";  // can switch to <|zh|> for Chinese language
    config.task = "transcribe";
    config.return_timestamps = false;
    config.word_timestamps = true;

    // Pipeline expects normalized audio with Sample Rate of 16kHz
    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);
    auto result = pipeline.generate(raw_speech, config);

    std::cout << result << "\n";

    std::cout << std::fixed << std::setprecision(2);
    // for (auto& chunk : *result.chunks) {
    //     std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text <<
    //     "\n";
    // }

    // test_encoder_attention_qks(models_path,
    //                            "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/"
    //                            "whisper/librispeech_asr_dummy_wav_samples");

    // test_word_level_timestamps(models_path,
    //                            "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/"
    //                            "whisper/librispeech_asr_dummy_wav_samples",
    //                            "/home/asuvorov/projects/openvino.genai/tests/python_tests/data/whisper/"
    //                            "librispeech_asr_dummy_word_timestamps_reference_tiny.json");

    if (result.words) {
        std::cout << "Word-level timestamps:\n";
        for (const auto& word_info : *result.words) {
            std::cout << "  " << word_info.word << "  " << word_info.start_ts << " - " << word_info.end_ts << "\n";
        }
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
