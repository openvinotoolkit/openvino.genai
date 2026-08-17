// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "openvino/core/except.hpp"

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>

#ifdef _WIN32
#    include <fcntl.h>
#    include <io.h>
#endif

namespace {
bool is_wav_buffer(const std::string buf) {
    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    if (buf.size() < 12 || buf.substr(0, 4) != "RIFF" || buf.substr(8, 4) != "WAVE") {
        return false;
    }

    uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(buf.data() + 4);
    if (chunk_size + 8 != buf.size()) {
        return false;
    }

    return true;
}
}  // namespace

namespace utils {
namespace audio {

#define COMMON_SAMPLE_RATE 16000

ov::genai::RawSpeechInput read_wav(const std::string& filename) {
    drwav wav;
    std::vector<uint8_t> wav_data;  // used for pipe input from stdin or ffmpeg decoding output

    if (filename == "-") {
        {
#ifdef _WIN32
            _setmode(_fileno(stdin), _O_BINARY);
#endif

            uint8_t buf[1024];
            while (true) {
                const size_t n = fread(buf, 1, sizeof(buf), stdin);
                if (n == 0) {
                    break;
                }
                wav_data.insert(wav_data.end(), buf, buf + n);
            }
        }

        OPENVINO_ASSERT(drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr),
                        "Failed to open WAV file from stdin");

        fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, wav_data.size());
    } else if (is_wav_buffer(filename)) {
        OPENVINO_ASSERT(drwav_init_memory(&wav, filename.c_str(), filename.size(), nullptr),
                        "Failed to open WAV file from fname buffer");
    } else if (!drwav_init_file(&wav, filename.c_str(), nullptr)) {
#if defined(WHISPER_FFMPEG)
        OPENVINO_ASSERT(ffmpeg_decode_audio(fname, wav_data) == 0, "Failed to ffmpeg decode")

        OPENVINO_ASSERT(drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr),
                        "Failed to read wav data as wav")
#else
        throw std::runtime_error("failed to open as WAV file");
#endif
    }

    if (wav.channels != 1 && wav.channels != 2) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be mono or stereo");
    }

    if (wav.sampleRate != COMMON_SAMPLE_RATE) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be " + std::to_string(COMMON_SAMPLE_RATE / 1000) + " kHz");
    }

    const uint64_t n =
        wav_data.empty()
        ? wav.totalPCMFrameCount
        : (
            wav_data.size() /
            (static_cast<uint64_t>(wav.channels) * static_cast<uint64_t>(wav.bitsPerSample) / 8ul)
        );

    std::vector<int16_t> pcm16;
    pcm16.resize(n * wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    // convert to mono, float
    std::vector<float> pcmf32;
    pcmf32.resize(n);
    if (wav.channels == 1) {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[i]) / 32768.0f;
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
        }
    }

    return pcmf32;
}

ov::Tensor read_wav_as_tensor(const std::string& filename) {
    ov::genai::RawSpeechInput pcm = read_wav(filename);
    const size_t sample_count = pcm.size();

    // Move the decoded samples into an allocator so the tensor owns them directly, avoiding a
    // copy of the PCM buffer. Mirrors SharedImageAllocator in visual_language_chat/load_image.cpp.
    // The buffer is held via shared_ptr so copying the allocator is cheap and ownership is unambiguous.
    struct SharedPcmAllocator {
        std::shared_ptr<std::vector<float>> pcm;
        void* allocate(size_t bytes, size_t) const {
            OPENVINO_ASSERT(bytes == pcm->size() * sizeof(float),
                            "Unexpected number of bytes was requested to allocate.");
            return pcm->data();
        }
        void deallocate(void*, size_t, size_t) const noexcept {}
        bool is_equal(const SharedPcmAllocator& other) const noexcept {
            return pcm == other.pcm;
        }
    };
    return ov::Tensor(ov::element::f32,
                      ov::Shape{sample_count},
                      SharedPcmAllocator{std::make_shared<std::vector<float>>(std::move(pcm))});
}

void save_to_wav(const float* waveform_ptr,
                 size_t waveform_size,
                 const std::filesystem::path& file_path,
                 uint32_t bits_per_sample,
                 uint32_t sample_rate) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = sample_rate;
    format.bitsPerSample = bits_per_sample;

    drwav wav;
    OPENVINO_ASSERT(drwav_init_file_write(&wav, file_path.string().c_str(), &format, nullptr),
                    "Failed to initialize WAV writer");

    size_t total_samples = waveform_size * format.channels;

    drwav_uint64 frames_written = drwav_write_pcm_frames(&wav, total_samples, waveform_ptr);
    OPENVINO_ASSERT(frames_written == total_samples, "Failed to write not all frames");

    drwav_uninit(&wav);
}

ov::Tensor read_speaker_embedding(const std::filesystem::path& file_path, const ov::Shape& shape) {
    std::ifstream input(file_path, std::ios::binary);
    OPENVINO_ASSERT(input, "Failed to open speaker embedding file: " + file_path.string());

    input.seekg(0, std::ios::end);
    const size_t buffer_size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);

    OPENVINO_ASSERT(buffer_size % sizeof(float) == 0,
                    "Speaker embedding file size is not a multiple of sizeof(float).");
    const size_t num_floats = buffer_size / sizeof(float);

    // Validate that file size matches expected shape.
    size_t expected_floats = 1;
    for (size_t d : shape) expected_floats *= d;
    OPENVINO_ASSERT(num_floats == expected_floats,
                    "Speaker embedding file contains ", num_floats, " float32 values "
                    "but expected shape ", shape, " (total ", expected_floats, " floats).");

    ov::Tensor tensor(ov::element::f32, shape);
    input.read(reinterpret_cast<char*>(tensor.data()), buffer_size);
    OPENVINO_ASSERT(input, "Failed to read all data from speaker embedding file.");
    return tensor;
}

}  // namespace audio
}  // namespace utils
