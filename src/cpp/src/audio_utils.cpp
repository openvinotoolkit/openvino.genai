// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/audio_utils.hpp"

#include <iostream>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "openvino/genai/dr_wav.h"

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

namespace ov {
namespace genai {
namespace utils {
namespace audio {

#define COMMON_SAMPLE_RATE 16000

bool read_wav(const std::string& fname,
              std::vector<float>& pcmf32,
              std::vector<std::vector<float>>& pcmf32s,
              bool stereo) {
    drwav wav;
    std::vector<uint8_t> wav_data;  // used for pipe input from stdin or ffmpeg decoding output

    if (fname == "-") {
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

        if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false) {
            fprintf(stderr, "error: failed to open WAV file from stdin\n");
            return false;
        }

        fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, wav_data.size());
    } else if (is_wav_buffer(fname)) {
        if (drwav_init_memory(&wav, fname.c_str(), fname.size(), nullptr) == false) {
            fprintf(stderr, "error: failed to open WAV file from fname buffer\n");
            return false;
        }
    } else if (drwav_init_file(&wav, fname.c_str(), nullptr) == false) {
#if defined(WHISPER_FFMPEG)
        if (ffmpeg_decode_audio(fname, wav_data) != 0) {
            fprintf(stderr, "error: failed to ffmpeg decode '%s' \n", fname.c_str());
            return false;
        }
        if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false) {
            fprintf(stderr, "error: failed to read wav data as wav \n");
            return false;
        }
#else
        fprintf(stderr, "error: failed to open '%s' as WAV file\n", fname.c_str());
        return false;
#endif
    }

    if (wav.channels != 1 && wav.channels != 2) {
        fprintf(stderr, "%s: WAV file '%s' must be mono or stereo\n", __func__, fname.c_str());
        drwav_uninit(&wav);
        return false;
    }

    if (stereo && wav.channels != 2) {
        fprintf(stderr, "%s: WAV file '%s' must be stereo for diarization\n", __func__, fname.c_str());
        drwav_uninit(&wav);
        return false;
    }

    if (wav.sampleRate != COMMON_SAMPLE_RATE) {
        fprintf(stderr, "%s: WAV file '%s' must be %i kHz\n", __func__, fname.c_str(), COMMON_SAMPLE_RATE / 1000);
        drwav_uninit(&wav);
        return false;
    }

    const uint64_t n =
        wav_data.empty() ? wav.totalPCMFrameCount : wav_data.size() / (wav.channels * wav.bitsPerSample / 8);

    std::vector<int16_t> pcm16;
    pcm16.resize(n * wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    // convert to mono, float
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

    if (stereo) {
        // convert to stereo, float
        pcmf32s.resize(2);

        pcmf32s[0].resize(n);
        pcmf32s[1].resize(n);
        for (uint64_t i = 0; i < n; i++) {
            pcmf32s[0][i] = float(pcm16[2 * i]) / 32768.0f;
            pcmf32s[1][i] = float(pcm16[2 * i + 1]) / 32768.0f;
        }
    }

    return true;
}
}  // namespace audio
}  // namespace utils
}  // namespace genai
}  // namespace ov