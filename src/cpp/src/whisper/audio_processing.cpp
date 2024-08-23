// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_processing.hpp"

#ifdef _WIN32
#    define _USE_MATH_DEFINES
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <openvino/core/except.hpp>
#include <thread>
#include <vector>

namespace {

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    const int32_t n_mel = 80;
    const int32_t n_fft = 201;  // 1 + (WHISPER_N_FFT / 2)

    std::vector<float> data;
};

static bool hann_window(int length, bool periodic, std::vector<float>& output) {
    if (output.size() < static_cast<size_t>(length)) {
        output.resize(length);
    }
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }

    return true;
}

#define SIN_COS_N_COUNT WHISPER_N_FFT
static float sin_vals[SIN_COS_N_COUNT];
static float cos_vals[SIN_COS_N_COUNT];

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const std::vector<float>& in, std::vector<float>& out) {
    int N = in.size();

    out.resize(N * 2);
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT);  // t = 2*M_PI*k*n/N
            re += in[n] * cos_vals[idx];                           // cos(t)
            im -= in[n] * sin_vals[idx];                           // sin(t)
        }

        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// input is real-valued
// output is complex-valued
static void fft(const std::vector<float>& in, std::vector<float>& out) {
    out.resize(in.size() * 2);

    int N = in.size();

    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    if (N % 2 == 1) {
        dft(in, out);
        return;
    }

    std::vector<float> even;
    std::vector<float> odd;

    even.reserve(N / 2);
    odd.reserve(N / 2);

    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            even.push_back(in[i]);
        } else {
            odd.push_back(in[i]);
        }
    }

    std::vector<float> even_fft;
    std::vector<float> odd_fft;

    fft(even, even_fft);
    fft(odd, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < N / 2; k++) {
        int idx = k * sin_cos_step;  // t = 2*M_PI*k/N
        float re = cos_vals[idx];    // cos(t)
        float im = -sin_vals[idx];   // sin(t)

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

static void log_mel_spectrogram_worker_thread(int ith,
                                              const std::vector<float>& hann,
                                              const std::vector<float>& samples,
                                              int n_samples,
                                              int frame_size,
                                              int frame_step,
                                              int n_threads,
                                              const whisper_filters& filters,
                                              whisper_mel& mel) {
    std::vector<float> fft_in(frame_size, 0.0);
    std::vector<float> fft_out(2 * frame_size);
    int n_fft = filters.n_fft;
    int i = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    assert(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hanning window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in, fft_out);

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;

            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum += fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                       fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                       fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                       fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }

            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }

            sum = log10(std::max(sum, 1e-10));

            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

// python implementation: https://github.com/huggingface/transformers/blob/check_gemma/src/transformers/audio_utils.py

// mel_scale = "htk"
float hertz_to_mel(float hertz) {
    return 2595 * std::log10(1 + hertz / 700.0);
}

float mel_to_hertz(float mel) {
    return 700 * (std::pow(10, mel / 2595.0) - 1);
}

}  // namespace

namespace ov {
namespace genai {
namespace utils {
namespace audio {

// num_frequency_bins (`int`):
//     Number of frequencies used to compute the spectrogram (should be the same as in `stft`).
// num_mel_filters (`int`):
//     Number of mel filters to generate.
// min_frequency (`float`):
//     Lowest frequency of interest in Hz.
// max_frequency (`float`):
//     Highest frequency of interest in Hz. This should not exceed `sampling_rate / 2`.
// sampling_rate (`int`):
//     Sample rate of the audio waveform.

std::vector<std::vector<float>> mel_filter_bank(const int64_t num_frequency_bins,
                                                const int64_t num_mel_filters,
                                                const int64_t sampling_rate,
                                                const float min_frequency,
                                                const float max_frequency) {
    OPENVINO_ASSERT(max_frequency <= (sampling_rate / 2), "max_frequency should be less or equal sampling_rate / 2");

    float mel_min = hertz_to_mel(min_frequency);
    float mel_max = hertz_to_mel(max_frequency);

    float mel_freqs_step = (mel_max - mel_min) / float(num_mel_filters + 1);
    std::vector<float> mel_freqs(num_mel_filters + 2);
    for (size_t i = 0; i < (int)mel_freqs.size(); i++) {
        mel_freqs[i] = mel_min + i * mel_freqs_step;
    }

    // our points are in Mels, but we use fft bins, so we have to convert
    // from mel to Hz to fft bin number
    for (int i = 0; i < (int)mel_freqs.size(); i++) {
        // melpoints[i] = round(mel2hz(melpoints[i]) * nfft / samplerate);
        mel_freqs[i] = floor(mel_to_hertz(mel_freqs[i]) * ((num_frequency_bins - 1) * 2 + 1) / sampling_rate);
    }

    std::vector<std::vector<float>> filterbank(num_mel_filters, std::vector<float>(num_frequency_bins));
    for (int j = 0; j < num_mel_filters; j++) {
        // Create first half of triangle
        for (int i = int(mel_freqs[j]); i < int(mel_freqs[j + 1]); i++) {
            filterbank[j][i] = (i - mel_freqs[j]) / (mel_freqs[j + 1] - mel_freqs[j]);
        }

        // Create second half of triangle
        for (int i = int(mel_freqs[j + 1]); i < int(mel_freqs[j + 2]); i++) {
            filterbank[j][i] = (mel_freqs[j + 2] - i) / (mel_freqs[j + 2] - mel_freqs[j + 1]);
        }
    }

    return filterbank;
}

// In FFT, we frequently use sine and cosine operations with the same values.
// We can use precalculated values to speed up the process.
void fill_sin_cos_table() {
    static bool is_filled = false;
    if (is_filled)
        return;
    for (int i = 0; i < SIN_COS_N_COUNT; i++) {
        double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
        sin_vals[i] = sinf(theta);
        cos_vals[i] = cosf(theta);
    }
    is_filled = true;
}

std::vector<float> mel_spectrogram_convert_audio(const std::vector<float> pcmf32,
                                                 const int sample_rate,
                                                 const int frame_size,
                                                 const int frame_step,
                                                 const int n_threads) {
    const float* samples = pcmf32.data();
    const int n_samples = pcmf32.size();
    whisper_filters filters;
    whisper_mel mel;

    std::ifstream file_fdata;
    // todo: handle path. Copy into mel_filters_data.bin into build, install dirs
    file_fdata.open("./assets/whisper/mel_filters_data.bin", std::ios::binary);

    std::vector<float> filter_data;

    OPENVINO_ASSERT(file_fdata.is_open(), "Failed to open file models/mel_filters_data.bin for reading.");

    size_t numElements = filters.n_fft * filters.n_mel;
    filter_data.resize(numElements);

    file_fdata.read(reinterpret_cast<char*>(&filter_data[0]), numElements * sizeof(float));
    file_fdata.close();

    filters.data = filter_data;

    // Hanning window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    std::vector<float> hann;
    hann_window(frame_size, true, hann);

    // Calculate the length of padding
    int64_t stage_1_pad = sample_rate * 30;
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad,
              samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad,
              0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());
    // eric tmp
    samples_padded.resize(stage_1_pad + stage_2_pad * 2);

    mel.n_mel = filters.n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(log_mel_spectrogram_worker_thread,
                                      iw + 1,
                                      std::cref(hann),
                                      samples_padded,
                                      n_samples + stage_2_pad,
                                      frame_size,
                                      frame_step,
                                      n_threads,
                                      std::cref(filters),
                                      std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0,
                                          hann,
                                          samples_padded,
                                          n_samples + stage_2_pad,
                                          frame_size,
                                          frame_step,
                                          n_threads,
                                          filters,
                                          mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0) / 4.0;
    }

    return mel.data;
}

}  // namespace audio
}  // namespace utils
}  // namespace genai
}  // namespace ov
