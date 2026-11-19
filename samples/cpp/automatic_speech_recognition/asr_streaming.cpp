// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Streaming ASR with ov::genai::ASRStreamingSession (Qwen3-ASR, Voxtral Realtime).
//
// The WAV file is fed at real-time speed in STEP_MS pieces, as a live microphone would deliver it. After every push
// the available results are read and printed as a timeline. At the end the streaming latency is compared with offline
// ASRPipeline::generate() over the same audio.
//
// With --streams N, N sessions transcribe the same file concurrently from N threads. On the PagedAttention backend
// (--attention-backend PA) the pipeline batches their inference.
//
// Usage:
//   asr_streaming <MODEL_DIR> <WAV_FILE> [--device DEVICE] [--chunk-sec SEC] [--step-ms MS]
//                 [--window-chunks N] [--unfixed-chunks N] [--unfixed-tokens N] [--streams N]
//                 [--attention-backend PA|SDPA]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <thread>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <windows.h>
#endif

#include "../speech_generation/audio_utils.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t SAMPLE_RATE = 16000;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<std::string> split_words(const std::string& text) {
    std::istringstream stream{text};
    return {std::istream_iterator<std::string>{stream}, std::istream_iterator<std::string>{}};
}

/// Word-level edit distance: substituted, inserted and deleted words.
size_t word_differences(const std::string& text, const std::string& reference) {
    const std::vector<std::string> words = split_words(text);
    const std::vector<std::string> ref = split_words(reference);
    std::vector<size_t> row(ref.size() + 1);
    for (size_t j = 0; j <= ref.size(); ++j) {
        row[j] = j;
    }
    for (size_t i = 1; i <= words.size(); ++i) {
        size_t diagonal = row[0];
        row[0] = i;
        for (size_t j = 1; j <= ref.size(); ++j) {
            const size_t above = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (words[i - 1] == ref[j - 1] ? 0 : 1)});
            diagonal = above;
        }
    }
    return row[ref.size()];
}

/// Prints one timeline line per result and collects latencies of one stream.
class StreamReport {
public:
    StreamReport(size_t stream_id, float chunk_sec, float audio_sec, Clock::time_point audio_start)
        : m_stream_id{stream_id},
          m_chunk_sec{chunk_sec},
          m_audio_sec{audio_sec},
          m_audio_start{audio_start} {}

    // read_ms: time spent in ASRStreamingSession::read().
    // Latency: time from the moment the last sample of the chunk was spoken until read() returned its result.
    void add(const ov::genai::ASRStreamingResult& result, double read_ms) {
        const float begin_sec = std::min(m_chunk_sec * m_results, m_audio_sec);
        const float end_sec = result.is_final ? m_audio_sec : std::min(begin_sec + m_chunk_sec, m_audio_sec);
        const double latency_ms = ms_since(m_audio_start) - end_sec * 1000.0;
        m_latencies_ms.push_back(latency_ms);
        m_compute_ms += read_ms;
        ++m_results;
        if (result.is_final) {
            m_text = result.committed_text;
        }

        static std::mutex print_mutex;
        const std::lock_guard<std::mutex> lock{print_mutex};
        std::printf("[stream %zu] %-7s audio %5.1f-%5.1f s | read %6.0f ms | latency %6.0f ms | +\"%s\" [%s]\n",
                    m_stream_id,
                    result.is_final ? "final" : "partial",
                    begin_sec,
                    end_sec,
                    read_ms,
                    latency_ms,
                    result.new_committed_text.c_str(),
                    result.partial_text.c_str());
    }

    const std::string& text() const {
        return m_text;
    }
    const std::vector<double>& latencies_ms() const {
        return m_latencies_ms;
    }
    /// Total time spent in read(), i.e. the inference cost of the whole stream.
    double compute_ms() const {
        return m_compute_ms;
    }

private:
    size_t m_stream_id;
    float m_chunk_sec;
    float m_audio_sec;
    Clock::time_point m_audio_start;
    size_t m_results = 0;
    double m_compute_ms = 0;
    std::vector<double> m_latencies_ms;
    std::string m_text;
};

StreamReport transcribe(ov::genai::ASRPipeline& pipeline,
                        const ov::genai::ASRStreamingConfig& streaming_config,
                        const std::vector<float>& audio,
                        size_t step_samples,
                        size_t stream_id) {
    const auto audio_start = Clock::now();
    StreamReport report{stream_id,
                        streaming_config.chunk_size_sec,
                        static_cast<float>(audio.size()) / SAMPLE_RATE,
                        audio_start};
    // Reads all results that are due and adds them to the report.
    const auto read_available = [&](ov::genai::ASRStreamingSession& session) {
        while (true) {
            const auto read_start = Clock::now();
            const std::optional<ov::genai::ASRStreamingResult> result = session.read();
            if (!result) {
                break;
            }
            report.add(*result, ms_since(read_start));
        }
    };

    // 1. Open a streaming session on the pipeline.
    ov::genai::ASRStreamingSession session(pipeline, streaming_config);

    for (size_t pos = 0; pos < audio.size(); pos += step_samples) {
        const size_t end = std::min(pos + step_samples, audio.size());
        // Simulate a live source: the piece is delivered once it has been spoken.
        std::this_thread::sleep_until(audio_start + std::chrono::milliseconds(end * 1000 / SAMPLE_RATE));

        // 2. Feed audio. push() only buffers it.
        session.push({audio.begin() + pos, audio.begin() + end});
        // 3. Read results. read() runs inference once a full chunk of audio is buffered, else returns std::nullopt.
        read_available(session);
    }

    // 4. Close input. The next read() calls transcribe the remaining audio; the last result has is_final == true.
    session.end();
    read_available(session);
    return report;
}

void print_summary(const std::vector<std::optional<StreamReport>>& reports,
                   float audio_sec,
                   double offline_ms,
                   const std::string& offline_text) {
    std::vector<double> latencies;
    double final_latency_sum = 0;
    double compute_sum = 0;
    for (const std::optional<StreamReport>& report : reports) {
        const std::vector<double>& stream_latencies = report->latencies_ms();
        latencies.insert(latencies.end(), stream_latencies.begin(), stream_latencies.end());
        final_latency_sum += stream_latencies.back();
        compute_sum += report->compute_ms();
    }
    std::sort(latencies.begin(), latencies.end());
    const auto percentile = [&](double p) {
        return latencies[std::min(latencies.size() - 1, static_cast<size_t>(p * latencies.size()))];
    };
    const double mean_compute = compute_sum / reports.size();

    // Latency: offline generate() starts only after the speech has ended, so all of its time is latency.
    // Streaming runs most inference while the speech goes on; only the last chunks remain after it ends.
    std::cout << "\n=== Latency: time from end of speech to text ===\n";
    std::printf("streaming, every result: p50 %6.0f ms | p95 %6.0f ms | max %6.0f ms\n",
                percentile(0.5),
                percentile(0.95),
                latencies.back());
    std::printf("streaming, final result: %6.0f ms\n", final_latency_sum / reports.size());
    std::printf("offline generate()     : %6.0f ms\n", offline_ms);

    // Compute: total inference time of the whole audio, per stream. Above 1.0 s per 1 s of audio, streaming can't
    // keep up with a live source and its latency grows with every chunk.
    std::cout << "\n=== Compute: total inference time per stream ===\n";
    std::printf("streaming, sum of read(): %6.0f ms | %4.2f s per 1 s of audio\n",
                mean_compute,
                mean_compute / 1000.0 / audio_sec);
    std::printf("offline generate()      : %6.0f ms | %4.2f s per 1 s of audio\n",
                offline_ms,
                offline_ms / 1000.0 / audio_sec);

    std::cout << "\n=== Text ===\n";
    for (size_t id = 0; id < reports.size(); ++id) {
        const std::string& text = reports[id]->text();
        std::cout << "[stream " << id << "] " << word_differences(text, offline_text) << " of "
                  << split_words(offline_text).size() << " words differ from offline: " << text << "\n";
    }
    std::cout << "[offline] " << offline_text << "\n";
}

}  // namespace

int main(int argc, char* argv[]) try {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <WAV_FILE> [--device DEVICE] [--chunk-sec SEC] [--step-ms MS] "
                                 "[--window-chunks N] [--unfixed-chunks N] [--unfixed-tokens N] [--streams N] "
                                 "[--attention-backend PA|SDPA]");
    }

    const std::filesystem::path models_path = argv[1];
    const std::string wav_file = argv[2];
    std::string device = "CPU";
    size_t step_ms = 500;
    size_t streams = 1;
    ov::AnyMap properties;
    ov::genai::ASRStreamingConfig streaming_config;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--device") {
            device = value;
        } else if (arg == "--chunk-sec") {
            streaming_config.chunk_size_sec = std::stof(value);
        } else if (arg == "--step-ms") {
            step_ms = std::stoul(value);
        } else if (arg == "--window-chunks") {
            streaming_config.window_chunk_num = std::stoul(value);
        } else if (arg == "--unfixed-chunks") {
            streaming_config.unfixed_chunk_num = std::stoul(value);
        } else if (arg == "--unfixed-tokens") {
            streaming_config.unfixed_token_num = std::stoul(value);
        } else if (arg == "--streams") {
            streams = std::stoul(value);
        } else if (arg == "--attention-backend") {
            properties["ATTENTION_BACKEND"] = value;
        } else {
            throw std::runtime_error("Unexpected argument: " + arg);
        }
    }

    ov::genai::ASRPipeline pipeline(models_path, device, properties);
    const ov::genai::RawSpeechInput audio = utils::audio::read_wav(wav_file);
    const size_t step_samples = std::max<size_t>(1, step_ms * SAMPLE_RATE / 1000);

    std::vector<std::optional<StreamReport>> reports(streams);
    std::vector<std::exception_ptr> errors(streams);
    std::vector<std::thread> threads;
    for (size_t id = 0; id < streams; ++id) {
        threads.emplace_back([&, id] {
            try {
                reports[id] = transcribe(pipeline, streaming_config, audio, step_samples, id);
            } catch (...) {
                errors[id] = std::current_exception();
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (const std::exception_ptr& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }

    // Offline baseline: the whole file in one generate() call, which can only start once the speech has ended.
    const auto offline_start = Clock::now();
    const ov::genai::ASRDecodedResults offline = pipeline.generate(audio);
    const double offline_ms = ms_since(offline_start);

    print_summary(reports, static_cast<float>(audio.size()) / SAMPLE_RATE, offline_ms, offline.texts.at(0));
    return 0;
} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
}
