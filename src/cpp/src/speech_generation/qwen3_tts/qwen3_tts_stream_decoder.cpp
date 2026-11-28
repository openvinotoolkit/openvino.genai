// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_stream_decoder.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <regex>

#include <nlohmann/json.hpp>

#include "omni/speech_streamer_utils.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"

namespace ov::genai::qwen3_tts {

namespace {

void parse_graph_array(const nlohmann::json& graphs,
                       const std::filesystem::path& models_path,
                       std::vector<StreamDecoderSpec>& out) {
    if (!graphs.is_array()) {
        return;
    }
    for (const auto& entry : graphs) {
        if (!entry.is_object() || !entry.contains("file") || !entry["file"].is_string()) {
            continue;
        }
        const auto file_path = models_path / entry["file"].get<std::string>();
        if (!std::filesystem::exists(file_path)) {
            continue;
        }
        StreamDecoderSpec spec;
        spec.path = file_path;
        spec.left_context_frames = entry.value("left_context", entry.value("left_context_frames", static_cast<size_t>(0)));
        spec.chunk_frames = entry.value("chunk_frames", static_cast<size_t>(0));
        if (spec.chunk_frames > 0) {
            out.push_back(std::move(spec));
        }
    }
}

}  // namespace

std::vector<StreamDecoderSpec> discover_stream_decoder_specs(const std::filesystem::path& models_path) {
    std::vector<StreamDecoderSpec> specs;
    if (!std::filesystem::exists(models_path)) {
        return specs;
    }

    const auto ov_config_path = models_path / "openvino_config.json";
    if (std::filesystem::exists(ov_config_path)) {
        std::ifstream ifs(ov_config_path);
        if (ifs.is_open()) {
            const nlohmann::json cfg = nlohmann::json::parse(ifs, nullptr, false);
            if (!cfg.is_discarded() && cfg.is_object()) {
                if (cfg.contains("streaming_codec_decoders")) {
                    const auto& node = cfg["streaming_codec_decoders"];
                    if (node.is_array()) {
                        parse_graph_array(node, models_path, specs);
                    } else if (node.is_object() && node.contains("graphs")) {
                        parse_graph_array(node["graphs"], models_path, specs);
                    }
                }
                if (specs.empty() && cfg.contains("qwen3_tts") && cfg["qwen3_tts"].is_object()) {
                    const auto& q3 = cfg["qwen3_tts"];
                    if (q3.contains("streaming_decoders") && q3["streaming_decoders"].is_object() &&
                        q3["streaming_decoders"].contains("graphs")) {
                        parse_graph_array(q3["streaming_decoders"]["graphs"], models_path, specs);
                    }
                }
            }
        }
    }

    if (specs.empty()) {
        static const std::regex pattern(R"(openvino_codec_decoder_stream_c(\d+)_t(\d+)\.xml)");
        for (const auto& entry : std::filesystem::directory_iterator(models_path)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string filename = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(filename, match, pattern)) {
                StreamDecoderSpec spec;
                spec.path = entry.path();
                spec.left_context_frames = static_cast<size_t>(std::stoull(match[1].str()));
                spec.chunk_frames = static_cast<size_t>(std::stoull(match[2].str()));
                if (spec.chunk_frames > 0) {
                    specs.push_back(std::move(spec));
                }
            }
        }
    }

    std::sort(specs.begin(), specs.end(), [](const StreamDecoderSpec& a, const StreamDecoderSpec& b) {
        if (a.left_context_frames != b.left_context_frames) {
            return a.left_context_frames < b.left_context_frames;
        }
        return a.chunk_frames < b.chunk_frames;
    });
    return specs;
}

void Qwen3TTSStreamDecoderPool::initialize(const std::filesystem::path& models_path,
                                           const std::string& decoder_device,
                                           const ov::AnyMap& decoder_properties,
                                           uint32_t num_quantizers,
                                           uint32_t decode_upsample_rate) {
    m_num_quantizers = num_quantizers;
    m_decode_upsample_rate = decode_upsample_rate;
    m_tiers.clear();

    const auto specs = discover_stream_decoder_specs(models_path);
    for (const auto& spec : specs) {
        const std::string role_name = std::string(roles::SPEECH_TOKENIZER_DECODER) + "_stream_c" +
                                      std::to_string(spec.left_context_frames) + "_t" +
                                      std::to_string(spec.chunk_frames);
        auto request = compile_request(spec.path, role_name, decoder_device, decoder_properties);
        register_compiled_tier(spec.left_context_frames, spec.chunk_frames, std::move(request));
    }
}

void Qwen3TTSStreamDecoderPool::register_compiled_tier(size_t left_context_frames,
                                                       size_t chunk_frames,
                                                       ov::InferRequest request) {
    CompiledTier tier;
    tier.left_context_frames = left_context_frames;
    tier.chunk_frames = chunk_frames;
    tier.input_frames = left_context_frames + chunk_frames;
    tier.request = std::move(request);

    try {
        const auto pshape = tier.request.get_compiled_model().input(0).get_partial_shape();
        if (pshape.rank().is_static() && pshape.size() == 3 && pshape[2].is_static()) {
            tier.static_input = true;
            tier.input_frames = static_cast<size_t>(pshape[2].get_length());
        }
    } catch (...) {
        tier.static_input = false;
    }
    m_tiers.push_back(std::move(tier));
}

Qwen3TTSStreamDecoderPool::CompiledTier* Qwen3TTSStreamDecoderPool::select_tier(size_t left_context_frames,
                                                                                size_t chunk_frames) {
    CompiledTier* exact = nullptr;
    CompiledTier* smallest_fit = nullptr;
    const size_t required_window = left_context_frames + chunk_frames;
    for (auto& tier : m_tiers) {
        if (tier.left_context_frames == left_context_frames && tier.chunk_frames == chunk_frames) {
            exact = &tier;
            break;
        }
        if (tier.left_context_frames == left_context_frames && tier.input_frames >= required_window) {
            if (!smallest_fit || tier.input_frames < smallest_fit->input_frames) {
                smallest_fit = &tier;
            }
        }
    }
    return exact ? exact : smallest_fit;
}

Qwen3TTSStreamSession::Qwen3TTSStreamSession(Qwen3TTSStreamDecoderPool& pool,
                                             Qwen3TTSStreamDecoderPool::FallbackChunkDecodeFn fallback_decode,
                                             const SpeechGenerationConfig& config,
                                             const AudioStreamerVariant& streamer,
                                             std::chrono::steady_clock::time_point generation_start,
                                             const std::vector<int64_t>& prefix_ref_codes)
    : m_pool(pool),
      m_fallback_decode(std::move(fallback_decode)),
      m_streamer(streamer),
      m_generation_start(generation_start),
      m_num_quantizers(pool.num_quantizers()),
      m_upsample_rate(pool.decode_upsample_rate()),
      m_first_chunk_frames(config.first_chunk_frames),
      m_steady_chunk_frames(config.steady_chunk_frames),
      m_first_context_frames(config.first_context_frames),
      m_steady_context_frames(config.steady_context_frames),
      m_async(config.async_audio_decode),
      m_streaming_active(is_speech_streamer_active(streamer)) {
    if (!prefix_ref_codes.empty()) {
        OPENVINO_ASSERT(prefix_ref_codes.size() % m_num_quantizers == 0,
                        "prefix_ref_codes length must be a multiple of num_quantizers");
        m_all_codes = prefix_ref_codes;
        m_emitted_frames = prefix_ref_codes.size() / m_num_quantizers;
    }

    if (m_async) {
        m_worker_thread = std::thread([this]() { worker_loop(); });
    }
}

Qwen3TTSStreamSession::~Qwen3TTSStreamSession() {
    if (!m_finished) {
        stop_worker_for_cleanup();
        if (m_streaming_active) {
            try {
                end_speech_streamer(m_streamer);
            } catch (...) {
            }
        }
    }
}

bool Qwen3TTSStreamSession::push_frame(const int64_t* frame_codes, size_t num_code_groups) {
    rethrow_worker_error_if_any();
    if (should_stop()) {
        return false;
    }
    OPENVINO_ASSERT(frame_codes != nullptr && num_code_groups == m_num_quantizers,
                    "push_frame expected ", m_num_quantizers, " codec groups, got ", num_code_groups);

    m_all_codes.insert(m_all_codes.end(), frame_codes, frame_codes + num_code_groups);
    ++m_pending_frames;

    const size_t threshold = (m_chunk_index == 0) ? m_first_chunk_frames : m_steady_chunk_frames;
    if (m_pending_frames >= threshold) {
        enqueue_or_run(build_task(/*is_final=*/false));
    }

    rethrow_worker_error_if_any();
    return !should_stop();
}

StreamSessionResult Qwen3TTSStreamSession::finish() {
    if (m_finished) {
        return {};
    }
    m_finished = true;

    SpeechStreamerGuard streamer_guard(m_streamer, m_streaming_active);
    try {
        rethrow_worker_error_if_any();
        if (m_pending_frames > 0 && !should_stop()) {
            enqueue_or_run(build_task(/*is_final=*/true));
        }

        if (m_async) {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_stop_worker = true;
            }
            m_cv.notify_all();
            if (m_worker_thread.joinable()) {
                m_worker_thread.join();
            }
            rethrow_worker_error_if_any();
        }
    } catch (...) {
        stop_worker_for_cleanup();
        streamer_guard.end();
        throw;
    }

    streamer_guard.end();

    StreamSessionResult result;
    const auto status = static_cast<StreamingStatus>(m_stop_status.load(std::memory_order_acquire));
    result.stopped_early = (status != StreamingStatus::RUNNING);
    result.cancelled = (status == StreamingStatus::CANCEL);
    if (!result.cancelled) {
        result.waveform = std::move(m_accumulated_waveform);
    }
    result.ttfa_us = m_ttfa_us;
    result.has_ttfa = m_has_ttfa;
    result.total_decode_us = m_total_decode_us;
    return result;
}

Qwen3TTSStreamSession::DecodeTask Qwen3TTSStreamSession::build_task(bool is_final) {
    const size_t total_frames = m_all_codes.size() / m_num_quantizers;
    const size_t new_frames = total_frames - m_emitted_frames;
    const bool first_chunk = (m_chunk_index == 0);
    const size_t target_context = first_chunk ? m_first_context_frames : m_steady_context_frames;
    const size_t target_chunk = first_chunk ? m_first_chunk_frames : m_steady_chunk_frames;

    auto* tier = m_pool.select_tier(target_context, is_final ? new_frames : target_chunk);
    if (!tier && is_final && new_frames < target_chunk) {
        tier = m_pool.select_tier(target_context, target_chunk);
    }

    const size_t context_start = (m_emitted_frames > target_context) ? (m_emitted_frames - target_context) : 0;
    const size_t available_ctx = m_emitted_frames - context_start;

    DecodeTask task;
    task.chunk_index = m_chunk_index;
    task.new_frames = new_frames;
    task.static_tier = tier;

    // For static tiers, prepend left-context padding if fewer than target_context frames exist yet.
    const size_t effective_ctx = tier ? target_context : available_ctx;
    const size_t left_pad_frames = (effective_ctx > available_ctx) ? (effective_ctx - available_ctx) : 0;
    task.context_frames = effective_ctx;

    const size_t real_window_frames = left_pad_frames + available_ctx + new_frames;
    const size_t final_window_frames = (tier && tier->static_input) ? tier->input_frames : real_window_frames;
    OPENVINO_ASSERT(real_window_frames <= final_window_frames,
                    "Streaming decoder window (", real_window_frames, ") exceeds static input frames (",
                    final_window_frames, ")");

    task.window_frames = final_window_frames;
    task.window_frame_major.reserve(final_window_frames * m_num_quantizers);

    if (left_pad_frames > 0) {
        const int64_t* pad_src =
            m_all_codes.data() + ((available_ctx > 0 ? context_start : m_emitted_frames) * m_num_quantizers);
        for (size_t i = 0; i < left_pad_frames; ++i) {
            task.window_frame_major.insert(task.window_frame_major.end(), pad_src, pad_src + m_num_quantizers);
        }
    }

    const int64_t* real_begin = m_all_codes.data() + context_start * m_num_quantizers;
    const int64_t* real_end = m_all_codes.data() + total_frames * m_num_quantizers;
    task.window_frame_major.insert(task.window_frame_major.end(), real_begin, real_end);

    // Right-edge replicate padding when a final short chunk is fed to a static streaming graph.
    if (final_window_frames > real_window_frames) {
        const size_t right_pad_frames = final_window_frames - real_window_frames;
        const int64_t* last_frame = m_all_codes.data() + (total_frames - 1) * m_num_quantizers;
        for (size_t i = 0; i < right_pad_frames; ++i) {
            task.window_frame_major.insert(task.window_frame_major.end(), last_frame, last_frame + m_num_quantizers);
        }
    }

    m_emitted_frames = total_frames;
    m_pending_frames = 0;
    ++m_chunk_index;
    return task;
}

void Qwen3TTSStreamSession::enqueue_or_run(DecodeTask&& task) {
    if (!m_async) {
        execute_task(task);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_worker_error) {
            std::rethrow_exception(m_worker_error);
        }
        m_queue.push_back(std::move(task));
    }
    m_cv.notify_one();
}

void Qwen3TTSStreamSession::execute_task(const DecodeTask& task) {
    if (task.new_frames == 0 || should_stop()) {
        return;
    }

    const auto decode_start = std::chrono::steady_clock::now();
    std::vector<float> chunk_samples;

    if (task.static_tier != nullptr) {
        // Transpose [window_frames, Q] -> channel-first [1, Q, window_frames] matching upstream contract.
        ov::Tensor audio_codes(ov::element::i64, ov::Shape{1, m_num_quantizers, task.window_frames});
        int64_t* dst = audio_codes.data<int64_t>();
        const int64_t* src = task.window_frame_major.data();
        for (size_t t = 0; t < task.window_frames; ++t) {
            for (size_t q = 0; q < m_num_quantizers; ++q) {
                dst[q * task.window_frames + t] = src[t * m_num_quantizers + q];
            }
        }

        auto& req = task.static_tier->request;
        req.set_tensor("audio_codes", audio_codes);
        req.infer();

        const auto out = req.get_output_tensor(0);
        OPENVINO_ASSERT(out.get_element_type() == ov::element::f32,
                        "Streaming codec decoder output must be f32");
        const float* out_ptr = out.data<const float>();
        const size_t out_size = out.get_size();

        const size_t full_window_samples = task.window_frames * m_upsample_rate;
        const size_t ctx_samples = task.context_frames * m_upsample_rate;
        const size_t emit_samples = task.new_frames * m_upsample_rate;

        // Static streaming decoders with baked left-context trim emit T*upsample samples directly;
        // decoders without baked trim emit (C+T)*upsample samples and require host-side left trim.
        size_t start_offset = 0;
        if (task.context_frames > 0 && out_size >= full_window_samples) {
            start_offset = ctx_samples;
        }
        const size_t available = (out_size > start_offset) ? (out_size - start_offset) : 0;
        const size_t count = std::min(emit_samples, available);
        chunk_samples.assign(out_ptr + start_offset, out_ptr + start_offset + count);
    } else {
        OPENVINO_ASSERT(m_fallback_decode, "No static streaming decoder tier or fallback decoder available");
        chunk_samples = m_fallback_decode(task.window_frame_major, task.context_frames, task.new_frames);
    }

    const auto decode_end = std::chrono::steady_clock::now();
    m_total_decode_us += static_cast<double>(PerfMetrics::get_microsec(decode_end - decode_start));

    if (!m_has_ttfa && !chunk_samples.empty()) {
        m_ttfa_us = static_cast<double>(PerfMetrics::get_microsec(decode_end - m_generation_start));
        m_has_ttfa = true;
    }

    if (m_streaming_active && !chunk_samples.empty()) {
        ov::Tensor chunk_tensor(ov::element::f32, ov::Shape{1, 1, chunk_samples.size()});
        std::memcpy(chunk_tensor.data<float>(), chunk_samples.data(), chunk_samples.size() * sizeof(float));
        const StreamingStatus status = invoke_speech_streamer(m_streamer, chunk_tensor);
        if (status == StreamingStatus::STOP || status == StreamingStatus::CANCEL) {
            m_stop_status.store(static_cast<int>(status), std::memory_order_release);
            if (status == StreamingStatus::CANCEL) {
                return;
            }
        }
    }

    m_accumulated_waveform.insert(m_accumulated_waveform.end(), chunk_samples.begin(), chunk_samples.end());
}

void Qwen3TTSStreamSession::worker_loop() {
    while (true) {
        DecodeTask task;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this]() { return m_stop_worker || !m_queue.empty(); });
            if (m_queue.empty()) {
                if (m_stop_worker) {
                    return;
                }
                continue;
            }
            task = std::move(m_queue.front());
            m_queue.pop_front();
        }
        try {
            execute_task(task);
            if (should_stop()) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_queue.clear();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_worker_error = std::current_exception();
            m_stop_worker = true;
            m_queue.clear();
            m_stop_status.store(static_cast<int>(StreamingStatus::CANCEL), std::memory_order_release);
            m_cv.notify_all();
            return;
        }
    }
}

void Qwen3TTSStreamSession::stop_worker_for_cleanup() noexcept {
    if (!m_async) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop_worker = true;
        m_queue.clear();
    }
    m_cv.notify_all();
    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }
}

void Qwen3TTSStreamSession::rethrow_worker_error_if_any() {
    if (!m_async) {
        return;
    }
    std::exception_ptr err;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        err = m_worker_error;
    }
    if (err) {
        std::rethrow_exception(err);
    }
}

}  // namespace ov::genai::qwen3_tts
