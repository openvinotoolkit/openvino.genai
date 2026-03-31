// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// sampling_causal_lm: multinomial text generation with configurable sampling
// parameters and a per-run performance breakdown.
//
// Usage:
//   sampling_causal_lm <MODEL_DIR> "<PROMPT>" \
//       [--temperature T]     (default 1.0)
//       [--top_p P]           (default 1.0)
//       [--top_k K]           (default 50)
//       [--max_new_tokens N]  (default 200)
//       [--device D]          (default CPU)
//
// Performance output after generation:
//   - Tokens generated
//   - Total latency (full pipeline round-trip)
//   - TTFT  – Time To First Token
//   - TPOT  – Time Per Output Token
//   - Throughput (tokens/s)
//   - Infer time    – pure model forward-pass time (all tokens)
//   - Sampling time – dedicated timer around sampler.sample() only:
//                     temperature scaling + top-k/top-p filtering + token draw
//   - Sampling / Infer ratio (%)

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "openvino/genai/llm_pipeline.hpp"

namespace {

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <MODEL_DIR> \"<PROMPT>\""
              << " [--temperature T] [--top_p P] [--top_k K]"
              << " [--max_new_tokens N] [--device D]\n";
}

struct Args {
    std::string model_dir;
    std::string prompt;
    std::string device        = "CPU";
    float       temperature   = 1.0f;
    float       top_p         = 1.0f;
    size_t      top_k         = 50;
    size_t      max_new_tokens = 200;
};

Args parse_args(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        throw std::runtime_error("Not enough arguments.");
    }

    Args a;
    a.model_dir = argv[1];
    a.prompt    = argv[2];

    for (int i = 3; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        std::string val = argv[i + 1];

        if      (key == "--temperature")    a.temperature    = std::stof(val);
        else if (key == "--top_p")          a.top_p          = std::stof(val);
        else if (key == "--top_k")          a.top_k          = static_cast<size_t>(std::stoul(val));
        else if (key == "--max_new_tokens") a.max_new_tokens = static_cast<size_t>(std::stoul(val));
        else if (key == "--device")         a.device         = val;
        else throw std::runtime_error("Unknown argument: " + key);
    }
    return a;
}

} // namespace

int main(int argc, char* argv[]) try {
    Args args = parse_args(argc, argv);

    std::cout << "Model  : " << args.model_dir  << "\n"
              << "Device : " << args.device      << "\n"
              << "Config : temperature=" << args.temperature
                           << "  top_p="       << args.top_p
                           << "  top_k="       << args.top_k
                           << "  max_new_tokens=" << args.max_new_tokens << "\n\n";

    ov::genai::LLMPipeline pipe(args.model_dir, args.device);

    ov::genai::GenerationConfig config;
    // temperature == 0 means greedy (argmax); only enable random sampling above that.
    config.do_sample      = (args.temperature > 0.0f);
    config.temperature    = args.temperature;
    config.top_p          = args.top_p;
    config.top_k          = args.top_k;
    config.max_new_tokens = args.max_new_tokens;

    std::cout << "Prompt : " << args.prompt << "\n\n";
    std::cout << "--- Generated text ---\n";

    // Streamer: prints each subword token as it is decoded.
    // The generate() call still returns DecodedResults (with perf_metrics)
    // even when a streamer is attached.
    auto streamer = [](std::string subword) -> ov::genai::StreamingStatus {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    // ── Model forward + sampling timing ────────────────────────────────────
    // The pipeline internally timestamps every forward call (infer_duration)
    // and the whole generate() call (generate_duration).  We also wrap the
    // call in a wall-clock timer as a cross-check for the total latency.
    auto wall_start = std::chrono::steady_clock::now();

    ov::genai::DecodedResults result = pipe.generate(args.prompt, config, streamer);

    auto wall_end     = std::chrono::steady_clock::now();
    float wall_ms     = std::chrono::duration<float, std::milli>(wall_end - wall_start).count();

    std::cout << "\n--- End of generation ---\n\n";

    // ── Performance breakdown ───────────────────────────────────────────────
    ov::genai::PerfMetrics& m = result.perf_metrics;
    // Getters call evaluate_statistics() lazily; explicit call is not needed.

    const size_t num_tokens = m.get_num_generated_tokens();

    // Pipeline-internal timers (all in milliseconds).
    const float gen_ms        = m.get_generate_duration().mean;    // full pipeline, including tok/detok
    const float infer_ms      = m.get_inference_duration().mean;   // ModelRunner::forward() (stateful: start_async…wait)
    const float pure_infer_ms = m.get_pure_infer_duration().mean;  // m_request.infer() only (CB pipeline); -1 otherwise
    const float sample_ms     = m.get_sampling_duration().mean;         // sampler.sample() only
    const float xform_ms      = m.get_logit_transform_duration().mean;  // logit_processor.apply()
    const float dist_ms       = m.get_dist_construct_duration().mean;   // discrete_distribution build
    const float draw_ms       = m.get_draw_duration().mean;             // token draw
    // logit transform sub-breakdown
    const float misc_ms       = m.get_misc_transform_duration().mean;   // EOS / structured-output
    const float penalties_ms  = m.get_penalties_duration().mean;        // rep/freq/presence penalties
    const float temp_ms       = m.get_temperature_duration().mean;      // TemperatureLogitTransform
    const float top_p_ms      = m.get_top_p_duration().mean;            // TopPFilter
    const float top_k_ms      = m.get_top_k_duration().mean;            // TopKFilter
    const float tok_ms        = m.get_tokenization_duration().mean;   // < 0 when not instrumented
    const float detok_ms      = m.get_detokenization_duration().mean; // < 0 when not instrumented

    const float ttft_ms    = m.get_ttft().mean;
    const float tpot_ms    = m.get_tpot().mean;
    const float throughput = m.get_throughput().mean;

    // Use the finest-grained infer number available for the ratio.
    const float infer_ref_ms = (pure_infer_ms > 0.f) ? pure_infer_ms : infer_ms;
    const float ratio_pct = (infer_ref_ms > 0.f) ? (sample_ms / infer_ref_ms * 100.f) : 0.f;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Performance Summary ===\n";
    std::cout << "  Tokens generated  : " << num_tokens  << " tokens\n";
    std::cout << "  Total latency     : " << gen_ms      << " ms  (pipeline)"
              << "  /  " << wall_ms << " ms  (wall-clock)\n";
    std::cout << "  TTFT              : " << ttft_ms     << " ms\n";
    std::cout << "  TPOT              : " << tpot_ms     << " ms/token\n";
    std::cout << "  Throughput        : " << throughput  << " tokens/s\n";
    std::cout << "\n";
    std::cout << "  --- Latency breakdown ---\n";
    std::cout << "  Forward           : " << infer_ms      << " ms"
              << "  (ModelRunner::forward incl. tensor packing)\n";
    if (pure_infer_ms > 0.f)
        std::cout << "  Pure inference    : " << pure_infer_ms << " ms"
                  << "  (m_request.infer() only, CB pipeline)\n";
    std::cout << "  Sampling          : " << sample_ms     << " ms"
              << "  (temp scaling + top-k/top-p filter + token draw)\n";
    if (xform_ms > 0.f || dist_ms > 0.f || draw_ms > 0.f) {
        std::cout << "    Logit transforms  : " << xform_ms << " ms"
                  << "  (temperature scaling, penalties, top-p/top-k)\n";
        if (misc_ms      > 0.f) std::cout << "      EOS/struct-out  : " << misc_ms     << " ms\n";
        if (penalties_ms > 0.f) std::cout << "      Penalties       : " << penalties_ms << " ms  (rep/freq/presence)\n";
        if (temp_ms      > 0.f) std::cout << "      Temperature     : " << temp_ms     << " ms  (expf loop, O(vocab))\n";
        if (top_p_ms     > 0.f) std::cout << "      Top-P filter    : " << top_p_ms    << " ms  (sort + prefix scan)\n";
        if (top_k_ms     > 0.f) std::cout << "      Top-K filter    : " << top_k_ms    << " ms  (partial sort)\n";
        std::cout << "    Dist. construct   : " << dist_ms  << " ms"
                  << "  (discrete_distribution build, float->double, O(vocab))\n";
        std::cout << "    Token draw        : " << draw_ms  << " ms"
                  << "  (CDF sample, O(log vocab))\n";
    }
    if (tok_ms   > 0.f)
        std::cout << "  Tokenization      : " << tok_ms   << " ms\n";
    if (detok_ms > 0.f)
        std::cout << "  Detokenization    : " << detok_ms << " ms\n";
    std::cout << "\n";
    std::cout << "  Sampling / Infer  : " << ratio_pct   << " %"
              << (pure_infer_ms > 0.f ? "  (vs pure inference)" : "  (vs forward)") << "\n";

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
