#pragma once

#include <vector>
#include <chrono>
#include <map>
#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/perf_metrics.hpp"

namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS RawImageGenerationPerfMetrics {
    std::vector<MicroSeconds> unet_inference_durations; // unet durations for each step
    std::vector<MicroSeconds> transformer_inference_durations; // transformer durations for each step
    std::vector<MicroSeconds> iteration_durations;  //  durations of each step
};

struct OPENVINO_GENAI_EXPORTS ImageGenerationPerfMetrics {
    float load_time; // model load time (includes reshape & read_model time)
    float generate_duration; // duration of method generate(...)

    MeanStdPair iteration_duration; // Mean-Std time of one generation iteration
    std::map<std::string, float> encoder_inference_duration; // inference durations for each encoder
    MeanStdPair unet_inference_duration; // inference duration for unet model, should be filled with zeros if we don't have unet
    MeanStdPair transformer_inference_duration; // inference duration for transformer model, should be filled with zeros if we don't have transformer
    float vae_encoder_inference_duration; // inference duration of vae_encoder model, should be filled with zeros if we don't use it
    float vae_decoder_inference_duration; // inference duration of  vae_decoder model

    bool m_evaluated = false;

    RawImageGenerationPerfMetrics raw_metrics;

    void clean_up();
    void evaluate_statistics();

    MeanStdPair get_unet_inference_duration();

    MeanStdPair get_transformer_inference_duration();
    MeanStdPair get_iteration_duration();

    float get_inference_total_duration();

};
}