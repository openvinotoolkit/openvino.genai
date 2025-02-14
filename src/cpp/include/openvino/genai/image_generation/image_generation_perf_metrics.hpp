#pragma once

#include <vector>
#include <chrono>
#include <map>
#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/perf_metrics.hpp"

namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS RawImageGenerationPerfMetrics {
    std::vector<MicroSeconds> unet_inference_durations; // unet inference durations for each step
    std::vector<MicroSeconds> transformer_inference_durations; // transformer inference durations for each step
    std::vector<MicroSeconds> iteration_durations;  //  durations of each step
};

struct OPENVINO_GENAI_EXPORTS ImageGenerationPerfMetrics {
    float load_time; // model load time (includes reshape & read_model time), ms
    float generate_duration; // duration of method generate(...), ms

    MeanStdPair iteration_duration; // Mean-Std time of one generation iteration, ms
    std::map<std::string, float> encoder_inference_duration; // inference durations for each encoder, ms
    MeanStdPair unet_inference_duration; // inference duration for unet model, should be filled with zeros if we don't have unet, ms
    MeanStdPair transformer_inference_duration; // inference duration for transformer model, should be filled with zeros if we don't have transformer, ms
    float vae_encoder_inference_duration; // inference duration of vae_encoder model, should be filled with zeros if we don't use it, ms
    float vae_decoder_inference_duration; // inference duration of vae_decoder model, ms

    bool m_evaluated = false;

    RawImageGenerationPerfMetrics raw_metrics;

    void clean_up();
    void evaluate_statistics();

    MeanStdPair get_unet_infer_meanstd();
    MeanStdPair get_transformer_infer_meanstd();
    MeanStdPair get_iteration_meanstd();
    float get_encoder_infer_duration();
    float get_decoder_infer_duration();
    float get_all_infer_duration();
    float get_load_time();
    float get_generate_duration();
    void get_iteration_duration(float &first_iter, float &other_iter_avg);
    void get_unet_infer_duration(float &first_infer, float &other_infer_avg);
    void get_transformer_infer_duration(float &first_infer, float &other_infer_avg);
};
}