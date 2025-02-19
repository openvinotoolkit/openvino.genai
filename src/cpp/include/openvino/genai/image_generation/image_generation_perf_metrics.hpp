// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <chrono>
#include <map>
#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/perf_metrics.hpp"

namespace ov::genai {

struct RawImageGenerationPerfMetrics {
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

    MeanStdPair get_unet_infer_duration();
    MeanStdPair get_transformer_infer_duration();
    MeanStdPair get_iteration_duration();
    float get_vae_encoder_infer_duration() const;
    float get_vae_decoder_infer_duration() const;
    std::map<std::string, float> get_text_encoder_infer_duration() const;
    float get_inference_duration();
    float get_load_time() const;
    float get_generate_duration();
    void get_first_and_other_iter_duration(float& first_iter, float& other_iter_avg);
    void get_first_and_other_unet_infer_duration(float& first_infer, float& other_infer_avg);
    void get_first_and_other_trans_infer_duration(float& first_infer, float& other_infer_avg);
};
}