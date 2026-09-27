// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/c/ov_shape.h"
#include "openvino/c/ov_tensor.h"
#include "openvino/genai/c/perf_metrics.h"
#include "openvino/genai/c/speech_generation_config.h"

typedef struct ov_genai_text2speech_pipeline_opaque ov_genai_text2speech_pipeline;
typedef struct ov_genai_text2speech_decoded_results_opaque ov_genai_text2speech_decoded_results;

/** Create a pipeline. property_args_size counts key/value arguments and must be even. */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2speech_pipeline_create(const char* models_path,
                                                                          const char* device,
                                                                          size_t property_args_size,
                                                                          ov_genai_text2speech_pipeline** pipeline,
                                                                          ...);
OPENVINO_GENAI_C_EXPORTS void ov_genai_text2speech_pipeline_free(ov_genai_text2speech_pipeline* pipeline);

/** Generate one waveform. speaker_embedding may be NULL for models with a default voice; Kokoro requires it. */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_generate(ov_genai_text2speech_pipeline* pipeline,
                                       const char* text,
                                       const ov_tensor_t* speaker_embedding,
                                       ov_genai_text2speech_decoded_results** results);

/** Generate one waveform per text. texts must contain count non-NULL strings. */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_generate_batch(ov_genai_text2speech_pipeline* pipeline,
                                             const char* const* texts,
                                             size_t count,
                                             const ov_tensor_t* speaker_embedding,
                                             ov_genai_text2speech_decoded_results** results);

OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_get_generation_config(const ov_genai_text2speech_pipeline* pipeline,
                                                    ov_genai_speech_generation_config** config);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_set_generation_config(ov_genai_text2speech_pipeline* pipeline,
                                                    const ov_genai_speech_generation_config* config);
/** The returned shape must be released with ov_shape_free. */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_get_speaker_embedding_shape(const ov_genai_text2speech_pipeline* pipeline,
                                                          ov_shape_t* shape);

OPENVINO_GENAI_C_EXPORTS void ov_genai_text2speech_decoded_results_free(ov_genai_text2speech_decoded_results* results);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_speeches_count(const ov_genai_text2speech_decoded_results* results,
                                                        size_t* count);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_output_sample_rate(const ov_genai_text2speech_decoded_results* results,
                                                            uint32_t* sample_rate);
/** Copies the indexed waveform to a new tensor. Release it with ov_tensor_free. */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_speech_at(const ov_genai_text2speech_decoded_results* results,
                                                   size_t index,
                                                   ov_tensor_t** speech);
/** Copies base performance metrics. Release with ov_genai_text2speech_decoded_results_perf_metrics_free. */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_perf_metrics(const ov_genai_text2speech_decoded_results* results,
                                                      ov_genai_perf_metrics** metrics);
OPENVINO_GENAI_C_EXPORTS void ov_genai_text2speech_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics);
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_num_generated_samples(const ov_genai_text2speech_decoded_results* results,
                                                               size_t* count);
