/**
 * @brief C API for text-to-speech pipeline
 * @file text2speech_pipeline.h
 */

#pragma once

#include "speech_generation_config.h"
#include "speech_generation_perf_metrics.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle for text-to-speech pipeline
 */
typedef struct text2speech_pipeline_t* text2speech_pipeline_handle_t;

/**
 * @brief Structure containing generated speech data
 */
typedef struct speech_data {
    float* samples;           /**< Array of audio samples */
    size_t num_samples;      /**< Number of samples in the array */
    int sample_rate;         /**< Sample rate of the audio (e.g., 16000 Hz) */
} speech_data_t;

/**
 * @brief Structure containing text-to-speech generation results
 */
typedef struct text2speech_decoded_results {
    speech_data_t* speeches;                      /**< Array of generated speech data */
    size_t num_speeches;                          /**< Number of speeches in the array */
    speech_generation_perf_metrics_t perf_metrics; /**< Performance metrics for the generation */
} text2speech_decoded_results_t;

/**
 * @brief Creates a text-to-speech pipeline
 * @param[out] pipeline Pointer to store the created pipeline handle
 * @param[in] models_path Path to the directory containing model files
 * @param[in] device Device to run inference on (e.g., "CPU", "GPU")
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int text2speech_pipeline_create(text2speech_pipeline_handle_t* pipeline,
                                                    const char* models_path,
                                                    const char* device);

/**
 * @brief Destroys a text-to-speech pipeline
 * @param[in] pipeline Pipeline handle to destroy
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int text2speech_pipeline_destroy(text2speech_pipeline_handle_t pipeline);

/**
 * @brief Generates speech from a single text input
 * @param[in] pipeline Pipeline handle
 * @param[in] text Input text to convert to speech
 * @param[in] speaker_embedding Optional speaker embedding tensor (can be NULL)
 * @param[in] speaker_embedding_size Size of the speaker embedding tensor
 * @param[out] results Pointer to store the generation results
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int text2speech_pipeline_generate(text2speech_pipeline_handle_t pipeline,
                                                      const char* text,
                                                      const float* speaker_embedding,
                                                      size_t speaker_embedding_size,
                                                      text2speech_decoded_results_t** results);

/**
 * @brief Generates speech from multiple text inputs
 * @param[in] pipeline Pipeline handle
 * @param[in] texts Array of input texts
 * @param[in] num_texts Number of input texts
 * @param[in] speaker_embedding Optional speaker embedding tensor (can be NULL)
 * @param[in] speaker_embedding_size Size of the speaker embedding tensor
 * @param[out] results Pointer to store the generation results
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int text2speech_pipeline_generate_batch(text2speech_pipeline_handle_t pipeline,
                                                            const char** texts,
                                                            size_t num_texts,
                                                            const float* speaker_embedding,
                                                            size_t speaker_embedding_size,
                                                            text2speech_decoded_results_t** results);

/**
 * @brief Gets the generation configuration from the pipeline
 * @param[in] pipeline Pipeline handle
 * @param[out] config Pointer to store the configuration handle
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int text2speech_pipeline_get_generation_config(text2speech_pipeline_handle_t pipeline,
                                                                   speech_generation_config_handle_t* config);

/**
 * @brief Sets the generation configuration for the pipeline
 * @param[in] pipeline Pipeline handle
 * @param[in] config Configuration handle to set
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int text2speech_pipeline_set_generation_config(text2speech_pipeline_handle_t pipeline,
                                                                   speech_generation_config_handle_t config);

/**
 * @brief Destroys a text-to-speech decoded results object
 * @param[in] results Results object to destroy
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int text2speech_decoded_results_destroy(text2speech_decoded_results_t* results);

#ifdef __cplusplus
}  // extern "C"
#endif
