/**
 * @brief C API for speech generation performance metrics
 * @file speech_generation_perf_metrics.h
 */

#pragma once

#include "perf_metrics.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure containing speech generation performance metrics
 */
typedef struct speech_generation_perf_metrics {
    perf_metrics_t base;              /**< Base performance metrics */
    int num_generated_samples;        /**< Number of generated audio samples */
} speech_generation_perf_metrics_t;

/**
 * @brief Creates a speech generation performance metrics object
 * @param[out] metrics Pointer to store the created metrics
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_perf_metrics_create(speech_generation_perf_metrics_t** metrics);

/**
 * @brief Destroys a speech generation performance metrics object
 * @param[in] metrics Metrics object to destroy
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_perf_metrics_destroy(speech_generation_perf_metrics_t* metrics);

/**
 * @brief Gets the number of generated audio samples
 * @param[in] metrics Performance metrics object
 * @param[out] value Pointer to store the number of generated samples
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_perf_metrics_get_num_generated_samples(
    const speech_generation_perf_metrics_t* metrics,
    int* value);

/**
 * @brief Gets the generation duration in milliseconds
 * @param[in] metrics Performance metrics object
 * @param[out] value Pointer to store the duration
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_perf_metrics_get_generate_duration(
    const speech_generation_perf_metrics_t* metrics,
    float* value);

/**
 * @brief Gets the throughput (samples per second)
 * @param[in] metrics Performance metrics object
 * @param[out] value Pointer to store the throughput
 * @return 0 if successful, non-zero otherwise
 */
OPENVINO_GENAI_C_API int speech_generation_perf_metrics_get_throughput(
    const speech_generation_perf_metrics_t* metrics,
    float* value);

#ifdef __cplusplus
}  // extern "C"
#endif
