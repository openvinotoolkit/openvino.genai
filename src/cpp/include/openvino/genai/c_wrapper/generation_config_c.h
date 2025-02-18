// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for  ov::genai::GenerationConfig class.
 *
 * @file generation_config_c.h
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "../visibility.hpp"

#ifdef __cplusplus
OPENVINO_EXTERN_C {
#endif

#include "stdio.h"

    typedef enum { EARLY, HEURISTIC, NEVER } StopCriteria;
    /**
     * @struct GenerationConfigHandle
     * @brief type define GenerationConfigHandle from OpaqueGenerationConfig
     */
    typedef struct OpaqueGenerationConfig GenerationConfigHandle;

    OPENVINO_GENAI_EXPORTS GenerationConfigHandle* CreateGenerationConfig();
    OPENVINO_GENAI_EXPORTS GenerationConfigHandle* CreateGenerationConfigFromJson(const char* json_path);
    OPENVINO_GENAI_EXPORTS void DestroyGenerationConfig(GenerationConfigHandle * handle);

    // Generic
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetMaxNewTokens(GenerationConfigHandle * handle, size_t value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetMaxLength(GenerationConfigHandle * config, size_t value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetIgnoreEOS(GenerationConfigHandle * config, bool value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetMinNewTokens(GenerationConfigHandle * config, size_t value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetEcho(GenerationConfigHandle * config, bool value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetLogProbs(GenerationConfigHandle * config, size_t value);

    OPENVINO_GENAI_EXPORTS void GenerationConfigSetStopStrings(GenerationConfigHandle * config,
                                                               const char* strings[],
                                                               size_t count);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetIncludeStopStrInOutput(GenerationConfigHandle * config, bool value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetStopTokenIds(GenerationConfigHandle * config,
                                                                int64_t * token_ids,
                                                                size_t token_ids_num);

    // Beam Search
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNumBeamGroups(GenerationConfigHandle * config, size_t value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNumBeams(GenerationConfigHandle * config, size_t value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetDiversityPenalty(GenerationConfigHandle * config, float value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetLengthPenalty(GenerationConfigHandle * config, float value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNumReturnSequences(GenerationConfigHandle * config, size_t value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNoRepeatNgramSize(GenerationConfigHandle * config, size_t value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetStopCriteria(GenerationConfigHandle * config, StopCriteria value);

    OPENVINO_GENAI_EXPORTS void GenerationConfigSetTemperature(GenerationConfigHandle * config, float value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetTopP(GenerationConfigHandle * config, float value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetTopK(GenerationConfigHandle * config, size_t value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetDoSample(GenerationConfigHandle * config, bool value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetRepetitionPenalty(GenerationConfigHandle * config, float value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetPresencePenalty(GenerationConfigHandle * config, float value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetFrequencyPenalty(GenerationConfigHandle * config, float value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetRngSeed(GenerationConfigHandle * config, size_t value);

    OPENVINO_GENAI_EXPORTS void GenerationConfigSetAssistantConfidenceThreshold(GenerationConfigHandle * config,
                                                                                float value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNumAssistantTokens(GenerationConfigHandle * config, size_t value);
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetMaxNgramSize(GenerationConfigHandle * config, size_t value);

    OPENVINO_GENAI_EXPORTS void GenerationConfigSetEOSTokenID(GenerationConfigHandle * config, int64_t id);

    OPENVINO_GENAI_EXPORTS size_t GenerationConfigGetMaxNewTokens(GenerationConfigHandle * config);
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsGreedyDecoding(GenerationConfigHandle * config);
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsBeamSearch(GenerationConfigHandle * config);
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsMultinomial(GenerationConfigHandle * config);
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsAssistingGeneration(GenerationConfigHandle * config);
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsPromptLookup(GenerationConfigHandle * config);
    OPENVINO_GENAI_EXPORTS void GenerationConfigValidate(GenerationConfigHandle * config);

#ifdef __cplusplus
}
#endif