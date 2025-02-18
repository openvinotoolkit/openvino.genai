// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c_wrapper/generation_config_c.h"

#include "common_c.hpp"
#include "openvino/genai/generation_config.hpp"

#ifdef __cplusplus
OPENVINO_EXTERN_C {
#endif

    GenerationConfigHandle* CreateGenerationConfig() {
        GenerationConfigHandle* config = new GenerationConfigHandle;
        config->object = std::make_shared<ov::genai::GenerationConfig>();
        return config;
    }
    GenerationConfigHandle* CreateGenerationConfigFromJson(const char* json_path) {
        if (json_path) {
            GenerationConfigHandle* config = new GenerationConfigHandle;
            config->object = std::make_shared<ov::genai::GenerationConfig>(std::filesystem::path(json_path));
            return config;
        }
        return NULL;
    }
    void DestroyGenerationConfig(GenerationConfigHandle * config) {
        if (config) {
            delete config;
        }
    }

    // Generic
    void GenerationConfigSetMaxNewTokens(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->max_new_tokens = value;
        }
    }
    void GenerationConfigSetMaxLength(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->max_length = value;
        }
    }
    void GenerationConfigSetIgnoreEOS(GenerationConfigHandle * config, bool value) {
        if (config && config->object) {
            config->object->ignore_eos = value;
        }
    }
    void GenerationConfigSetMinNewTokens(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->min_new_tokens = value;
        }
    }
    void GenerationConfigSetEcho(GenerationConfigHandle * config, bool value) {
        if (config && config->object) {
            config->object->echo = value;
        }
    }
    void GenerationConfigSetLogProbs(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->logprobs = value;
        }
    }

    void GenerationConfigSetIncludeStopStrInOutput(GenerationConfigHandle * config, bool value) {
        if (config && config->object) {
            config->object->include_stop_str_in_output = value;
        }
    }
    void GenerationConfigSetStopStrings(GenerationConfigHandle * config, const char* strings[], size_t count) {
        if (config && config->object) {
            std::set<std::string> stopStrings;
            for (size_t i = 0; i < count; i++) {
                stopStrings.insert(strings[i]);
            }
            config->object->stop_strings = stopStrings;
        }
    }
    void GenerationConfigSetStopTokenIds(GenerationConfigHandle * config, int64_t * token_ids, size_t token_ids_num) {
        if (config && config->object) {
            std::set<int64_t> stop_token_ids;
            for (size_t i = 0; i < token_ids_num; i++) {
                stop_token_ids.insert(token_ids[i]);
            }
            config->object->stop_token_ids = stop_token_ids;
        }
    }
    // Beam Search
    void GenerationConfigSetNumBeamGroups(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->num_beam_groups = value;
        }
    }
    void GenerationConfigSetNumBeams(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->num_beams = value;
        }
    }
    void GenerationConfigSetDiversityPenalty(GenerationConfigHandle * config, float value) {
        if (config && config->object) {
            config->object->diversity_penalty = value;
        }
    }
    void GenerationConfigSetLengthPenalty(GenerationConfigHandle * config, float value) {
        if (config && config->object) {
            config->object->length_penalty = value;
        }
    }
    void GenerationConfigSetNumReturnSequences(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->num_return_sequences = value;
        }
    }
    void GenerationConfigSetNoRepeatNgramSize(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->no_repeat_ngram_size = value;
        }
    }

    void GenerationConfigSetStopCriteria(GenerationConfigHandle * config, StopCriteria value) {
        if (config && config->object) {
            config->object->stop_criteria = static_cast<ov::genai::StopCriteria>(value);
        }
    }

    void GenerationConfigSetTemperature(GenerationConfigHandle * config, float value) {
        if (config && config->object) {
            config->object->temperature = value;
        }
    }
    void GenerationConfigSetTopP(GenerationConfigHandle * config, float value) {
        if (config && config->object) {
            config->object->top_p = value;
        }
    }
    void GenerationConfigSetTopK(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->top_k = value;
        }
    }
    void GenerationConfigSetDoSample(GenerationConfigHandle * config, bool value) {
        if (config && config->object) {
            config->object->do_sample = value;
        }
    }
    void GenerationConfigSetRepetitionPenalty(GenerationConfigHandle * config, float value) {
        if (config && config->object) {
            config->object->repetition_penalty = value;
        }
    }
    void GenerationConfigSetPresencePenalty(GenerationConfigHandle * config, float value) {
        if (config && config->object) {
            config->object->presence_penalty = value;
        }
    }
    void GenerationConfigSetFrequencyPenalty(GenerationConfigHandle * config, float value) {
        if (config && config->object) {
            config->object->frequency_penalty = value;
        }
    }
    void GenerationConfigSetRngSeed(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->rng_seed = value;
        }
    }

    void GenerationConfigSetAssistantConfidenceThreshold(GenerationConfigHandle * config, float value) {
        if (config && config->object) {
            config->object->assistant_confidence_threshold = value;
        }
    }
    void GenerationConfigSetNumAssistantTokens(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->num_assistant_tokens = value;
        }
    }
    void GenerationConfigSetMaxNgramSize(GenerationConfigHandle * config, size_t value) {
        if (config && config->object) {
            config->object->max_ngram_size = value;
        }
    }

    void GenerationConfigSetEOSTokenID(GenerationConfigHandle * config, int64_t id) {
        if (config && config->object) {
            config->object->eos_token_id = id;
        }
    }

    size_t GenerationConfigGetMaxNewTokens(GenerationConfigHandle * config) {
        if (config && config->object) {
            return config->object->max_new_tokens;
        }
        return 0;
    }
    bool GenerationConfigIsGreedyDecoding(GenerationConfigHandle * config) {
        if (config && config->object) {
            return config->object->is_greedy_decoding();
        }
        return false;
    }
    bool GenerationConfigIsBeamSearch(GenerationConfigHandle * config) {
        if (config && config->object) {
            return config->object->is_beam_search();
        }
        return false;
    }
    bool GenerationConfigIsMultinomial(GenerationConfigHandle * config) {
        if (config && config->object) {
            return config->object->is_multinomial();
        }
        return false;
    }
    bool GenerationConfigIsAssistingGeneration(GenerationConfigHandle * config) {
        if (config && config->object) {
            return config->object->is_assisting_generation();
        }
        return false;
    }
    bool GenerationConfigIsPromptLookup(GenerationConfigHandle * config) {
        if (config && config->object) {
            return config->object->is_prompt_lookup();
        }
        return false;
    }
    void GenerationConfigValidate(GenerationConfigHandle * config) {
        if (config && config->object) {
            config->object->validate();
        }
    }

#ifdef __cplusplus
}
#endif