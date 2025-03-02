// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/generation_config_c.h"

#include "types_c.h"
#include "openvino/genai/generation_config.hpp"

ov_genai_generation_config* ov_genai_generation_config_create() {
    ov_genai_generation_config* config = new ov_genai_generation_config;
    config->object = std::make_shared<ov::genai::GenerationConfig>();
    return config;
}
ov_genai_generation_config* ov_genai_generation_config_create_from_json(const char* json_path) {
    if (json_path) {
        ov_genai_generation_config* config = new ov_genai_generation_config;
        config->object = std::make_shared<ov::genai::GenerationConfig>(std::filesystem::path(json_path));
        return config;
    }
    return NULL;
}
void ov_genai_generation_config_free(ov_genai_generation_config* config) {
    if (config) {
        delete config;
    }
}

void ov_genai_generation_config_set_max_new_tokens(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->max_new_tokens = value;
    }
}
void ov_genai_generation_config_set_max_length(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->max_length = value;
    }
}
void ov_genai_generation_config_set_ignore_eos(ov_genai_generation_config* config, bool value) {
    if (config && config->object) {
        config->object->ignore_eos = value;
    }
}
void ov_genai_generation_config_set_min_new_tokens(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->min_new_tokens = value;
    }
}
void ov_genai_generation_config_set_echo(ov_genai_generation_config* config, bool value) {
    if (config && config->object) {
        config->object->echo = value;
    }
}
void ov_genai_generation_config_set_logprobs(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->logprobs = value;
    }
}

void ov_genai_generation_config_set_include_stop_str_in_output(ov_genai_generation_config* config, bool value) {
    if (config && config->object) {
        config->object->include_stop_str_in_output = value;
    }
}
void ov_genai_generation_config_set_stop_strings(ov_genai_generation_config* config, const char* strings[], size_t count) {
    if (config && config->object) {
        std::set<std::string> stopStrings;
        for (size_t i = 0; i < count; i++) {
            stopStrings.insert(strings[i]);
        }
        config->object->stop_strings = stopStrings;
    }
}
void ov_genai_generation_config_set_stop_token_ids(ov_genai_generation_config* config, int64_t* token_ids, size_t token_ids_num) {
    if (config && config->object) {
        std::set<int64_t> stop_token_ids;
        for (size_t i = 0; i < token_ids_num; i++) {
            stop_token_ids.insert(token_ids[i]);
        }
        config->object->stop_token_ids = stop_token_ids;
    }
}
void ov_genai_generation_config_set_num_beam_groups(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->num_beam_groups = value;
    }
}
void ov_generation_config_set_num_beams(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->num_beams = value;
    }
}
void ov_genai_generation_config_set_diversity_penalty(ov_genai_generation_config* config, float value) {
    if (config && config->object) {
        config->object->diversity_penalty = value;
    }
}
void ov_genai_generation_config_set_length_penalty(ov_genai_generation_config* config, float value) {
    if (config && config->object) {
        config->object->length_penalty = value;
    }
}
void ov_genai_generation_config_num_return_sequences(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->num_return_sequences = value;
    }
}
void ov_genai_generation_config_set_no_repeat_ngram_size(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->no_repeat_ngram_size = value;
    }
}

void ov_genai_generation_config_set_stop_criteria(ov_genai_generation_config* config, StopCriteria value) {
    if (config && config->object) {
        config->object->stop_criteria = static_cast<ov::genai::StopCriteria>(value);
    }
}

void ov_genai_generation_config_set_temperature(ov_genai_generation_config* config, float value) {
    if (config && config->object) {
        config->object->temperature = value;
    }
}
void ov_genai_generation_config_set_top_p(ov_genai_generation_config* config, float value) {
    if (config && config->object) {
        config->object->top_p = value;
    }
}
void ov_genai_generation_config_set_top_k(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->top_k = value;
    }
}
void ov_genai_generation_config_set_do_sample(ov_genai_generation_config* config, bool value) {
    if (config && config->object) {
        config->object->do_sample = value;
    }
}
void ov_genai_generation_config_set_repetition_penalty(ov_genai_generation_config* config, float value) {
    if (config && config->object) {
        config->object->repetition_penalty = value;
    }
}
void ov_genai_generation_config_set_presence_penalty(ov_genai_generation_config* config, float value) {
    if (config && config->object) {
        config->object->presence_penalty = value;
    }
}
void ov_genai_generation_config_set_frequency_penalty(ov_genai_generation_config* config, float value) {
    if (config && config->object) {
        config->object->frequency_penalty = value;
    }
}
void ov_genai_generation_config_set_rng_seed(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->rng_seed = value;
    }
}

void ov_genai_generation_config_set_assistant_confidence_threshold(ov_genai_generation_config* config, float value) {
    if (config && config->object) {
        config->object->assistant_confidence_threshold = value;
    }
}
void ov_genai_generation_config_set_num_assistant_tokens(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->num_assistant_tokens = value;
    }
}
void ov_genai_generation_config_set_max_ngram_size(ov_genai_generation_config* config, size_t value) {
    if (config && config->object) {
        config->object->max_ngram_size = value;
    }
}

void ov_genai_generation_config_set_eos_token_id(ov_genai_generation_config* config, int64_t id) {
    if (config && config->object) {
        config->object->eos_token_id = id;
    }
}

size_t ov_genai_generation_config_get_max_new_tokens(ov_genai_generation_config* config) {
    if (config && config->object) {
        return config->object->max_new_tokens;
    }
    return 0;
}
bool ov_genai_generation_config_is_greedy_decoding(ov_genai_generation_config* config) {
    if (config && config->object) {
        return config->object->is_greedy_decoding();
    }
    return false;
}
bool ov_genai_generation_config_is_beam_search(ov_genai_generation_config* config) {
    if (config && config->object) {
        return config->object->is_beam_search();
    }
    return false;
}
bool ov_genai_generation_config_is_multinomial(ov_genai_generation_config* config) {
    if (config && config->object) {
        return config->object->is_multinomial();
    }
    return false;
}
bool ov_genai_generation_config_is_assisting_generation(ov_genai_generation_config* config) {
    if (config && config->object) {
        return config->object->is_assisting_generation();
    }
    return false;
}
bool ov_genai_generation_config_is_prompt_lookup(ov_genai_generation_config* config) {
    if (config && config->object) {
        return config->object->is_prompt_lookup();
    }
    return false;
}
void ov_genai_generation_config_validate(ov_genai_generation_config* config) {
    if (config && config->object) {
        config->object->validate();
    }
}
