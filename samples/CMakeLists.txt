# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(cpp/beam_search_causal_lm)
add_subdirectory(cpp/chat_sample)
add_subdirectory(cpp/greedy_causal_lm)
add_subdirectory(cpp/multinomial_causal_lm)
add_subdirectory(cpp/prompt_lookup_decoding_lm)
add_subdirectory(cpp/speculative_decoding_lm)

install(FILES requirements.txt DESTINATION samples
        COMPONENT cpp_samples_genai)

install(DIRECTORY
            cpp/beam_search_causal_lm
            cpp/chat_sample
            cpp/greedy_causal_lm
            cpp/multinomial_causal_lm
            # Don't install prompt_lookup_decoding_lm and speculative_decoding_lm because they don't use openvino_genai library and arent verifyed yet.
        DESTINATION samples/cpp COMPONENT cpp_samples_genai)

install(DIRECTORY
            python/beam_search_causal_lm
            python/chat_sample
            python/greedy_causal_lm
            python/multinomial_causal_lm
        DESTINATION samples/python COMPONENT cpp_samples_genai
        USE_SOURCE_PERMISSIONS)

if(ENABLE_CONTINUOUS_BATCHING AND ENABLE_APPS)
    add_subdirectory(cpp/accuracy_sample)
    add_subdirectory(cpp/throughput_benchmark)
endif()
