# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Samples do not need to be built for NPM package
if(CPACK_GENERATOR STREQUAL "NPM")
    return()
endif()

add_subdirectory(cpp/beam_search_causal_lm)
add_subdirectory(cpp/benchmark_genai)
add_subdirectory(cpp/chat_sample)
add_subdirectory(cpp/greedy_causal_lm)
add_subdirectory(cpp/lora_greedy_causal_lm)
add_subdirectory(cpp/multinomial_causal_lm)
add_subdirectory(cpp/prompt_lookup_decoding_lm)
add_subdirectory(cpp/speculative_decoding_lm)
add_subdirectory(cpp/text2image)
add_subdirectory(cpp/visual_language_chat)
add_subdirectory(cpp/whisper_speech_recognition)

install(FILES requirements.txt DESTINATION samples
        COMPONENT cpp_samples_genai)

install(DIRECTORY
            cpp/beam_search_causal_lm
            cpp/benchmark_genai
            cpp/chat_sample
            cpp/greedy_causal_lm
            cpp/lora_greedy_causal_lm
            cpp/multinomial_causal_lm
            # Don't install prompt_lookup_decoding_lm because it doesn't use openvino_genai library and is not verified yet.
            cpp/speculative_decoding_lm
            cpp/text2image
            cpp/visual_language_chat
            cpp/whisper_speech_recognition
        DESTINATION samples/cpp COMPONENT cpp_samples_genai)

install(DIRECTORY
            python/beam_search_causal_lm
            python/benchmark_genai
            python/chat_sample
            python/greedy_causal_lm
            python/multinomial_causal_lm
            python/speculative_decoding_lm
            python/text2image
            python/visual_language_chat
            python/whisper_speech_recognition
        DESTINATION samples/python COMPONENT cpp_samples_genai
        USE_SOURCE_PERMISSIONS)
