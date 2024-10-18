# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(cpp/beam_search_causal_lm)
add_subdirectory(cpp/chat_sample)
add_subdirectory(cpp/continuous_batching_accuracy)
add_subdirectory(cpp/continuous_batching_benchmark)
add_subdirectory(cpp/greedy_causal_lm)
add_subdirectory(cpp/lora_greedy_causal_lm)
add_subdirectory(cpp/multinomial_causal_lm)
add_subdirectory(cpp/prompt_lookup_decoding_lm)
add_subdirectory(cpp/visual_language_chat)
add_subdirectory(cpp/speculative_decoding_lm)
add_subdirectory(cpp/benchmark_genai)
add_subdirectory(cpp/whisper_speech_recognition)
add_subdirectory(cpp/text2image)

install(FILES requirements.txt DESTINATION samples
        COMPONENT cpp_samples_genai)

install(DIRECTORY
            cpp/beam_search_causal_lm
            cpp/chat_sample
            cpp/greedy_causal_lm
            cpp/multinomial_causal_lm
            # Don't install prompt_lookup_decoding_lm and speculative_decoding_lm because they don't use openvino_genai library and arent verifyed yet.
            # Don't install continuous_batching_accuracy and continuous_batching_benchmark because CB isn't ready.
            cpp/visual_language_chat
            cpp/whisper_speech_recognition
            cpp/text2image
            cpp/lora_greedy_causal_lm
            cpp/speculative_decoding_lm
        DESTINATION samples/cpp COMPONENT cpp_samples_genai)

install(DIRECTORY
            python/beam_search_causal_lm
            python/chat_sample
            python/greedy_causal_lm
            python/multinomial_causal_lm
            python/visual_language_chat
            python/whisper_speech_recognition
            # python/speculative_decoding_lm
            # python/text2image
        DESTINATION samples/python COMPONENT cpp_samples_genai
        USE_SOURCE_PERMISSIONS)
